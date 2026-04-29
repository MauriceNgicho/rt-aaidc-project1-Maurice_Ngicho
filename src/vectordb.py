import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import chromadb
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL_NAME", "models/gemini-embedding-001"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.google_embeddings = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # Overlap to maintain context between chunks
            separators=["\n\n", "\n", " ", ""],  # Try to split on paragraphs, then lines, then spaces
        )
        chunks = splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents.
        """

        print(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc_idx, document in enumerate(documents):
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            chunks = self.chunk_text(content)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                all_chunks.append(chunk)
                all_metadatas.append({**metadata, "chunk_index": chunk_idx, "doc_index": doc_idx})
                all_ids.append(chunk_id)

        if not all_chunks:
            print("No chunks to add to the vector database.")
            return
        print(f"Encoding {len(all_chunks)} chunks...")
        embeddings = self.google_embeddings.embed_documents(all_chunks)  # Get embeddings for all chunks

        self.collection.upsert(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids,
        )

        print(f"Documents added to vector database ({len(all_chunks)} chunks total).")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        query_embedding = self.google_embeddings.embed_query(query)  # Get the single embedding vector

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),  # Ensure n_results does not exceed collection size
            include=["documents", "metadatas", "distances"],
        )
        return {
            "documents": results.get("documents", [[]])[0],  # Get the first (and only) query result
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
            "ids": results.get("ids", [[]])[0],
        }
