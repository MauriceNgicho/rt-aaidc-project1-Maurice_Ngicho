# rt-aaidc-project1-template

A **Retrieval-Augmented Generation (RAG) assistant** built with LangChain, ChromaDB, and support for multiple LLM providers. The assistant answers questions grounded strictly in a curated knowledge base of LangChain and LangGraph technical documentation — refusing to answer questions outside its context.

---

## Features

- **Multi-provider LLM support** — OpenAI, Groq, and Google Gemini, selected automatically based on available API keys
- **Google Gemini embeddings** — Uses `models/gemini-embedding-001` for document and query embedding
- **ChromaDB vector store** — Persistent local vector database with similarity search
- **Strict context grounding** — Prompt-engineered to refuse out-of-context questions rather than hallucinate
- **Recursive text chunking** — Documents split with overlap to preserve context across chunk boundaries
- **Modular architecture** — `VectorDB` and `RAGAssistant` are cleanly separated for easy extension

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        INDEXING                             │
│                                                             │
│  data/*.txt  →  Chunking  →  Embedding  →  ChromaDB        │
│  (documents)    (1500 chars)  (Gemini)      (vector store)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        QUERYING                             │
│                                                             │
│  User Question  →  Embed Query  →  Similarity Search        │
│       ↓                                  ↓                  │
│  LLM Answer     ←  Prompt + Context  ←  Top-K Chunks       │
└─────────────────────────────────────────────────────────────┘
```

1. **Load** — `.txt` files are read from the `data/` directory
2. **Chunk** — Each document is split into overlapping chunks using `RecursiveCharacterTextSplitter`
3. **Embed** — Chunks are converted to vector embeddings using Google Gemini
4. **Store** — Embeddings are stored in a persistent ChromaDB collection
5. **Retrieve** — At query time, the user's question is embedded and the top-K most similar chunks are retrieved
6. **Generate** — Retrieved chunks are injected into a prompt as context, and the LLM generates a grounded answer

---

## Project Structure

```
rt-aaidc-project1-template/
├── data/                          # Knowledge base documents (.txt)
│   ├── langchain_introduction.txt
│   ├── langchain_lcel.txt
│   ├── langchain_chat_models.txt
│   ├── langchain_vectorstores.txt
│   ├── langchain_retrievers.txt
│   ├── langchain_text_splitters.txt
│   ├── langchain_rag_tutorial.txt
│   ├── langgraph_concepts.txt
│   └── langgraph_quickstart.txt
├── src/
│   ├── app.py                     # RAGAssistant class and main entry point
│   ├── vectordb.py                # VectorDB class (ChromaDB + embeddings)
│   └── chroma_db/                 # Persistent ChromaDB storage (auto-generated)
├── .env                           # API keys and configuration (not committed)
├── requirements.txt               # Full dependencies
├── lean_requirements.txt          # Minimal dependencies
└── README.md
```

---

## Supported LLM Providers

The assistant checks for API keys in this order and uses the first available:

| Priority | Provider | Model (default) | Notes |
|----------|----------|-----------------|-------|
| 1st | OpenAI | `gpt-4o-mini` | Best quality, requires billing |
| 2nd | Groq | `llama-3.1-8b-instant` | Free tier, fast inference |
| 3rd | Google Gemini | `gemini-2.0-flash` | Free tier available |

> Embeddings always use **Google Gemini** (`models/gemini-embedding-001`) regardless of which LLM is selected.

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- At least one API key: [OpenAI](https://platform.openai.com/api-keys), [Groq](https://console.groq.com/keys), or [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rt-aaidc-project1-template.git
cd rt-aaidc-project1-template
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
# OpenAI (optional)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini

# Groq (optional)
GROQ_API_KEY=your-groq-key
GROQ_MODEL=llama-3.1-8b-instant

# Google Gemini (required for embeddings)
GOOGLE_API_KEY=your-google-key
GOOGLE_MODEL=gemini-2.0-flash

# Embedding model
EMBEDDING_MODEL_NAME=models/gemini-embedding-001

# ChromaDB
CHROMA_COLLECTION_NAME=rag_documents
```

> At minimum, `GOOGLE_API_KEY` is required for embeddings. Add at least one LLM key (`OPENAI_API_KEY`, `GROQ_API_KEY`, or `GOOGLE_API_KEY`) to power the answer generation.

### 5. Generate the knowledge base

```bash
python scripts/download_data.py
```

This writes 9 curated `.txt` files covering LangChain and LangGraph topics into the `data/` directory.

---

## How to Run

```bash
cd src
python app.py
```

The assistant will load documents, build the vector index, and start an interactive prompt:

```
Initializing RAG Assistant...
Using Groq model: llama-3.1-8b-instant
Vector database initialized with collection: rag_documents
RAG Assistant initialized successfully

Loading documents...
Loaded 9 sample documents
Documents added to vector database (21 chunks total).

Enter a question or 'quit' to exit:
```

Type a question and press Enter. Type `quit` to exit.

---

## Example Questions & Outputs

**In-context question:**
```
Enter a question or 'quit' to exit: What is LCEL and how do I use the pipe operator?

LCEL (LangChain Expression Language) is a declarative way to compose chains
using the pipe operator (|). Basic syntax:

    chain = prompt | llm | output_parser

Each component implements the Runnable interface, which provides invoke(),
stream(), batch(), and their async equivalents.
```

**Code generation:**
```
Enter a question or 'quit' to exit: How does LangGraph checkpointing work?

Checkpointing is done using the MemorySaver class from langgraph.checkpoint.memory.

    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "session-1"}}
    result = app.invoke(initial_state, config=config)

The state is automatically restored when you resume the same thread.
```

**Out-of-context question (correct refusal):**
```
Enter a question or 'quit' to exit: What is the name of the Kenyan President?

I don't have enough information to answer that question.
```

---

## Known Limitations

- **Static knowledge base** — Documents are loaded once at startup. Adding new documents requires restarting the app and re-indexing.
- **No conversation memory** — Each question is answered independently; the assistant does not retain previous turns.
- **Google Gemini free tier rate limits** — Embedding large document sets (500+ chunks) may hit quota limits on the free tier.
- **Single data directory** — Only `.txt` files in `data/` are loaded; no support for PDFs or URLs out of the box.
- **No source citation** — Answers do not indicate which document chunk they came from.

---

## Future Improvements

- [ ] Add conversation memory for multi-turn dialogue
- [ ] Support PDF and Markdown file ingestion
- [ ] Display source citations alongside answers
- [ ] Add a web UI (Streamlit or FastAPI + frontend)
- [ ] Implement streaming responses for real-time token output
- [ ] Support reranking for improved retrieval precision
- [ ] Add evaluation pipeline using RAGAS metrics (faithfulness, context recall, etc.)
- [ ] Dockerize the application for easier deployment

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain 0.3 |
| Vector Store | ChromaDB 1.0 |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| LLM | OpenAI / Groq / Google Gemini |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Environment | Python 3.11, python-dotenv |

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.