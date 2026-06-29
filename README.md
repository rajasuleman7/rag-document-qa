# DocMind — RAG Document Q&A System

Upload any PDF or text document and ask questions in natural language. Answers are grounded strictly in your document, with source citations showing exactly which page and passage was used to construct each response.

Built with **LangChain**, **OpenAI embeddings**, **FAISS vector search**, and **Streamlit**.

---

## Demo

![DocMind Screenshot](https://i.imgur.com/placeholder.png)

> Ask "What are the key findings?" or "Summarise section 3" and get a precise, cited answer — not hallucination.

---

## Features

- **Upload any PDF or TXT** — no size limits enforced by the model, up to 20 MB
- **Intelligent chunking** — `RecursiveCharacterTextSplitter` with 800-token chunks and 150-token overlap preserves sentence boundaries
- **OpenAI embeddings** — `text-embedding-3-small` converts each chunk to a 1536-dim vector
- **FAISS vector search** — sub-millisecond similarity search across thousands of chunks
- **MMR retrieval** — Maximal Marginal Relevance reduces redundant retrieved chunks
- **Conversational memory** — remembers the last 6 exchanges; ask follow-up questions naturally
- **Source citations** — every answer shows which page(s) and excerpts were retrieved
- **Vectorstore caching** — documents are re-embedded only when content changes (SHA-256 hash check)
- **Model selector** — choose GPT-3.5 Turbo, GPT-4o Mini, or GPT-4o
- **Suggested questions** — auto-generated starter questions from the document content
- **Export chat** — download the full Q&A session as a Markdown file
- **Dark theme UI** — fully styled Streamlit interface with custom CSS

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| RAG Framework | LangChain 0.3+ |
| LLM | OpenAI GPT-3.5 / GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| PDF Parsing | PyPDF |
| UI | Streamlit 1.38 |
| Memory | `ConversationBufferWindowMemory` (k=6) |

---

## How RAG Works — Architecture

```
User Question
     │
     ▼
 Embed question
 (text-embedding-3-small)
     │
     ▼
 FAISS vector search
 (MMR, top-k=5 chunks)
     │
     ▼
 Retrieved chunks + chat history
     │
     ▼
 Prompt assembly
 (system + context + history + question)
     │
     ▼
 ChatOpenAI (GPT-3.5 / 4o)
     │
     ▼
 Answer + Source citations
```

**At indexing time:**
```
Document (PDF/TXT)
     │
     ▼
 Text extraction (PyPDF / TextLoader)
     │
     ▼
 RecursiveCharacterTextSplitter
 (chunk_size=800, overlap=150)
     │
     ▼
 OpenAI embeddings per chunk
     │
     ▼
 FAISS index (saved to disk)
```

---

## Project Structure

```
rag-document-qa/
├── app.py                          # Streamlit UI — main entry point
├── src/
│   ├── document_processor.py       # Load, chunk, embed, FAISS build/load
│   ├── qa_chain.py                 # ConversationalRetrievalChain + source formatting
│   ├── utils.py                    # Token counting, suggested Qs, chat export
│   └── config.py                   # Model options, chunk params, UI constants
├── tests/
│   ├── test_processor.py           # Unit tests for chunking, hashing, loaders
│   └── test_utils.py               # Unit tests for utility functions
├── sample_docs/
│   └── sample_research_paper.txt   # Sample document to test with immediately
├── vectorstore/                    # Cached FAISS indexes (git-ignored)
├── .streamlit/
│   └── config.toml                 # Dark theme configuration
├── .env.example
└── requirements.txt
```

---

## Setup and Running

### 1. Clone and install

```bash
git clone https://github.com/rajasuleman7/rag-document-qa.git
cd rag-document-qa
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key:  OPENAI_API_KEY=sk-...
```

Or enter it directly in the app sidebar — no file needed.

### 3. Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### 4. Try it immediately

Upload `sample_docs/sample_research_paper.txt` and ask:

- *"What methodology was used?"*
- *"What percentage improvement did RAG show over keyword search?"*
- *"What are the key findings about chunk size?"*
- *"What are the limitations of this study?"*

### 5. Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Key Technical Decisions

**Why FAISS over ChromaDB?**
FAISS is a pure C++ library with Python bindings — no server, no Docker, sub-millisecond search. For single-user document Q&A, it outperforms ChromaDB on latency. ChromaDB is better for multi-user, persistent, server-based deployments.

**Why `text-embedding-3-small`?**
It delivers 98% of `text-embedding-ada-002` quality at 5× lower cost. For document Q&A, embedding quality matters more than raw generation quality.

**Why chunk overlap of 150 tokens?**
Sentences split across chunk boundaries lose context. 150-token overlap ensures any given sentence's surrounding context is always present in at least one chunk.

**Why MMR over pure similarity search?**
A naive top-k similarity search often returns 5 near-identical chunks from the same paragraph. MMR balances relevance against diversity, giving the LLM a broader view of the document per query.

**Why `ConversationBufferWindowMemory` with k=6?**
Unlimited memory causes context window overflow on long documents. A window of 6 exchanges keeps follow-up questions coherent without blowing the token budget.

---

## Future Improvements

- Support DOCX, PPTX, and web URL ingestion
- Hybrid search (BM25 sparse + dense vectors) for better keyword recall
- Fine-tuned domain-specific embeddings for legal/medical documents
- Multi-document querying across a folder of PDFs
- Streaming token output for faster perceived response
- Deployment on Streamlit Community Cloud
