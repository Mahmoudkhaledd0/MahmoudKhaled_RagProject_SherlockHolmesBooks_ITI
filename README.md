# Sherlock Holmes RAG

A Retrieval-Augmented Generation (RAG) system that answers questions about the
Sherlock Holmes canon. It ingests the original books, embeds them into a vector
database, and serves grounded, source-cited answers through a FastAPI backend
and a lightweight web UI.

The pipeline routes every incoming message: book questions hit the retriever,
greetings get a friendly reply, and anything off-topic is politely declined —
so the model only answers from the source text and never hallucinates outside
the corpus.

---

## Features

- **Query routing** — an LLM classifier tags each message as `retrieve`,
  `chitchat`, or `off-topic` and dispatches it accordingly.
- **Semantic retrieval** — dense vector search over the full Sherlock Holmes
  corpus using multilingual E5 embeddings.
- **Grounded answers** — responses are generated strictly from retrieved
  passages
- **Source citations** — every answer includes book name, page number
- **Multi-provider LLMs** — Groq for fast routing/chitchat, Google Gemini for
  answer synthesis.
- **Web interface** — a single-page `index.html` for interactive Q&A.

---

## Architecture

```
                 ┌─────────────┐
  User query ──▶ │  Router LLM │ (Groq)
                 └──────┬──────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   retrieve         chitchat         off-topic
        │               │                │
        ▼               ▼                ▼
  Embed query     Friendly reply    Decline politely
        │           (Groq LLM)      (fixed message)
        ▼
  Qdrant vector search (top-k)
        │
        ▼
  Answer LLM (Gemini) + page sources
```

**Data pipeline** (`rag-pipeline.ipynb`):

1. **Extract** — `Sherlock Holmes.pdf` is converted to Markdown (`output.md`).
2. **Split** — the combined text is sliced by book and page into per-page
   Markdown files under `dataset/` (one file per page).
3. **Embed** — each page is encoded with `intfloat/multilingual-e5-large`
   (1024-dim vectors).
4. **Index** — vectors and payloads (book name, page number, content) are
   upserted into a Qdrant collection.

---

## Tech Stack

| Layer          | Technology                                      |
| -------------- | ----------------------------------------------- |
| API            | FastAPI + Uvicorn                               |
| Vector DB      | Qdrant                                          |
| Embeddings     | `sentence-transformers` (multilingual E5 large) |
| LLM (routing)  | Groq (`langchain-groq`)                         |
| LLM (answers)  | Google Gemini (`langchain-google-genai`)        |
| Frontend       | Static HTML/JS (`index.html`)                   |

---

## Project Structure

```
rag/
├── rag_api.py          # FastAPI app: /query, /health endpoints
├── rag-pipeline.ipynb  # PDF → Markdown → embeddings → Qdrant pipeline
├── index.html          # Web UI client
├── dataset/            # Per-page Markdown files (source corpus)
├── output.md           # Full extracted book text
├── Sherlock Holmes.pdf # Source PDF
├── requirements.txt    # Python dependencies
└── .env                # Configuration (not committed)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A running Qdrant instance (cloud or self-hosted)
- API keys: Groq and Google Gemini

### Installation

```bash
git clone https://github.com/Mahmoudkhaledd0/MahmoudKhaled_RagProject_ITI.git
cd MahmoudKhaled_RagProject_ITI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Qdrant
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=sherlock_holmes
TOP_K=2

# Embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Gemini (answer generation)
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.1-flash-lite

# Groq (routing + chitchat)
GROQ_API_KEY=your-groq-key
GROQ_MODEL=openai/gpt-oss-120b
```

### Build the index

Run `rag-pipeline.ipynb` to extract the PDF, generate the dataset,
embed the pages, and populate the Qdrant collection.

### Run the API

```bash
uvicorn rag_api:app --reload --host 127.0.0.1 --port 8000
```

- API root: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

### Run the UI

Open `index.html` in a browser (it calls the API at `127.0.0.1:8000`).

---



## Corpus

The knowledge base covers the full Sherlock Holmes canon:

- A Study in Scarlet
- The Sign of the Four
- The Adventures of Sherlock Holmes
- The Memoirs of Sherlock Holmes
- The Return of Sherlock Holmes
- The Hound of the Baskervilles
- The Valley of Fear
- His Last Bow
- The Case-Book of Sherlock Holmes

---

## Notes

- The Sherlock Holmes canon is in the public domain.
- `.env`, `venv/`, and `__pycache__/` are git-ignored — never commit secrets.
- The API loads the embedding model at startup; the same model used to build
  the index **must** be used at query time to keep vector dimensions aligned.


