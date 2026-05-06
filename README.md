# DocRAG

Single-document RAG system: upload a PDF, ask questions, get grounded answers with citations.

## Stack

| Layer | Technology |
|-------|-----------|
| PDF extraction | PyMuPDF (`fitz`) |
| Embeddings | `gemini-embedding-2` (768-dim) |
| Chunking | tiktoken-aware sentence-boundary splitter |
| Answering | Gemini 2.5 Flash (streamed) |
| Vector store | Postgres 16 + pgvector (HNSW) |
| Backend | FastAPI + asyncpg |
| Frontend | Single-file HTML/CSS/JS |

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose
- A Gemini API key

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY
```

### 3. Start the database

```bash
docker compose up -d
```

Wait ~10 seconds for the health check to pass: `docker compose ps` should show `db` as `healthy`.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Run

```bash
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

## Usage

1. Drag a PDF onto the sidebar or click to upload
2. Wait for the ingestion progress bar to complete
3. Type a question and press Enter or click →
4. The answer streams back with `§ Chunk N` citation pills — hover for raw text

If the document doesn't contain enough information to answer, the system responds: *"I don't know."*

## Running tests

> Requires the Docker Compose database to be running.

```bash
pytest -v
```

## Design decisions

**PyMuPDF for PDF extraction**
Gemini's generative extraction is bounded by its output token limit (65,536 tokens max), which silently truncates large documents — a 100-page PDF that should produce 100+ chunks comes back as 6. PyMuPDF extracts all text verbatim with no output limits, making it reliable for documents of any size. Gemini is still used for embeddings and answering.

**`gemini-embedding-2` for embeddings**
768-dim vectors, consistent similarity space with the same provider used for answering, and supports separate `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types which improve retrieval accuracy.

**Chunking — token-aware sentence-boundary splitter**
Rather than splitting on raw character count, `tiktoken` measures token length accurately and splits only at sentence boundaries. This keeps chunks semantically coherent and avoids splitting mid-sentence, which degrades retrieval quality.

**pgvector HNSW index**
Fast approximate nearest-neighbour search with good recall at demo scale. No separate vector database service needed — Postgres handles both structured data and vector search.

**Single HTML file frontend**
No build tools, no Node.js, no npm. The UI is served directly by FastAPI, so setup is a single `uvicorn` command.
