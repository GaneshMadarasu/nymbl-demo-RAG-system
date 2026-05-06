# DocRAG

Single-document RAG system: upload a PDF, ask questions, get grounded answers with citations.

## Stack

| Layer | Technology |
|-------|-----------|
| PDF extraction | PyMuPDF (`fitz`) |
| Embeddings | `BAAI/bge-base-en-v1.5` via fastembed (768-dim, local ONNX) |
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

**PDF extraction — PyMuPDF instead of Gemini**
The spec recommends using Gemini to extract plaintext from PDFs. We use PyMuPDF locally instead. For PDFs that are already structured text (not scanned images), a local parser produces equivalent quality with no API cost, no quota consumption, and no added latency. Gemini extraction makes sense when the input includes images or complex layouts; for text-native PDFs it's unnecessary overhead.

**Embeddings — fastembed (`bge-base-en-v1.5`) instead of Gemini Embeddings**
The spec recommends Gemini Embeddings. The Gemini embedding API has a hard daily quota on the free tier that is easy to exhaust during development and demos. `BAAI/bge-base-en-v1.5` runs locally via ONNX (no GPU required), requires no API key, produces the same 768-dim vectors, and is a well-benchmarked open model. This makes the demo reliably runnable without any quota management.

**Chunking — token-aware sentence-boundary splitter**
The spec asks for "reasonable chunk size with overlap." Rather than splitting on raw character count, we use `tiktoken` to measure token length accurately and split only at sentence boundaries. This keeps chunks semantically coherent and avoids splitting mid-sentence, which degrades retrieval quality.

**pgvector HNSW index**
Fast approximate nearest-neighbour search with good recall at demo scale. No separate vector database service needed — Postgres handles both structured data and vector search.

**Single HTML file frontend**
No build tools, no Node.js, no npm. The UI is served directly by FastAPI, so setup is a single `uvicorn` command.
