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
768-dim vectors, consistent similarity space with the same provider used for answering, and supports separate `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types which improve retrieval accuracy over a generic symmetric embedding.

**tiktoken for token counting (`cl100k_base`)**
Used in two places: (1) in the chunker, to measure when a chunk is full; (2) before chunking, to count the document's total tokens so `compute_params` can pick the right `chunk_size` and `k`. Character or word counts are not used because they're unreliable across document types — dense technical prose and whitespace-heavy PDFs can have the same character count but differ wildly in token count. `cl100k_base` is chosen because it's a close approximation for modern LLMs (Gemini included), is fast (Rust-backed, runs in microseconds), and keeps both use sites consistent with the same vocabulary.

**Auto-scaled chunk size and retrieval k**
A fixed chunk size performs poorly at both extremes: tiny documents get over-split (chunks too small to carry meaning), large documents get under-split (chunks too large to embed accurately). `compute_params` selects `(chunk_size, k)` from five tiers based on document token count — small docs use tight chunks and fewer retrieved results; large docs use wider chunks and more results. The k value is persisted to the database so it survives server restarts and is used for every query against that document.

**Sentence-boundary overlap**
Each new chunk seeds itself with the last few sentences of the previous chunk (up to 64 tokens worth). This prevents context loss at chunk boundaries, where splitting mid-argument would make either chunk meaningless in isolation.

**pgvector HNSW index**
Fast approximate nearest-neighbour search with good recall at demo scale. No separate vector database service needed — Postgres handles both relational metadata and vector search. HNSW is preferred over IVFFlat because it requires no pre-training (`ANALYZE`) and maintains good recall without tuning `lists`.

**asyncpg over SQLAlchemy/psycopg**
asyncpg speaks the Postgres binary wire protocol directly, which gives lower latency and no ORM overhead. Since the schema is minimal (two tables, straightforward queries), there's no benefit to an ORM and the direct driver keeps the dependency footprint small.

**Retry logic for Gemini 429/503**
Embedding large documents concurrently hits rate limits. Retries use exponential backoff (2, 4, 8 … 64 s) and only trigger on retryable status codes (`429 ResourceExhausted`, `503 ServiceUnavailable`, `UNAVAILABLE`). Non-retryable errors (bad request, auth failure) propagate immediately so they surface clearly rather than timing out.

**Single HTML file frontend**
No build tools, no Node.js, no npm. The UI is served directly by FastAPI, so setup is a single `uvicorn` command.
