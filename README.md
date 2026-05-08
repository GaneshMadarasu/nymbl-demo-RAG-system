# DocRAG

Single-document RAG system: upload a PDF, ask questions, get grounded answers with citations.

## Stack

| Layer | Technology |
|-------|-----------|
| PDF extraction | PyMuPDF (`fitz`) |
| Embeddings | `gemini-embedding-2` (768-dim) |
| Chunking | tiktoken-aware sentence-boundary splitter |
| Retrieval | Hybrid search — pgvector HNSW (dense) + `tsvector` BM25 (sparse), fused via RRF |
| Re-ranking | Gemini 2.5 Flash (same model as answering, orders retrieved chunks before answering) |
| Answering | Gemini 2.5 Flash (streamed) |
| Vector store | Postgres 16 + pgvector (HNSW) |
| Backend | FastAPI + asyncpg |
| Frontend | Single-file HTML/CSS/JS + PDF viewer with chunk highlighting |
| Logging | Python `logging` — console + rotating file (`logs/app.log`, 5 MB × 3 backups) |

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
make db
```

Wait ~10 seconds for the health check to pass: `docker compose ps` should show `db` as `healthy`.

### 4. Install Python dependencies

```bash
make setup
```

### 5. Run

```bash
make dev
```

Open **http://localhost:8000** in your browser.

## Common commands

| Command | Description |
|---------|-------------|
| `make setup` | Copy `.env.example` and install dependencies |
| `make db` | Start Postgres via Docker Compose |
| `make dev` | Run the FastAPI server with auto-reload |
| `make test` | Run the test suite |
| `make logs` | Tail `logs/app.log` live |
| `make clean` | Remove `__pycache__`, `.pytest_cache`, and `logs/` |
| `make reset` | Restart the database (stop + start) |
| `make stop` | Stop Docker Compose |

## Usage

1. Drag a PDF onto the sidebar or click to upload
2. Wait for the ingestion progress bar to complete
3. Type a question and press Enter or click →
4. The answer streams back with `§ Chunk N` citation pills — hover for a text preview, click to open the PDF viewer with the chunk highlighted in yellow

If the document doesn't contain enough information to answer, the system responds: *"I don't know."*

## Running tests

> Requires the Docker Compose database to be running.

```bash
make test
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

**Hybrid search (dense + sparse, RRF fusion)**
Vector similarity alone misses exact keyword matches — a query for a specific model number or proper noun may rank semantically similar but wrong chunks higher. Combining pgvector cosine search (dense) with Postgres `tsvector` BM25 ranking (sparse) and fusing results via Reciprocal Rank Fusion gives better coverage across both semantic and lexical queries.

**Two-model re-ranking pipeline**
Retrieved chunks are re-ordered by `gemini-2.5-flash` before the top half are passed to the same model for answering. Using a smaller model for ranking keeps latency low — ranking is a simple ordering task that doesn't need the full capability of the answering model — while still improving the quality of context the answering model receives.

**Ingest deduplication / skip re-embedding**
Each uploaded PDF is identified by a SHA-256 hash of its bytes. If the same file is re-uploaded while its chunks are still in the database, the embedding step is skipped entirely and the cached result is returned immediately. `doc_meta` is preserved across uploads so the hash can be checked even after the chunks table is cleared for a new document; the skip only fires when both the metadata record and the actual chunks are present.

**Normalised error responses**
All HTTP errors return `{"error": "plain English message"}` via a global FastAPI exception handler. The same key is used for streaming ingest errors. Every error is also logged as a warning with method, path, status code, and message so failures are traceable in `logs/app.log` without exposing tracebacks to the client.

**Rotating file logging**
Logs write to both the console and `logs/app.log`. The file handler rotates at 5 MB and keeps three backups, preventing unbounded disk growth during long-running demo sessions.

**PDF viewer with chunk highlighting**
Clicking a citation pill opens a separate viewer page (`/viewer`) that renders the full document via PDF.js and highlights the cited chunk in yellow. The chunk's page number is stored in the database at ingest time (by scanning each PDF page with PyMuPDF and matching the chunk text), so the viewer navigates directly to the right page rather than scanning the whole document at query time.

**Cross-page chunk highlighting**
A single chunk can span two PDF pages. The viewer highlights up to three pages: the stored page (always), the page before it (in case the chunk's real start is there due to math-heavy openings that confuse the page-finder), and the page after (in case the chunk spills over). Each direction uses a different strategy: the target page uses substring anchors then word-sequence matching; the preceding page searches for the chunk start working forward; the following page anchors from the chunk's end working backward, with a `localAnchors` fallback that finds words present in both the chunk text and the page text. This fallback is necessary because PyMuPDF (used at ingest) and PDF.js (used in the browser) extract math and symbols differently, so normalized forms can diverge — shared plain-English words are always reliable anchors regardless of extraction differences.
