<!-- generated-by: gsd-doc-writer -->
# DocRAG — Single-Document RAG Demo

A small but realistic Retrieval-Augmented Generation system that answers questions about a single uploaded PDF. Built for the Nymbl technical assessment. Upload a PDF, the backend extracts text and figures, embeds them with Gemini, stores them in Postgres + pgvector, and a FastAPI streaming endpoint serves grounded answers with inline citations and clickable image previews.

## Stack

| Layer        | Choice                                                          |
| ------------ | --------------------------------------------------------------- |
| Language     | Python 3.11+                                                    |
| API          | FastAPI 0.115 + Uvicorn (SSE streaming for ingest and query)    |
| Database     | Postgres 16 + `pgvector` (HNSW vector index, GIN tsvector)      |
| DB driver    | `asyncpg` 0.29                                                  |
| Embeddings   | `gemini-embedding-2` at 768 dims (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`) |
| Generation   | `gemini-2.5-flash` (text + multimodal vision)                   |
| PDF parsing  | PyMuPDF (`pymupdf` 1.27)                                        |
| Tokenization | `tiktoken` `cl100k_base`                                        |
| Frontend     | Single-file HTML + vanilla JS (`frontend/index.html`), `marked` for markdown |
| Tests        | `pytest` 8.3 + `pytest-asyncio` (auto mode)                     |
| Dev infra    | `docker compose` (Postgres only), GNU Make targets              |

## Prerequisites

- Python 3.11 or newer
- Docker (for the Postgres container)
- A Gemini API key

## Setup

```bash
# 1. Create the .env from the template and install Python deps
make setup

# 2. Edit .env — set GEMINI_API_KEY to a real key
#    DATABASE_URL is already pre-pointed at the docker-compose Postgres
#    (postgresql://rag:rag@localhost:5432/ragdb)

# 3. Start Postgres + pgvector in the background
make db

# 4. Run the FastAPI dev server (auto-reload)
make dev
```

Open <http://localhost:8000>. Drop a PDF on the upload zone, pick an ingest mode in the modal, and start asking questions.

### Make targets

| Command      | Description                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| `make setup` | Copy `.env.example` → `.env` (if absent) and `pip install -r requirements.txt` |
| `make db`    | Start the `pgvector/pgvector:pg16` Postgres container via `docker compose`   |
| `make dev`   | Run `uvicorn backend.main:app --reload`                                      |
| `make test`  | Run the pytest suite                                                         |
| `make logs`  | `tail -f logs/app.log`                                                       |
| `make stop`  | `docker compose down` — stop the database                                    |
| `make reset` | `make stop` followed by `make db` — fresh DB process, same volume            |
| `make clean` | Remove `__pycache__/`, `.pytest_cache/`, and `logs/`                         |

## Usage Flow

1. **Upload**. The frontend (`frontend/index.html`) opens an ingest-mode modal so you can pick what the pipeline should do for this PDF:
   - **+Images** — extract text *and* figures, caption every figure with Gemini Vision, and store image bytes alongside captions.
   - **Text** — text-only ingest; skips image extraction and Vision captioning.
   - **Hand Written?** — runs Gemini Vision OCR on pages whose extracted text is sparse (≤ 50 chars). Useful for scanned or handwritten PDFs.
   - **Markings?** — runs a separate Vision pass to detect hand-drawn markup (underlines, highlights, margin notes, circles, arrows) and folds the result into the page text. An optional page-range box (e.g. `1-50, 100, 200-220`) limits the scan.
2. **Ingest** (`POST /ingest`, streamed via SSE). The server walks through stages — `extracting`, optional `ocr`/`markup`, `chunking`, `clearing`, `embedding`, optional `extracting_images` / `captioning` / `embedding_images`, then `done`. Progress is rendered live in the sidebar.
3. **Query** (`POST /query`, streamed via SSE). For follow-up questions containing pronouns, the question is rewritten against the last 6 turns of history so retrieval sees a self-contained query. The query is embedded, hybrid-searched, and the chunks are stuffed into a Gemini prompt; image chunks are attached as inline image parts so the model can see the picture. Tokens stream back to the chat.
4. **Citations & previews**. The model is instructed to cite as `[Chunk N]` (1-indexed within the retrieved context). The frontend renders those as clickable badges that jump to the PDF page in a side viewer (`/viewer`). Image chunks render as orange image pills, and bolded artwork titles in answers are linked back to the image they describe via fuzzy caption matching.

## Architecture in one diagram

```
                  ┌────────────────────────────────────────────────────┐
   PDF upload ──▶ │ backend/ingest.py                                  │
                  │  • PyMuPDF text + PDF annotation extraction        │
                  │  • optional Vision OCR for sparse pages            │
                  │  • optional Vision markup detection                │
                  │  • token-aware chunking (chunks.py)                │
                  │  • image extraction (embedded + page-render fall)  │
                  │  • Vision captioning → embed caption (BM25 + dense)│
                  │  • bounded-concurrency Gemini calls                │
                  └────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌────────────────────────────────────────────────────┐
                  │ Postgres 16 + pgvector  (backend/db.py)            │
                  │  chunks(embedding vector(768), tsv tsvector,       │
                  │         text, parent_text, image_data, ...)        │
                  │  HNSW index on embedding • GIN index on tsv        │
                  │  doc_meta(doc_id, chunk_count, k)                  │
                  └────────────────────────────────────────────────────┘
                                          │
                                          ▼
   Question ───▶ ┌────────────────────────────────────────────────────┐
                  │ backend/query.py                                   │
                  │  • pronoun-aware query rewrite (Gemini)            │
                  │  • embed query, hybrid search (dense + BM25, RRF)  │
                  │  • build multimodal prompt (text + image parts)    │
                  │  • stream Gemini answer back as SSE                │
                  └────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              FastAPI / SSE  ──▶ frontend/index.html
```

## API surface

| Method   | Path                       | Purpose                                                        |
| -------- | -------------------------- | -------------------------------------------------------------- |
| `GET`    | `/`                        | Serves the chat UI (`frontend/index.html`)                     |
| `GET`    | `/viewer`                  | Serves the side PDF viewer                                     |
| `GET`    | `/image-viewer`            | Serves the standalone image viewer                             |
| `GET`    | `/health`                  | `{"status": "ok"}`                                             |
| `GET`    | `/doc/info`                | Loaded-document summary (or `{"loaded": false}`)               |
| `GET`    | `/doc/pdf`                 | Raw bytes of the currently loaded PDF                          |
| `GET`    | `/doc/chunk/{i}`           | Text + page number for chunk index `i`                         |
| `GET`    | `/doc/images`              | All image chunks with caption + page text (no bytes)           |
| `GET`    | `/image/{i}`               | Image bytes for image chunk `i`                                |
| `GET`    | `/image/{i}/meta`          | Caption + page number + mime for image chunk `i`               |
| `POST`   | `/ingest`                  | Multipart upload; SSE stream of stage events. Form flags: `process_images`, `ocr_scanned`, `detect_markup`, `markup_pages`. 500 MB upload cap. |
| `POST`   | `/query`                   | JSON `{question, history}`; SSE stream of `status` / `sources` / `token` / `done` events. |
| `DELETE` | `/doc`                     | Truncate the chunks table and clear server-side doc state      |

## Repository layout

```
backend/
  main.py        FastAPI app, lifespan state, SSE endpoints, logging setup
  config.py      Loads .env, hard-fails on missing GEMINI_API_KEY / DATABASE_URL
  db.py          asyncpg pool, schema + idempotent migration, hybrid SQL
  ingest.py      PDF → text/images → Gemini embed/caption → DB
  query.py       Query rewrite, embed, hybrid search, multimodal answer
  chunks.py      tiktoken-aware sentence-boundary chunker with overlap
frontend/
  index.html     Chat UI, ingest-mode modal, citation pills, image links
  viewer.html    PDF viewer pane used for jump-to-chunk
  image-viewer.html
tests/           pytest suite (api, chunks, db, ingest, query, integration_ocr)
docker-compose.yml   pgvector/pgvector:pg16 service
Makefile         setup / db / dev / test / logs / clean / reset / stop
requirements.txt
```

## Configuration

| Variable         | Required | Description                                              |
| ---------------- | -------- | -------------------------------------------------------- |
| `GEMINI_API_KEY` | Yes      | API key for `gemini-2.5-flash` and `gemini-embedding-2`. The process exits at import time if this is missing. |
| `DATABASE_URL`   | Yes      | Postgres connection string. Defaults in `.env.example` to the docker-compose service: `postgresql://rag:rag@localhost:5432/ragdb`. |

The schema and migrations are applied automatically the first time the app touches the database — no separate migration command. See `SCHEMA` and `MIGRATION` constants in `backend/db.py`.

## Testing

```bash
make test          # runs pytest with asyncio_mode=auto
```

The suite includes unit tests (`test_chunks.py`, `test_db.py`, `test_query.py`, `test_ingest.py`), an HTTP-level smoke test (`test_api.py`), and an integration-style OCR test (`test_integration_ocr.py`). All Gemini calls are mocked; the DB tests run against the real Postgres container started by `make db`. `tests/conftest.py` provides an `asyncpg` pool fixture and truncates `chunks` / `doc_meta` between tests.

## Key design decisions

- **Hybrid retrieval, no LLM re-rank.** `db.search_chunks` runs a dense ANN search (cosine via `<=>` on the HNSW index) and a BM25-style sparse search (`ts_rank_cd` on a generated `tsvector`) in a single SQL CTE, then fuses both rankings with Reciprocal Rank Fusion (`1 / (60 + rank)`). An earlier version added an LLM re-ranking step on top of the fused result; that step was **removed** because it consistently added latency without a measurable quality lift on this corpus, and RRF alone proved good enough.
- **Caption-then-embed for images, not multimodal embeddings.** Each figure is captioned by Gemini Vision and the *caption* is embedded as a normal text chunk. Captions are load-bearing: they participate in BM25, they drive the fuzzy painting-name → image link in the chat, and they feed the `(image, page N): "..."` label that the model sees in the prompt. A multimodal embedding approach was considered and rejected because it would lose those textual surfaces. The original image bytes are stored in `chunks.image_data` and re-attached as a Gemini `Part` at answer time so the model still *sees* the picture.
- **Known limitation — image-caption recall.** Captions are ingested, embedded, and BM25-indexed correctly, but retrieval can still miss the right figure for a natural-language visual query. The sparse path uses `plainto_tsquery`, which ANDs every term, so a single synonym or filler word zeroes the match (`"red coat"` doesn't match a caption that says `"red cloak"`); and dense ranking is dominated by a document's own prose, so a terse caption falls below the top-`k` cutoff. The net effect: *"show me a painting with a red coat"* can return "I don't know" even when a matching figure exists. The fix — OR-based BM25 plus reserved image slots gated by a similarity floor — is specced in [`docs/superpowers/specs/2026-05-20-image-caption-retrieval-recall-design.md`](docs/superpowers/specs/2026-05-20-image-caption-retrieval-recall-design.md) and **not yet implemented**.
- **Two-stage image extraction.** Stage 1 walks every page's image XRefs and stores the *raw embedded bytes* (re-rendering the bbox would burn page text overlays into the image). Stage 2 falls back to a 150-DPI full-page render only on pages that have *no* embedded images and *very little* text — the heuristic for "this page is mostly a scan or a full-bleed painting." Captions like "blank", "page of text", "appears empty" are filtered before any embed call is spent on them.
- **Document-size-adaptive chunking.** `compute_params(total_tokens)` scales chunk size from 256 tokens / `k=5` (under ~20 pages) up to 1024 tokens / `k=20` (1000+ pages), so a small datasheet doesn't get chopped into noise and a large book doesn't drown the model in tiny snippets.
- **Parent-text windows.** Each chunk is stored next to its `parent_text` — the previous + current + next chunk concatenated. The dense/BM25 search ranks on the small chunk (precise), but the prompt is built from the parent (more context), trading none of retrieval precision for fewer "the answer was in the next chunk" misses.
- **PDF annotations and Vision-detected markup share a path.** Highlights, underlines, strikethroughs, and sticky notes are read directly from PDF annotation objects in `_format_annotations`. The "Markings?" toggle adds a Vision pass for markup that's *baked into the page raster* (handwritten ink, marker highlights, hand-drawn circles) and folds the result into the same bracketed `[underlined: ...]` / `[note: ...]` shape, so a query like "what did I underline?" matches both sources uniformly.
- **Bounded-concurrency Gemini calls.** Embed/caption/OCR calls run through a semaphore-bounded gather (`_GEMINI_CONCURRENCY = 8`); the lighter-weight markup detection runs at `_MARKUP_CONCURRENCY = 32`. This is reliably faster than unbounded `gather()` once the count exceeds ~24, because it avoids 429 retries and their exponential-backoff penalty.
- **Pronoun-aware query rewriting.** Follow-up questions containing pronouns (`it`, `this`, `they`, `here`, …) are rewritten against the last 6 turns of conversation history *before* embedding. The original question is still what the LLM answers — only retrieval sees the rewritten form.
- **Idempotent ingest by content hash.** `doc_id` is the SHA-256 (truncated to 16 hex chars) of the PDF bytes, so re-uploading the same file short-circuits to a cached `done` event and skips re-embedding.
- **Single-document state.** The server keeps the loaded `doc_id` / `chunk_count` / `k` in module-level memory, restored on startup from the most recent `doc_meta` row. Uploading a new PDF truncates the `chunks` table — this is a demo, not a multi-tenant service.

## License

Not licensed for redistribution. Built as a technical assessment.
