<!-- generated-by: gsd-doc-writer -->

# Configuration

This document catalogs every configuration knob that affects DocRAG's behavior:
required environment variables, infrastructure (Docker / Postgres) settings, and
the in-source constants that govern chunking, retry, concurrency, OCR, and
multimodal Vision behavior. Operators tuning ingest cost, throughput, or
retrieval quality should start here.

## Environment Variables

DocRAG reads two environment variables at startup. Both are **required** —
if either is missing or empty, the process prints `ERROR: missing required env
vars: ...` to stderr and exits with code `1` (`backend/config.py:18-23`).

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | _none_ | Google Gemini API key used by the embedding (`gemini-embedding-2`), Vision (`gemini-2.5-flash`), and generation models. Loaded once at process start in `backend/config.py` and shared by `backend/ingest.py` and `backend/query.py`. |
| `DATABASE_URL` | Yes | _none_ | Postgres connection string in libpq URL form. Passed directly to `asyncpg.create_pool()` in `backend/db.py`. The bundled `docker-compose.yml` provisions a database matching the example value below. |

Variables are loaded from a `.env` file in the project root via `python-dotenv`
(`backend/config.py:3-5`). A canonical `.env.example` is committed:

```bash
GEMINI_API_KEY=your_key_here
DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb
```

To get started locally, copy the example and fill in your key:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY
```

## Database Configuration (docker-compose.yml)

The bundled Postgres instance is defined in `docker-compose.yml`:

| Setting | Value | Notes |
| --- | --- | --- |
| Image | `pgvector/pgvector:pg16` | Postgres 16 with `pgvector` extension preinstalled. |
| Database | `ragdb` | Created on first boot. |
| User / password | `rag` / `rag` | Hard-coded in `docker-compose.yml`. Change these (and `DATABASE_URL`) before exposing the service outside localhost. |
| Host port | `5432` | Mapped to container `5432`. |
| Volume | `pgdata` | Named Docker volume — embeddings and chunks survive `docker compose down`. Use `docker compose down -v` to wipe. |
| Healthcheck | `pg_isready -U rag -d ragdb` | Runs every 5s, 5s timeout, 5 retries. |

The schema and indexes are created automatically by `backend/db.py` on first
connection (`SCHEMA` and `MIGRATION` blocks at lines 7-71): the `chunks` and
`doc_meta` tables, the HNSW vector index (`chunks_embedding_idx`), and the GIN
index over the `tsv` full-text column (`chunks_tsv_idx`).

## Upload Limits

Defined in `backend/main.py`:

| Constant | Value | Description |
| --- | --- | --- |
| `_MAX_UPLOAD_BYTES` | `500 * 1024 * 1024` (500 MB) | Maximum PDF upload size accepted by `POST /ingest`. Requests over this limit return HTTP `413`. |

The `/ingest` route also enforces a `.pdf` extension check (`main.py:184`); any
other file type returns HTTP `400`.

The PDF the user uploaded is cached on disk at
`{tempfile.gettempdir()}/docrag_current.pdf` (`main.py:33`) so the
`/doc/pdf` route can serve it back to the in-browser viewer. Cleared on
`DELETE /doc`.

## Logging

Logging is configured at process start in `backend/main.py:14-27`:

| Setting | Value | Description |
| --- | --- | --- |
| Log directory | `logs/` (project root) | Created on startup if missing. |
| Log file | `logs/app.log` | Rotated by `RotatingFileHandler`. |
| Max bytes per file | `5 * 1024 * 1024` (5 MB) | Rotation trigger. |
| Backup count | `3` | Keeps `app.log.1` … `app.log.3`. |
| Level | `INFO` | Set via `logging.basicConfig`. |
| Handlers | Console + rotating file | Both receive the same format string `"%(asctime)s %(levelname)s %(name)s: %(message)s"`. |

## Embedding & Generation Models

These are hard-coded model identifiers used by both `backend/ingest.py` and
`backend/query.py`. They are **not** environment-configurable — change them in
source if you need to swap models.

| Constant | Value | File | Used for |
| --- | --- | --- | --- |
| `_EMBED_MODEL` | `gemini-embedding-2` | `ingest.py:24`, `query.py:18` | Document and query embeddings. |
| `_VISION_MODEL` | `gemini-2.5-flash` | `ingest.py:25` | Image captioning, OCR, hand-drawn markup detection. |
| `_GEN_MODEL` | `gemini-2.5-flash` | `query.py:17` | Streaming answer generation and pronoun-resolution query rewriting. |
| `_EMBED_DIM` | `768` | `ingest.py:26`, `query.py:19`, `db.py` (schema `vector(768)`) | Embedding output dimensionality. **Changing this requires dropping the `chunks` table** because the column type is `vector(768)`. |

The query embedding is requested with `task_type="RETRIEVAL_QUERY"` and the
document embedding with `task_type="RETRIEVAL_DOCUMENT"`
(`ingest.py:279`, `query.py:61`).

## Retry Behavior

All Gemini API calls (embed, caption, OCR, markup detection) share the same
retry policy. Defined separately but identically in `backend/ingest.py` and
`backend/query.py`:

| Constant | Value | Description |
| --- | --- | --- |
| `_MAX_RETRIES` | `6` | Total attempts including the initial call. |
| `_RETRY_BASE` | `2.0` | Exponential backoff base — delays are `2, 4, 8, 16, 32, 64` seconds. |

The retry predicate `_is_retryable()` (`ingest.py:259-268`, `query.py:41-50`)
catches 429, 503, `ResourceExhausted`, `ServiceUnavailable`, `UNAVAILABLE`, and
timeouts. Anything else propagates immediately.

After all retries are exhausted, OCR and markup-detection calls return empty
results (a single bad page never aborts the ingest); embedding failures
propagate (the ingest aborts).

## Concurrency Limits

Two independent semaphores cap how many in-flight Gemini calls the ingest
pipeline issues at once. Both live in `backend/ingest.py`:

| Constant | Value | Used for | Rationale (from inline comments) |
| --- | --- | --- | --- |
| `_GEMINI_CONCURRENCY` | `8` | Caption, OCR, embed (text & image captions). | Below the per-minute limits of `gemini-embedding-2` / `gemini-2.5-flash` and reduces overall wall time vs unlimited `gather()` once you have ~30+ images. |
| `_MARKUP_CONCURRENCY` | `32` | Hand-drawn markup detection (one Vision call per text page). | Markup calls are lighter (small JSON output, no extracted-text in prompt); 32 cuts an 800-page markup pass from ~15 min to ~3 min. |

Both are enforced through `_gather_bounded(coros, limit)`
(`ingest.py:734-743`), an `asyncio.Semaphore` wrapper around
`asyncio.gather()`.

## Chunking Parameters

### Document-size tiers

`compute_params(total_tokens)` in `backend/ingest.py:161-172` selects a
chunk size and retrieval `k` based on the total token count of the document.
This is the only place `k` is determined — it is then persisted in the
`doc_meta` table and replayed for every query against that doc.

| Total tokens (approx pages) | `chunk_size` | `k` |
| --- | --- | --- |
| `< 10_000` (< ~20 pages) | `256` | `5` |
| `10_000 – 50_000` (20–100 pages) | `384` | `8` |
| `50_000 – 200_000` (100–400 pages) | `512` | `12` |
| `200_000 – 500_000` (400–1000 pages) | `768` | `15` |
| `>= 500_000` (1000+ pages) | `1024` | `20` |

Tokens are counted with the `cl100k_base` tiktoken encoder
(`ingest.py:18`, `chunks.py:4`).

### Chunker constants

`chunk_text(text, max_tokens, overlap)` in `backend/chunks.py:7`:

| Parameter | Default | Description |
| --- | --- | --- |
| `max_tokens` | `512` (overridden per-doc by `compute_params`) | Maximum tokens per chunk. The chunker flushes the in-progress chunk once adding the next sentence would exceed this. |
| `overlap` | `64` | Token budget for trailing sentences carried over to the next chunk. Helps cross-chunk context for retrieval. |

The chunker splits on sentence boundaries (`.!?` + whitespace) **or** raw
newlines (`chunks.py:15`). The newline branch handles unpunctuated content
(handwritten OCR output, bullet lists, headers) so the chunker still finds
split points instead of collapsing the whole text into one giant chunk.

## Image Extraction Heuristics

These constants in `backend/ingest.py` filter out icons, dividers, and
decorative artifacts before paying for Vision captions.

| Constant | Value | Description |
| --- | --- | --- |
| `_MIN_IMAGE_PX` | `200` | Skip embedded images smaller than 200 px in either dimension. |
| `_MIN_IMAGE_BYTES` | `5_000` | Skip embedded images under 5 KB (icons, dividers, decorative). |
| `_SUPPORTED_MIMES` | `{image/png, image/jpeg, image/webp, image/gif}` | Other MIME types are ignored. |
| `_FULLPAGE_DPI` | `150` | DPI for stage-2 full-page fallback render (pages with no embedded images and very little text). |
| `_PAGE_TEXT_FALLBACK_LIMIT` | `200` | Stage 2 fires only on pages whose stripped text length is below this; heuristic for "this page is mostly a painting/scan." |
| `_CAPTION_MAX_DIM` | `1024` | Longest-edge resize before sending to Vision (`_downsize_for_vision`). Cuts upload payload 10-20× for high-DPI page renders. |
| `_CAPTION_JPEG_QUALITY` | `85` | JPEG quality used when re-encoding the downsized image. |

`_BLANK_PHRASES` (`ingest.py:58-84`) is a tuple of lowercased substrings used
by `_is_blank_caption()` to drop captions like "no meaningful visual",
"text only", "blank page", etc. Captions matching these are filtered before
embed calls are spent on them.

## OCR Configuration

Triggered only when `ocr_scanned=true` is sent to `POST /ingest`.

| Constant | Value | Description |
| --- | --- | --- |
| `_OCR_PAGE_DPI` | `200` | DPI used when rendering a page for OCR (and for hand-drawn markup detection). |
| `_OCR_EMPTY_PAGE_THRESHOLD` | `50` | Pages whose stripped text length is `<=` this threshold are queued for OCR. |

OCR uses `_GEMINI_CONCURRENCY` (8) for in-flight calls (`ingest.py:700`).

## Hand-Drawn Markup Detection

Triggered only when `detect_markup=true` is sent to `POST /ingest`.
Uses `_MARKUP_CONCURRENCY` (32) and accepts an optional `markup_pages` form
field — a comma-separated page-range string like `"1-50, 100, 200-220"`
parsed by `_parse_page_ranges()` (`ingest.py:613`). When omitted, every
text-bearing page is scanned.

The Vision response is constrained to JSON via
`response_mime_type="application/json"` (`ingest.py:528`).

PDF annotation objects (typed `Highlight`, `Underline`, `Squiggly`,
`StrikeOut`, `FreeText`, `Text`) are extracted unconditionally during text
extraction and inlined into the page body — see `_ANNOT_HIGHLIGHT_LIKE` and
`_ANNOT_LABEL` in `backend/ingest.py:175-181`. There is no toggle for this.

## Retrieval Configuration

`backend/db.py` defines the SQL for retrieval. The hybrid search query
(`_HYBRID_SQL`, lines 129-166) fuses dense vector similarity and sparse BM25
(via Postgres `tsvector` / `ts_rank_cd`) with **Reciprocal Rank Fusion**.

| Knob | Value | Where | Description |
| --- | --- | --- | --- |
| RRF constant | `60.0` | `db.py:157-158` (`1.0 / (60.0 + rank)`) | Standard RRF damping. Lower values give more weight to top-ranked items. |
| Per-pool oversampling | `$3 * 2` | `db.py:136, 147` (`LIMIT $3 * 2`) | Dense and sparse pools each fetch `2 * k` candidates before RRF fusion picks the final `k`. |
| Final `k` | Document-dependent | Set by `compute_params()` and persisted in `doc_meta.k` | Replayed by `backend/main.py` on every `/query`. The default fallback when no doc is loaded is `8` (`main.py:35, 222, 35` etc.). |
| Vector index | HNSW, `vector_cosine_ops` | `db.py:32` | Created on `chunks.embedding`. |
| BM25 column | `tsv tsvector` (English) | `db.py:21` | `GENERATED ALWAYS AS to_tsvector('english', text)` with GIN index. |

Dense-only retrieval (`_DENSE_SQL`) is used as a fallback when the question
string is empty (`db.py:186-189`).

## Query-Side Configuration

In `backend/query.py`:

| Constant / setting | Value | Description |
| --- | --- | --- |
| `_PRONOUN_RE` | regex matching `it\|this\|that\|they\|them\|its\|their\|these\|those\|he\|she\|we\|here\|there` | Triggers query rewriting for follow-ups. Matched case-insensitively. |
| History window | last `6` messages | `query.py:84, 111` (`history[-6:]`). Used both for pronoun resolution and for the `Conversation so far:` block in the answer prompt. |
| `_CHUNK_REF_RE` | regex `\[Chunk\s+\d+\]` | Used to strip prior `[Chunk N]` citations from history before sending it to the model (so old chunk numbers don't poison new answers). |
| `SYSTEM_PROMPT` | hard-coded in `query.py:21-35` | Defines answer style ("ONLY the provided context", "I don't know." fallback, citation format `[Chunk N]`, image-link format `image:N`). Edit in source to change persona. |

## Tuning Cheat Sheet

| To do this | Edit |
| --- | --- |
| Use a different Gemini model | `_EMBED_MODEL` / `_VISION_MODEL` in `backend/ingest.py`; `_GEN_MODEL` / `_EMBED_MODEL` in `backend/query.py`. Changing `_EMBED_DIM` requires dropping the `chunks` table. |
| Reduce 429 errors | Lower `_GEMINI_CONCURRENCY` and/or `_MARKUP_CONCURRENCY` in `backend/ingest.py`. |
| Reduce wall time on large docs | Raise `_MARKUP_CONCURRENCY`; if your Gemini quota allows, raise `_GEMINI_CONCURRENCY`. |
| Tighter / looser chunk sizing | Adjust the tier table in `compute_params()` (`backend/ingest.py:161`). The persisted `k` in `doc_meta` is set at ingest time — re-ingest the doc to replay. |
| Capture more / fewer images | Lower `_MIN_IMAGE_PX` / `_MIN_IMAGE_BYTES` (more), or raise them (fewer). Edit `_BLANK_PHRASES` to control caption-based filtering. |
| Allow larger uploads | Raise `_MAX_UPLOAD_BYTES` in `backend/main.py`. |
| Change log retention | `maxBytes` and `backupCount` on `_file_handler` in `backend/main.py`. |
| Add a new env var | Add it as a typed annotation on `_Settings` in `backend/config.py` and read it in `__init__`. The class auto-validates that any annotated field is non-empty at startup. |

## Per-Environment Overrides

There is **no environment-tier mechanism** (no `.env.development`,
`.env.production`, `NODE_ENV`-style branching). The application reads a single
`.env` file at startup. To run multiple environments, point the working
directory or shell at the appropriate `.env` and bring up a separate
Postgres instance (or change `DATABASE_URL`).
