# NYMBL — DocRAG

Single-document RAG system built for the Nymbl technical assessment: upload a PDF, ask questions, get grounded answers with numbered citations and a built-in PDF viewer.

## Stack

| Layer | Technology |
|-------|-----------|
| PDF extraction | PyMuPDF (`fitz`) — text + embedded images |
| OCR (opt-in) | Gemini 2.5 Flash with structured JSON — line-level transcription + bboxes for scanned/handwritten pages |
| Image captioning | Gemini 2.5 Flash Vision (1-2 sentence captions per image at ingest) |
| Embeddings | `gemini-embedding-2` (768-dim) — text chunks AND image captions share the same space |
| Chunking | tiktoken-aware sentence-boundary splitter |
| Retrieval | Hybrid search — pgvector HNSW (dense) + `tsvector` BM25 (sparse), fused via RRF |
| Re-ranking | Removed — hybrid RRF retrieval quality made it redundant (see Design decisions) |
| Answering | Gemini 2.5 Flash multimodal (text + image bytes streamed) |
| Vector store | Postgres 16 + pgvector (HNSW); image bytes stored as `BYTEA` next to their captions |
| Backend | FastAPI + asyncpg |
| Frontend | Single-file HTML/CSS/JS + PDF viewer + image viewer with chunk highlighting (Nymbl light-mode design) |
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
2. A modal asks whether to **Process images** (extract figures, caption with Gemini Vision, embed) or do a **Text only** ingest (faster — skips Vision entirely). A separate checkbox enables **OCR scanned pages** for handwritten or scanned PDFs — empty-text pages are rendered and transcribed via Gemini Vision; cited lines are highlighted in yellow in the viewer.
3. Wait for the ingestion progress bar to complete
4. Type a question and press Enter or click →
5. A live status line shows each RAG stage: *Embedding query… → Searching database… → Generating answer…*
6. The answer streams in as rendered markdown; on completion, `[N]` citation badges are wired to source pills — text chunks open the PDF viewer (`§ Chunk N`, purple), image chunks open the image viewer (`🖼 Chunk N`, orange). Painting/figure names mentioned inline in the answer become clickable hyperlinks that open the matching image even when its chunk wasn't in the retrieved set

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

**Why a document can collapse into a single chunk**
Two distinct causes, both worth knowing about. **(1) Small-document floor.** The smallest tier (`< 10K` tokens) uses `chunk_size = 256`. A short handwritten note (150-400 tokens) fits entirely inside that budget, so the whole OCR'd text becomes one chunk by design — not a bug; the chunk still embeds and retrieves correctly, with citations pointing back at it. Lowering the floor in `compute_params` (e.g. `chunk_size = 128`) is the lever if you want finer granularity on tiny docs. **(2) Punctuation collapse.** The chunker splits on `.!?` followed by whitespace. A 16K-token handwritten transcript normally produces ~40 chunks at `chunk_size=384`, but if the OCR output has no sentence-ending punctuation (typical for notebooks, bullet lists, dated journal entries) the regex finds zero boundaries and the entire document becomes one giant "sentence" — and the chunker emits it as a single oversized chunk. The fix is to also split on raw `\n+`, treating newlines as soft sentence boundaries: typed prose is unchanged (sentences re-join with spaces inside each chunk anyway), but unpunctuated content now gets the split points it needs to honor `chunk_size`.

**Sentence-boundary overlap**
Each new chunk seeds itself with the last few sentences of the previous chunk (up to 64 tokens worth). This prevents context loss at chunk boundaries, where splitting mid-argument would make either chunk meaningless in isolation.

**pgvector HNSW index**
Fast approximate nearest-neighbour search with good recall at demo scale. No separate vector database service needed — Postgres handles both relational metadata and vector search. HNSW is preferred over IVFFlat because it requires no pre-training (`ANALYZE`) and maintains good recall without tuning `lists`.

**asyncpg over SQLAlchemy/psycopg**
asyncpg speaks the Postgres binary wire protocol directly, which gives lower latency and no ORM overhead. Since the schema is minimal (two tables, straightforward queries), there's no benefit to an ORM and the direct driver keeps the dependency footprint small.

**Retry logic for Gemini 429/503**
Embedding large documents concurrently hits rate limits. Retries use exponential backoff (2, 4, 8 … 64 s) and only trigger on retryable status codes (`429 ResourceExhausted`, `503 ServiceUnavailable`, `UNAVAILABLE`). Non-retryable errors (bad request, auth failure) propagate immediately so they surface clearly rather than timing out.

**Single HTML file frontend**
No build tools, no Node.js, no npm. The UI is served directly by FastAPI, so setup is a single `uvicorn` command. Two CDN libraries are loaded at runtime: marked.js (markdown rendering in the chat) and PDF.js (PDF rendering in the viewer).

**Hybrid search (dense + sparse, RRF fusion)**
Vector similarity alone misses exact keyword matches — a query for a specific model number or proper noun may rank semantically similar but wrong chunks higher. Combining pgvector cosine search (dense) with Postgres `tsvector` BM25 ranking (sparse) and fusing results via Reciprocal Rank Fusion gives better coverage across both semantic and lexical queries.

**Re-ranking removed — hybrid RRF retrieval is sufficient**
An LLM re-ranking step was originally included (Gemini 2.5 Flash re-ordering the top-k chunks before answering). It was removed because: (1) the hybrid dense + BM25 + RRF retrieval already produces well-ordered results, so re-ranking only dropped 2 of 8 chunks with marginal quality gain; (2) each re-rank call added a full synchronous Gemini round-trip to every query, making it the dominant source of latency; (3) for a single-document Q&A demo, the answering model receives enough context from the top-k RRF results without a separate ordering pass.

**Ingest deduplication / skip re-embedding**
Each uploaded PDF is identified by a SHA-256 hash of its bytes. If the same file is re-uploaded while its chunks are still in the database, the embedding step is skipped entirely and the cached result is returned immediately. `doc_meta` is preserved across uploads so the hash can be checked even after the chunks table is cleared for a new document; the skip only fires when both the metadata record and the actual chunks are present.

**Normalised error responses**
All HTTP errors return `{"error": "plain English message"}` via a global FastAPI exception handler. The same key is used for streaming ingest errors. Every error is also logged as a warning with method, path, status code, and message so failures are traceable in `logs/app.log` without exposing tracebacks to the client.

**Rotating file logging**
Logs write to both the console and `logs/app.log`. The file handler rotates at 5 MB and keeps three backups, preventing unbounded disk growth during long-running demo sessions.

**PDF viewer with chunk highlighting**
Clicking a citation pill opens a separate viewer page (`/viewer`) that renders the full document via PDF.js and highlights the cited chunk in yellow. The chunk's page number is stored in the database at ingest time (by scanning each PDF page with PyMuPDF and matching the chunk text), so the viewer navigates directly to the right page rather than scanning the whole document at query time.

**Multimodal image retrieval (opt-in)**
Images embedded in the PDF are extracted with PyMuPDF at ingest time, captioned with Gemini 2.5 Flash Vision (1-2 sentences each), and stored as image chunks alongside text chunks — the caption is what gets embedded and BM25-indexed, so a query like "show me the architecture diagram" can match an image via its caption. The user opts in or out at upload time via a modal: text-only ingest skips Vision entirely (fast); processing images takes longer but enables visual citations. Tiny/decorative images (< 200 px or < 5 KB) are dropped before captioning. A two-stage extractor handles both layouts: stage 1 takes raw embedded images for any page, stage 2 falls back to a full-page render for sparse-text pages where PyMuPDF reports no embedded images (scanned/full-bleed paintings). After captioning, any image whose Vision caption is `"blank"` or contains a verbose negation like *"contains only text"* is dropped — this keeps text-page renders out of the candidate set so they don't pollute retrieval. At answer time, image chunks are passed to Gemini 2.5 Flash as actual `Part.from_bytes` parts (not just their captions), so the model reasons over the picture itself. Image citations render as orange `🖼` pills that open a dedicated `/image-viewer` page showing the full image with its caption and source page.

**Painting-name fuzzy linking in answers**
The chat caches every image chunk's caption + adjacent-page text on doc load (`/doc/images`). After the answer renders, every `<strong>` phrase is substring-matched against that corpus — bold painting/figure names in the answer become clickable hyperlinks that open the matching image, even when the image chunk wasn't in the top-k retrieval and even when the LLM didn't follow the explicit `[Title](image:N)` format. Among multiple matches, the matcher picks the image whose page text contains the phrase earliest (title position vs cross-reference position). This makes "Show me Frodo" and "What's the Holy Family painting about?" work consistently across art catalogs and illustrated novels.

**Caption-then-embed over direct multimodal embedding**
A simpler-on-paper architecture would skip Vision captioning and embed image bytes directly with a multimodal embedding model (image → vector, one API call instead of two). It was rejected here because the caption pulls double duty in this app: (1) the painting-name fuzzy linking in answers substring-matches bold phrases against the caption text — opaque vectors don't give us that; (2) the LLM sees image chunks as `[Chunk N] (image, page X): "caption"` so it can reference them by name in its answer — without a caption it can only point at "the picture"; (3) BM25 keyword search over captions catches exact-token queries ("Madonna", "Holy Family") that vector similarity might miss. The latency saving (skip one round-trip per image) is small once images are downsized and gather is bounded — not worth losing three features. A multimodal embedding would be the right call for a pure visual-search product where humans never read or cite captions.

**Vision call latency optimizations**
Images are downsized to 1024 px on the longest side and re-encoded as JPEG quality-85 before being sent to Vision — Vision doesn't need full resolution to caption an image, and high-DPI page renders dominate the upload time. A semaphore caps concurrent Gemini calls (caption + embed) at 8: paradoxically faster overall than unlimited `asyncio.gather`, because it avoids 429s and the 2/4/8/16 s exponential-backoff retries that follow them.

**OCR via Gemini Vision (opt-in)**
Scanned and handwritten PDFs have no embedded text — PyMuPDF returns empty pages and the index ends up empty. When the user enables "OCR scanned pages" at upload, every page where PyMuPDF returned ≤ 50 chars is rendered at 200 DPI and sent to Gemini 2.5 Flash with `response_mime_type="application/json"` to get back structured `[{text, box}, ...]` data — text feeds the chunker normally, bboxes get stored in a separate `ocr_lines` table. Tesseract was rejected because it's poor at handwriting (the primary case here) and would still need Vision as an escalation, doubling complexity. The viewer transparently switches between text-layer highlighting (typed pages) and bbox-overlay highlighting (OCR'd pages) per page, so a mixed PDF works without a second code path.

**Markdown rendering in answers**
Gemini's responses use markdown — bold headings, bullet lists, inline code. Tokens stream into the bubble as they arrive and are re-rendered through marked.js on every token, so the answer appears progressively as formatted text rather than raw `**` syntax. On the `done` event, `renderFinalAnswer` re-parses the full text and replaces inline `[Chunk N]` references with interactive citation badges; this final pass runs once instead of per-token, so citations only become clickable when the answer is complete.

**RAG pipeline status indicator**
The backend emits `{"type": "status", "text": "..."}` SSE events before each slow step (`Embedding query…`, `Searching database…`, `Generating answer…`). The chat shows this next to the spinner so the user sees what the system is doing instead of staring at an idle indicator. The status is replaced by the streamed answer as soon as the first generation token arrives.

**Cross-page chunk highlighting**
A single chunk can span two PDF pages. The viewer highlights up to three pages: the stored page (always), the page before it (in case the chunk's real start is there due to math-heavy openings that confuse the page-finder), and the page after (in case the chunk spills over). Each direction uses a different strategy: the target page uses substring anchors then word-sequence matching; the preceding page searches for the chunk start working forward; the following page anchors from the chunk's end working backward, with a `localAnchors` fallback that finds words present in both the chunk text and the page text. This fallback is necessary because PyMuPDF (used at ingest) and PDF.js (used in the browser) extract math and symbols differently, so normalized forms can diverge — shared plain-English words are always reliable anchors regardless of extraction differences.
