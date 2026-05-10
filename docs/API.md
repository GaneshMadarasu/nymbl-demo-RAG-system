<!-- generated-by: gsd-doc-writer -->
# API Reference

REST + Server-Sent Events (SSE) API exposed by the FastAPI app in `backend/main.py`.
The app is mounted at the root path; in development it is reachable at
`http://localhost:8000` (see `docs/CONFIGURATION.md` for the host/port the
project's run command binds to). All endpoints are defined in a single module —
`backend/main.py` is the source of truth.

## Authentication

**None.** The API has no authentication, no API keys, no session cookies, and no
authorization headers. The service is designed to run as a local single-user
demo: it serves the frontend, accepts a PDF upload, indexes it into the
co-located Postgres database, and answers questions about it. There is no
multi-user model, no per-request identity, and no rate limiting.

External Gemini API calls (embeddings + generation) made *by the backend* use
the `GEMINI_API_KEY` environment variable described in `docs/CONFIGURATION.md`;
clients of this API never see or supply that key.

<!-- VERIFY: Public/production deployments must add their own auth layer; nothing in the repo enforces one. -->

## Base URL

| Environment | Base URL |
|-------------|----------|
| Local dev   | `http://localhost:8000` |
| Production  | <!-- VERIFY: no production base URL is defined in the repository --> |

## Endpoints overview

`backend/main.py` defines 13 routes. Eleven are JSON / static-file routes; the
two streaming routes (`POST /ingest`, `POST /query`) speak SSE.

| Method | Path | Description | Streaming |
|--------|------|-------------|-----------|
| GET    | `/`                          | Serve the chat UI (`frontend/index.html`).          | No  |
| GET    | `/viewer`                    | Serve the PDF viewer page (`frontend/viewer.html`). | No  |
| GET    | `/doc/pdf`                   | Download the currently-loaded PDF.                  | No  |
| GET    | `/doc/chunk/{chunk_index}`   | Fetch the text + page number of a single chunk.     | No  |
| GET    | `/doc/images`                | List every image chunk for the loaded doc (no bytes). | No |
| GET    | `/image-viewer`              | Serve the image viewer page (`frontend/image-viewer.html`). | No |
| GET    | `/image/{chunk_index}`       | Fetch the raw image bytes for an image chunk.       | No  |
| GET    | `/image/{chunk_index}/meta`  | Fetch image metadata (caption, page, mime).         | No  |
| GET    | `/health`                    | Liveness probe.                                     | No  |
| GET    | `/doc/info`                  | Whether a doc is loaded and its chunk count.        | No  |
| POST   | `/ingest`                    | Upload + index a PDF. Streams progress.             | SSE |
| DELETE | `/doc`                       | Clear the loaded document and truncate `chunks`.    | No  |
| POST   | `/query`                     | Ask a question. Streams retrieval status + tokens.  | SSE |

## Request / response conventions

- All non-streaming responses are JSON unless noted (file routes return binary
  payloads).
- Streaming endpoints return `Content-Type: text/event-stream`. Each event is a
  single line of the form `data: <JSON>\n\n` (the standard SSE wire format
  produced by `StreamingResponse` in `backend/main.py`).
- HTTP errors are returned by the global exception handler at `backend/main.py:61`
  as `{"error": "<detail message>"}` with the appropriate HTTP status code.

---

## GET `/`

Serve the chat UI.

- **Response:** `200 OK` — `frontend/index.html` (`text/html`).
- **Errors:** `500` if the file is missing on disk.

## GET `/viewer`

Serve the PDF viewer single-page app. Sends `Cache-Control: no-cache, no-store, must-revalidate`
so the viewer always reloads fresh after a new ingest.

- **Response:** `200 OK` — `frontend/viewer.html` (`text/html`).

## GET `/doc/pdf`

Download the currently-loaded PDF. The PDF is cached in `tempfile.gettempdir()/docrag_current.pdf`
on a successful ingest (`backend/main.py:33`, `:206`).

- **Response:** `200 OK` — `application/pdf`, `Content-Disposition: attachment; filename="document.pdf"`.
- **Errors:**
  - `404` `{"error": "No document loaded."}` — no doc indexed.
  - `404` `{"error": "PDF not available — re-upload the document."}` — the temp file was evicted (e.g. server restart on macOS where `/tmp` is volatile).

## GET `/doc/chunk/{chunk_index}`

Return the body text + 1-indexed page number for a single chunk.

- **Path params:** `chunk_index` (integer) — chunk index as stored in the DB
  (returned in `sources.chunks[].chunk_index` from `/query`).
- **Response:** `200 OK`
  ```json
  {
    "text": "…full chunk text…",
    "page_number": 47
  }
  ```
- **Errors:**
  - `404` `{"error": "No document loaded."}`
  - `404` `{"error": "Chunk 123 not found."}`

## GET `/doc/images`

List every image chunk for the loaded doc with its caption and adjacent-page
text. Used by the chat UI to fuzzy-link painting names mentioned in answers
even when the image chunk was not retrieved (`backend/main.py:106`).

- **Response:** `200 OK` — JSON array (empty `[]` if no doc is loaded):
  ```json
  [
    {
      "chunk_index": 142,
      "page_number": 23,
      "caption": "Bust portrait of a man in a dark coat against an ochre background.",
      "page_text": "…prev/this/next page text concatenated…"
    }
  ]
  ```
- **Errors:** none — returns `[]` rather than 404 when no doc is loaded.

## GET `/image-viewer`

Serve the image viewer single-page app. Same `no-cache` headers as `/viewer`.

- **Response:** `200 OK` — `frontend/image-viewer.html` (`text/html`).

## GET `/image/{chunk_index}`

Return the raw image bytes for an image chunk.

- **Path params:** `chunk_index` (integer).
- **Response:** `200 OK` — image bytes with `Content-Type` set from the stored
  `image_mime` (typically `image/png`, `image/jpeg`, `image/webp`, or `image/gif`).
- **Errors:**
  - `404` `{"error": "No document loaded."}`
  - `404` `{"error": "Image for chunk 123 not found."}` — chunk exists but is not an image chunk, or has no stored bytes.

## GET `/image/{chunk_index}/meta`

Return metadata for an image chunk without fetching the bytes.

- **Path params:** `chunk_index` (integer).
- **Response:** `200 OK`
  ```json
  {
    "caption": "Bust portrait of a man…",
    "page_number": 23,
    "mime": "image/jpeg"
  }
  ```
- **Errors:** same as `GET /image/{chunk_index}`.

## GET `/health`

Liveness probe. Always returns `{"status": "ok"}` — does not check DB connectivity.

- **Response:** `200 OK` — `{"status": "ok"}`.

## GET `/doc/info`

Report whether a doc is loaded and its chunk count.

- **Response:** `200 OK`
  - When no doc is loaded:
    ```json
    { "loaded": false }
    ```
  - When a doc is loaded:
    ```json
    {
      "loaded": true,
      "doc_id": "a3f1c2…",
      "chunk_count": 482,
      "embedding_dim": 768
    }
    ```

  `embedding_dim` is hard-coded to `768` (`backend/main.py:169`); it matches
  the `gemini-embedding-2` output dimension and the `vector(768)` column type.

## POST `/ingest`

Upload a PDF, run the full indexing pipeline (text extract → optional OCR →
optional hand-drawn markup detection → chunking → embedding → optional image
extraction + captioning + caption-embedding → DB insert), and stream progress
events. Replaces any previously-indexed document — `chunks` is truncated
mid-ingest.

### Request

`multipart/form-data` with the following fields:

| Field            | Type     | Required | Default | Description |
|------------------|----------|----------|---------|-------------|
| `file`           | file     | yes      | —       | The PDF. Filename must end in `.pdf` (case-insensitive). Maximum **500 MB** (`_MAX_UPLOAD_BYTES` at `backend/main.py:173`). |
| `process_images` | boolean  | no       | `true`  | If `true`, extract embedded images and full-page renders for sparse-text pages, caption them via Gemini Vision, and embed the captions as image chunks. |
| `ocr_scanned`    | boolean  | no       | `false` | If `true`, OCR pages whose extracted text length is `<= 50` chars (`_OCR_EMPTY_PAGE_THRESHOLD`) using Gemini Vision. |
| `detect_markup`  | boolean  | no       | `false` | If `true`, scan pages for hand-drawn underlines/highlights/circles/notes and append the result to each affected page's text before chunking. |
| `markup_pages`   | string   | no       | `""`    | Optional 1-indexed page filter for `detect_markup`, e.g. `"1-50, 100, 200-220"`. Empty / unparseable means "all text-bearing pages". |

### Response

`200 OK`, `Content-Type: text/event-stream`. Each event is `data: <JSON>\n\n`.

The pipeline emits a sequence of `status` events (one per phase), then a final
`done` (success) or `error` (failure) event. Phases are skipped when their
toggle is off or the input has nothing for them to do — for example a
text-only PDF with `process_images=false`, `ocr_scanned=false`,
`detect_markup=false` skips the `ocr`, `markup`, `extracting_images`,
`captioning`, and `embedding_images` events entirely.

#### Status events emitted by `/ingest`

Every event is a JSON object with a `status` field. Source: `backend/ingest.py:run_ingest`.

| `status`             | Additional fields | When |
|----------------------|-------------------|------|
| `extracting`         | `message`                     | Always — start of pipeline. |
| `ocr`                | `message`                     | Only if `ocr_scanned=true` and at least one sparse-text page exists. |
| `markup`             | `message`                     | Only if `detect_markup=true` and at least one target page has body text. |
| `chunking`           | `message`                     | Always (after extraction/OCR/markup). |
| `clearing`           | `message`                     | Always — `chunks` table is truncated here. |
| `embedding`          | `message`, `progress` (`0`)   | Always — embeds every text chunk. |
| `extracting_images`  | `message`                     | Only if `process_images=true`. |
| `captioning`         | `message`                     | Only if `process_images=true` and at least one image was extracted. |
| `embedding_images`   | `message`                     | Only if at least one non-blank caption survived filtering. |
| `done`               | `doc_id`, `chunk_count`, `image_count`, `k`, `message` | Final success event. |
| `error`              | `error`                       | Final failure event — emitted on any unhandled exception in the pipeline. |

There is also a fast-path `done` event emitted *immediately* (no other events
preceding it) when the uploaded PDF's content hash matches an already-indexed
document. It carries an extra field:

```json
{
  "status": "done",
  "doc_id": "a3f1c2…",
  "chunk_count": 482,
  "k": 12,
  "cached": true,
  "message": "Document already indexed — loaded from cache"
}
```

The `done` event from a fresh ingest has no `cached` field but does include
`image_count`:

```json
{
  "status": "done",
  "doc_id": "a3f1c2…",
  "chunk_count": 482,
  "image_count": 17,
  "k": 12,
  "message": "Done — 465 text + 17 image chunks stored"
}
```

`k` is the adaptive top-`k` chosen for this document by `compute_params()`
(`backend/ingest.py:161`) based on total token count — it is the same `k`
the backend will use for retrieval on subsequent `/query` calls.

#### Error semantics

- The HTTP response itself is `200 OK` even when ingestion fails — failures
  surface as a final `data: {"status": "error", "error": "<message>"}\n\n`
  event inside the stream (`backend/main.py:208`).
- Pre-stream validation errors (non-PDF filename, > 500 MB body) are
  returned as conventional HTTP errors before the stream opens:
  - `400` `{"error": "Only PDF files are supported."}`
  - `413` `{"error": "File too large (NNN MB). Maximum is 500 MB."}`
- Per-page failures inside the pipeline (OCR failure on one page, captioning
  failure on one image) are swallowed and logged — they never abort the
  ingest. See `backend/ingest.py` for the retry policy (6 attempts with
  exponential backoff: 2, 4, 8, 16, 32, 64 s on `429` / `503` / timeouts).

### Example

```bash
curl -N -X POST http://localhost:8000/ingest \
  -F "file=@my-document.pdf" \
  -F "process_images=true" \
  -F "ocr_scanned=false" \
  -F "detect_markup=false"
```

Sample stream (text-only mode):

```
data: {"status": "extracting", "message": "Extracting text from PDF…"}

data: {"status": "chunking", "message": "Splitting into chunks…"}

data: {"status": "clearing", "message": "Clearing previous document…"}

data: {"status": "embedding", "message": "Embedding 482 chunks…", "progress": 0}

data: {"status": "done", "doc_id": "a3f1c2b8…", "chunk_count": 482, "image_count": 0, "k": 12, "message": "Done — 482 text + 0 image chunks stored"}
```

## DELETE `/doc`

Clear the currently-loaded document. Truncates the `chunks` table
(`db.clear_all_chunks` issues `TRUNCATE TABLE chunks RESTART IDENTITY`),
resets in-process state, and deletes the cached PDF from the temp directory.

- **Response:** `200 OK` — `{"cleared": true}`.
- **Errors:** none under normal operation.

> Note: This endpoint does **not** delete `doc_meta` rows. The next
> startup's `lifespan` hook will see a stale `doc_meta` entry and try to
> restore it; because `chunks` is empty, queries will produce no
> retrieval results. Re-upload the same or a new PDF to fully recover.

## POST `/query`

Ask a question about the loaded document. Runs the full query pipeline
(optional pronoun-resolving rewrite using the conversation history → embed
the question → hybrid retrieval (dense pgvector + BM25, fused via RRF in
SQL) → multimodal Gemini 2.5 Flash answer) and streams the result.

### Request

`Content-Type: application/json`. Body schema (Pydantic `QueryRequest`,
`backend/main.py:226`):

```json
{
  "question": "Who painted the bust portrait on page 23?",
  "history": [
    { "role": "user",      "content": "Tell me about chapter 4." },
    { "role": "assistant", "content": "Chapter 4 covers… [Chunk 7]" }
  ]
}
```

| Field      | Type            | Required | Description |
|------------|-----------------|----------|-------------|
| `question` | string          | yes      | The user's natural-language question. |
| `history`  | `list[dict]`    | no (defaults to `[]`) | Prior chat turns. Each item is `{"role": "user" \| "assistant", "content": "..."}`. Only the last 6 turns are used (`backend/query.py:84`, `:111`). |

If `history` is non-empty *and* the question contains a pronoun
(`it`, `this`, `they`, `here`, … see `_PRONOUN_RE` at `backend/query.py:74`)
the backend rewrites the question to be self-contained before retrieval.

### Response

`200 OK`, `Content-Type: text/event-stream`. Each event is `data: <JSON>\n\n`.

#### Event types emitted by `/query`

Every event is a JSON object with a `type` field. Source: `backend/query.py:run_query`.

| `type`     | Additional fields | When |
|------------|-------------------|------|
| `status`   | `text`            | One per phase: `"Rewriting query…"` (only when pronoun rewrite fires), `"Embedding query…"`, `"Searching database…"`. |
| `sources`  | `chunks` (array)  | Once, after retrieval, before any `token` event. Lists the retrieved chunks the model is about to see. Omitted entirely when retrieval returns no chunks. |
| `token`    | `text`            | Many — incremental text chunks of the streaming Gemini answer. Concatenate in order to build the full answer. When retrieval returns nothing, a single `token` event with `text: "I don't know."` is emitted instead. |
| `done`     | (none)            | Final success event — the answer is complete. |
| `error`    | `message`         | Final failure event — emitted on any unhandled exception in the pipeline (`backend/main.py:246`). |

#### `sources` event shape

```json
{
  "type": "sources",
  "chunks": [
    {
      "position": 1,
      "chunk_index": 142,
      "chunk_type": "image",
      "text": "Bust portrait of a man…",
      "similarity": 0.812,
      "rrf_score": 0.032787,
      "page_number": 23
    }
  ]
}
```

- `position` — 1-indexed label matching the `[Chunk N]` references the LLM is
  instructed to cite in its answer (`backend/query.py:117`). The model is
  prompted to *only* cite numbers that appear in this list.
- `chunk_index` — DB-level index used by `/doc/chunk/{n}`, `/image/{n}`, and
  `/image/{n}/meta` for follow-up fetches.
- `chunk_type` — `"text"` or `"image"`. For image chunks, the text field
  carries the Vision-generated caption.
- `similarity` — cosine similarity from the dense pgvector lane (3 d.p.).
- `rrf_score` — Reciprocal Rank Fusion score from the hybrid query (6 d.p.).
- `page_number` — 1-indexed page in the source PDF.

#### Token streaming

Tokens stream as Gemini emits them:

```
data: {"type": "status", "text": "Embedding query…"}

data: {"type": "status", "text": "Searching database…"}

data: {"type": "sources", "chunks": [ … ]}

data: {"type": "token", "text": "The "}

data: {"type": "token", "text": "bust portrait on "}

data: {"type": "token", "text": "page 23 "}

…

data: {"type": "done"}
```

Concatenating every `token.text` in order yields the complete answer. The
answer may contain `[Chunk N]` citations (where `N` is a `position` value
from the `sources` event) and markdown image links of the form
`[Title](image:CHUNK_INDEX)` that the frontend renders as clickable image
previews — see `SYSTEM_PROMPT` at `backend/query.py:21` for the exact
contract.

#### Error semantics

- If no document is loaded, the request fails *before* the stream opens:
  `400` `{"error": "No document loaded. Upload a PDF first."}`.
- Errors during streaming surface as a final `data: {"type": "error", "message": "<msg>"}\n\n`
  event; the HTTP status remains `200 OK`.

### Example

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the author say about chapter 4?", "history": []}'
```

## Error codes

The global exception handler (`backend/main.py:61`) renders every
`HTTPException` as `{"error": "<detail>"}` with the matching status code.

| Status | Where | Meaning |
|--------|-------|---------|
| `200` | All success paths, including streaming responses where errors are surfaced as in-stream events. | OK. |
| `400` | `POST /ingest` (non-PDF filename), `POST /query` (no doc loaded). | Bad request. |
| `404` | `GET /doc/pdf`, `GET /doc/chunk/{n}`, `GET /image/{n}`, `GET /image/{n}/meta`. | No document loaded, or the requested chunk / image is missing. |
| `413` | `POST /ingest` (body > 500 MB). | Payload too large. |
| `500` | Any unhandled exception in a non-streaming handler (FastAPI default). | Internal server error. |

In-stream errors (after the SSE response has started) are reported as a
final `error` event inside the stream rather than as an HTTP error. Clients
must inspect every event's `status` (for `/ingest`) or `type` (for `/query`)
field to detect failure.

## Rate limits

**None at the application layer.** No rate-limiting middleware is configured
in `backend/main.py`, and no rate-limiting library appears in
`requirements.txt`. The service relies entirely on:

- The single-user / single-document design (one indexed doc at a time;
  `/ingest` truncates `chunks` on every successful upload).
- Bounded concurrency on outbound Gemini calls inside the ingest pipeline:
  `_GEMINI_CONCURRENCY = 8` for embed/caption/OCR and `_MARKUP_CONCURRENCY = 32`
  for visual-markup detection (`backend/ingest.py:47`, `:52`). These cap
  per-document fan-out, not per-client request rate.
- Built-in exponential-backoff retries (6 attempts, 2/4/8/16/32/64 s) on
  upstream `429` / `503` / timeout errors in `backend/ingest.py` and
  `backend/query.py`.

<!-- VERIFY: Public deployments should add an HTTP-layer rate limiter — none is enforced by this codebase. -->
