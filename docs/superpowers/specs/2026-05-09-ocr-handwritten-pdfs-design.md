# OCR for scanned / handwritten PDFs — design

> **Status: Historical — superseded by current implementation.** This document records the design as planned on **2026-05-09**. The shipped OCR feature **deliberately abandoned** the structured-bbox/JSON output path described here in favor of verbatim text only — the `ocr_lines` table, `insert_ocr_lines` / `get_ocr_lines` helpers, and the `GET /doc/ocr-lines/{page_num}` endpoint were never built (the spec's bbox payload added cost without UX value on a single-document demo). For the current OCR design see [docs/ARCHITECTURE.md](../../ARCHITECTURE.md) and the "OCR via Gemini Vision" section of [README.md](../../../README.md). Preserved here as a record of the original design intent.

**Date:** 2026-05-09
**Status:** Approved (historical) — partially implemented; bbox path abandoned
**Scope:** Additive feature on top of the existing RAG pipeline. No removal or rework of current behavior.

## Goal

Let users upload PDFs whose pages contain only scanned content (handwritten notebooks, photocopies, mixed typed-and-scanned documents) and have the system retrieve, answer, and cite over the transcribed text — including a visual highlight overlay on cited scanned pages.

## User-visible behavior

1. **Upload modal** — adds a third toggle, **"OCR scanned pages"**, independent of the existing "Process images" toggle. Any combination of (process_images, ocr_scanned) is valid.
2. **Per-page detection** — when OCR is on, every page where PyMuPDF's text extraction returns ≤ 50 non-whitespace characters is rendered and OCR'd via Gemini Vision. Pages with substantive embedded text are not OCR'd.
3. **Mixed PDFs work transparently** — typed pages keep their PyMuPDF text; scanned pages get OCR'd text. Both flow through the same chunker, embedder, and retriever.
4. **Citations on OCR'd pages** — clicking a citation pill opens the PDF viewer at the cited page and renders translucent yellow boxes over the OCR'd lines that match the cited chunk text.

## Why Gemini Vision OCR (and not Tesseract or a hybrid)

| Approach | Verdict |
|---|---|
| **A. Gemini Vision OCR** | **Chosen.** Already in the stack (`gemini-2.5-flash`). Best-in-class at handwriting and multilingual content. Returns structured JSON with bounding boxes natively, which we need for the overlay highlight. Reuses existing infrastructure: `_downsize_for_vision`, `_gather_bounded` semaphore (`_GEMINI_CONCURRENCY=8`), `_is_retryable` exponential backoff. Zero new dependencies. |
| B. Tesseract (local) | Rejected. Poor at handwriting — the primary use case here. Adds a system binary dependency (Tesseract) plus a Python package. Would need a separate path for handwriting anyway, doubling complexity. |
| C. Hybrid (Tesseract first, escalate to Vision) | Rejected. Most complex of the three. Needs a confidence heuristic to decide when to escalate. Two failure modes to debug. YAGNI for this project's scale. The latency saving on printed scans is small in absolute terms and disappears as soon as a single handwritten page is in the doc. |

If Vision quality or cost becomes a problem in production, B or C can be added later without rearchitecting the pipeline — `_ocr_one_page` is a single seam.

## Architecture

OCR is a *pre-step* before chunking. Its job is to fill in missing page text so nothing downstream changes:

```
Upload PDF (with "OCR scanned pages" toggle ON)
        │
        ▼
┌──────────────────────────────────────────────┐
│  _extract_text() — PyMuPDF                   │
│    pages: list[str]   (some may be empty)    │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  NEW: _ocr_empty_pages(pdf_bytes, pages)     │
│    For each page where len(text.strip()) ≤ 50│
│      render @ 200 DPI → Gemini Vision OCR    │
│        returns (text, [{text, box}, ...])    │
│    Replace pages[i] with transcript          │
│    Collect ocr_page_set + ocr_lines map      │
└──────────────────────────────────────────────┘
        │
        ├──► ocr_lines persisted to ocr_lines table
        │
        ▼
text = "\n\n".join(pages)   ← unchanged from here on
        │
        ▼
   chunking → embedding → DB insert (existing path)
```

**Key properties:**
- OCR'd text is indistinguishable from native text downstream — same chunker, same embedding model, same `chunks` table, same retrieval.
- The `pages` list stays 1-indexed and complete. `_find_chunk_page` keeps working unchanged.
- `ocr_page_set` is passed to `_extract_images` so the stage-2 full-page render is skipped for OCR'd pages (otherwise the same scan would be extracted as both text and an image).
- Stage-1 image extraction (raw embedded images) still runs on OCR'd pages — preserves real figures embedded in scanned reports.

## Components

### Backend (`backend/ingest.py`)

**New constants:**

```python
_OCR_PAGE_DPI = 200                # higher than _FULLPAGE_DPI (150) — text needs sharper render
_OCR_EMPTY_PAGE_THRESHOLD = 50     # pages with ≤ this many chars trigger OCR
_OCR_PROMPT = (
    "Transcribe ALL visible text in this image (printed and handwritten) as JSON. "
    "Output an array of objects, one per line of text, in reading order: "
    '[{"text": "...", "box": [ymin, xmin, ymax, xmax]}, ...] '
    "Coordinates are normalized 0-1000. Preserve line breaks faithfully. "
    "Do not include commentary. If unreadable, return []."
)
```

**New functions:**

```python
def _render_page(pdf_bytes: bytes, page_num: int, dpi: int) -> tuple[bytes, str]:
    """Render one PDF page to PNG bytes. Returns (bytes, mime)."""

async def _ocr_one_page(
    pdf_bytes: bytes, page_num: int
) -> tuple[str, list[dict]]:
    """Render a page and OCR it via Gemini Vision with structured JSON output.
    Returns (full_text, lines) where lines = [{"text": str, "box": [int,int,int,int]}].
    On failure or 'unreadable', returns ('', [])."""

async def _ocr_empty_pages(
    pdf_bytes: bytes, pages: list[str]
) -> tuple[list[str], set[int], dict[int, list[dict]]]:
    """For each page in `pages` whose stripped text length is ≤ threshold, OCR it.
    Returns (updated_pages, ocr_page_nums, ocr_lines_by_page)."""
```

**Modified functions:**

| Function | Change |
|---|---|
| `run_ingest()` | Accept new arg `ocr_scanned: bool = False`. After `_extract_text`, if true, call `_ocr_empty_pages` and use the updated pages to build `text`. Persist `ocr_lines` via a new db helper. Yield `{"status": "ocr", "message": "OCRing N scanned pages…"}` SSE event. |
| `_extract_images()` | Accept `skip_pages: set[int] = None`. Skip stage-2 full-page render for those pages. Stage 1 (embedded) still runs. |

**Reused infrastructure (unchanged):**
- `_downsize_for_vision()` — same 1024 px / JPEG-85 path. (Note: for OCR we may want a larger longest-side to preserve text legibility — see "Open questions".)
- `_gather_bounded()` with `_GEMINI_CONCURRENCY=8` — caps concurrent OCR calls.
- `_is_retryable()` + `_RETRY_BASE` exponential backoff — handles 429/503/UNAVAILABLE/timeouts.
- `chunk_text()`, `_embed_one()`, `db.insert_chunks()` — totally untouched.

### Database (`backend/db.py` + migration)

**New table:**

```sql
CREATE TABLE ocr_lines (
    doc_id    TEXT       NOT NULL,
    page_num  INTEGER    NOT NULL,
    line_idx  INTEGER    NOT NULL,
    text      TEXT       NOT NULL,
    bbox      INTEGER[]  NOT NULL,    -- length 4: [ymin, xmin, ymax, xmax], 0-1000
    PRIMARY KEY (doc_id, page_num, line_idx)
);
```

**New helpers:**

```python
async def insert_ocr_lines(pool, doc_id: str, lines_by_page: dict[int, list[dict]]) -> None
async def get_ocr_lines(pool, doc_id: str, page_num: int) -> list[dict]
```

`clear_all_chunks` is extended (or paired with a `clear_all_ocr_lines`) so re-uploading a different document drops stale OCR rows alongside chunks. The doc-cache skip path (existing) needs no change — if the doc is already indexed, `ocr_lines` is already there.

### API (`backend/main.py`)

**Modified:** the upload endpoint accepts a new form field `ocr_scanned: bool = Form(False)` and passes it through to `run_ingest`.

**New endpoint:**

```python
@app.get("/doc/ocr-lines/{page_num}")
async def get_ocr_lines_for_page(page_num: int) -> list[dict]:
    """Return OCR'd lines + bboxes for one page, or [] if the page is not OCR'd.
    Viewer fetches this once per cited OCR'd page."""
```

### Frontend

**Upload modal (`frontend/index.html` or wherever the modal lives):**
- Add a third toggle/checkbox "OCR scanned pages", independent of the existing "Process images" toggle.
- Submit the value as `ocr_scanned` form field on upload.

**PDF viewer (`/viewer`):**
On opening to a cited chunk, for the cited page (and ±1, matching the existing cross-page highlight logic):
1. `fetch('/doc/ocr-lines/' + pageNum)`.
2. If non-empty, the page is OCR'd:
   - For each line whose normalized `text` is a substring of the normalized chunk text (`" ".join(s.split())` — same rule used today), compute its position on the rendered canvas:
     - `top = box[0] / 1000 * canvasHeight`
     - `left = box[1] / 1000 * canvasWidth`
     - `height = (box[2] - box[0]) / 1000 * canvasHeight`
     - `width = (box[3] - box[1]) / 1000 * canvasWidth`
   - Append a translucent yellow `<div>` (rgba(255, 235, 0, 0.4) or similar) at those coordinates to the page's overlay layer.
3. If the response is empty, fall back to the existing PDF.js text-layer highlight path.

This makes the viewer transparently do the right thing per page — typed pages use text-layer highlighting (unchanged), OCR'd pages use bbox overlays, and a mixed PDF mixes both correctly.

## Error handling & failure modes

**Per-page OCR failures are isolated.** `_ocr_one_page` catches all exceptions and returns `('', [])`. One page failing doesn't abort ingest — that page stays empty in the `pages` list and contributes no chunks (same end-state as if OCR were off). A warning is logged with the page number.

**Retry behavior.** Same as captioning today: up to 6 retries on 429/503/UNAVAILABLE/timeouts with exponential backoff (2, 4, 8, 16, 32, 64 s). Non-retryable errors propagate after 1 attempt and surface via the existing normalized error response.

**The "unreadable" / empty-array response.** If Vision returns `[]` or non-JSON content matching `unreadable`, treat as `('', [])` — page contributes no chunks rather than embedding the literal word "unreadable".

**Malformed JSON.** Catch JSON parse errors, log warning, return `('', [])`. Ingest does not crash.

**Partial JSON (line missing `box`).** Keep the text; drop only that line entry from `lines`. Page contributes chunks normally; just no overlay on those lines.

| Scenario | Behavior |
|---|---|
| Typed PDF, OCR toggle on | Every page has text → no OCR calls. No-op. |
| Scanned PDF, OCR toggle off | Existing behavior — empty pages, no chunks. Out of scope to fix. |
| Vision rate-limits hard (all retries exhausted) | That page contributes no chunks. Ingest completes; logs show which pages failed. |
| Page render fails (corrupt PDF page) | Logged warning; page contributes no chunks; ingest continues. |
| All OCR pages fail | Ingest completes with whatever non-OCR pages produced. If zero, `total=0` is stored — same edge case as uploading a fully-empty PDF today. |
| OCR succeeds, bboxes malformed | Page contributes chunks; viewer falls back to "no highlight" for that page. |

**SSE progress events.** A new `{"type": "status", "text": "OCRing N scanned pages…"}` event fires before the OCR step, matching the existing `Embedding query…` pattern in the chat status indicator.

## Testing

**Unit tests (`tests/`):**

| Test | What it verifies |
|---|---|
| `test_ocr_threshold` | `_ocr_empty_pages` only OCRs pages where `len(text.strip()) ≤ 50`; substantive pages skipped. Fake `_ocr_one_page`. |
| `test_ocr_returns_lines_with_bboxes` | Fake Vision JSON response → lines + bboxes parsed correctly. |
| `test_ocr_malformed_json_falls_back_to_empty` | Vision returns garbage; ingest does not crash; page contributes no chunks. |
| `test_ocr_unreadable_filtered` | Fake OCR returns `[]`; that page contributes nothing. |
| `test_ocr_failure_isolated` | Fake `_ocr_one_page` raises on page 3; other pages still get transcripts; ingest completes. |
| `test_extract_images_skips_pages` | `_extract_images(pdf_bytes, skip_pages={2,3})` produces no stage-2 renders for those pages; stage 1 still runs. |
| `test_run_ingest_with_ocr_off` | Default `ocr_scanned=False` path is byte-identical to today's behavior — regression guard. |
| `test_run_ingest_with_ocr_on_no_empty_pages` | OCR on + fully-typed PDF → zero Vision OCR calls. |
| `test_get_ocr_lines_endpoint` | Returns expected lines for an OCR'd page, empty list for a typed page. |

**Integration test (`tests/test_integration_ocr.py`, `@pytest.mark.integration`, gated on `GEMINI_API_KEY`):**

Fixture: a small PDF with one typed page and one rendered handwritten/scanned page. Run real ingest with `ocr_scanned=True`. Assert:
- Chunks exist for both pages.
- The OCR'd page's chunk text contains an expected keyword from the handwriting sample.
- Citation page numbers are correct.
- `ocr_lines` table has rows for the scanned page.

**Manual smoke test (run before merging):**

1. Typed PDF, OCR off — ingest matches baseline timing and chunk count.
2. Same typed PDF, OCR on — same chunk count, same timing (no OCR fired).
3. Scanned/handwritten PDF, OCR off — zero chunks, retrieval returns "I don't know".
4. Same scanned PDF, OCR on — chunks created, query returns grounded answer, citation pill opens the correct page with translucent yellow overlays on the cited lines.
5. Mixed PDF (typed + scanned), OCR on — both content types retrievable; typed-page citations highlight via text layer (existing path), scanned-page citations highlight via bbox overlay (new path).

## Open questions / follow-ups (not blocking)

- **Render DPI tuning.** 200 DPI is a starting point. Handwriting may need 300; printed scans may be fine at 150. Adjust after first integration test.
- **Vision downsize for OCR.** The existing `_CAPTION_MAX_DIM=1024` may degrade text legibility. Either bump it for the OCR path (e.g. 1600) or skip downsize entirely for OCR'd renders. Decide after seeing real Vision output.
- **Caching OCR results across uploads.** Currently the SHA-256 doc-id cache skips re-embedding, which already covers re-uploads. No new caching needed unless OCR cost dominates wall time for repeated work.
- **Chunk-to-line mapping accuracy.** Substring match per OCR'd line is the simplest mapping. If many lines wrap, fall back to longest-common-subsequence over normalized words. Defer until we see real overlay misses.
