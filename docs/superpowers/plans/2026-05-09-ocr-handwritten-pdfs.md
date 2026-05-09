# OCR for scanned/handwritten PDFs — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-09-ocr-handwritten-pdfs-design.md`

**Goal:** Add an opt-in upload toggle that auto-detects empty-text pages, OCRs them with Gemini Vision (structured JSON with line bboxes), feeds the transcribed text through the existing chunker/embedder/retriever, and renders translucent yellow overlays on cited lines in the PDF viewer.

**Architecture:** OCR is a *pre-step* between PyMuPDF text extraction and chunking — it fills in missing page text so nothing downstream changes. Per-line bounding boxes from Vision are stored in a new `ocr_lines` table and read by the viewer to render an absolutely-positioned overlay layer on top of the rendered PDF.js canvas.

**Tech Stack:** Python 3.11, FastAPI, asyncpg, Postgres 16 + pgvector, Gemini 2.5 Flash (already wired up), PyMuPDF, vanilla JS / PDF.js.

---

## File structure

| File | Role |
|---|---|
| `backend/db.py` | Modify: extend SCHEMA + MIGRATION with `ocr_lines` table; add `insert_ocr_lines`, `get_ocr_lines`; extend `clear_all_chunks` to also truncate `ocr_lines`. |
| `backend/ingest.py` | Modify: add `_render_page`, `_ocr_one_page`, `_ocr_empty_pages`; thread `ocr_scanned` arg through `run_ingest`; pass `skip_pages` to `_extract_images`. |
| `backend/main.py` | Modify: add `ocr_scanned` form field to `/ingest`; add `GET /doc/ocr-lines/{page_num}`. |
| `frontend/index.html` | Modify: add "OCR scanned pages" checkbox to the upload modal; send as form field. |
| `frontend/viewer.html` | Modify: per-page, fetch `/doc/ocr-lines/{n}`; if non-empty, render bbox overlays; otherwise use existing text-layer highlight path. |
| `tests/test_db.py` | Add tests for new DB helpers. |
| `tests/test_ingest.py` | Add tests for `_ocr_empty_pages`, `_extract_images(skip_pages=...)`, `run_ingest(ocr_scanned=...)`. |
| `tests/test_api.py` | Add tests for `ocr_scanned` form field and `/doc/ocr-lines/{page_num}`. |
| `tests/test_integration_ocr.py` | New: gated integration test that hits real Gemini Vision on a fixture PDF. |

---

## Task 1: DB — `ocr_lines` table, helpers, clear extension

**Files:**
- Modify: `backend/db.py:7-33` (SCHEMA), `backend/db.py:36-69` (MIGRATION), `backend/db.py:260-262` (`clear_all_chunks`)
- Modify: `backend/db.py` (append new helpers)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test for insert + get**

Append to `tests/test_db.py`:

```python
async def test_insert_and_get_ocr_lines(pool):
    from backend.db import insert_ocr_lines, get_ocr_lines

    lines_by_page = {
        2: [
            {"text": "First line on page two", "box": [10, 20, 30, 800]},
            {"text": "Second line on page two", "box": [40, 20, 60, 800]},
        ],
        5: [{"text": "Lone line on page five", "box": [100, 50, 130, 700]}],
    }
    await insert_ocr_lines(pool, "doc-test-ocr", lines_by_page)

    page2 = await get_ocr_lines(pool, "doc-test-ocr", 2)
    assert len(page2) == 2
    assert page2[0]["text"] == "First line on page two"
    assert page2[0]["bbox"] == [10, 20, 30, 800]
    assert page2[0]["line_idx"] == 0

    page5 = await get_ocr_lines(pool, "doc-test-ocr", 5)
    assert len(page5) == 1

    page3 = await get_ocr_lines(pool, "doc-test-ocr", 3)
    assert page3 == []


async def test_clear_all_chunks_also_clears_ocr_lines(pool):
    from backend.db import insert_ocr_lines, get_ocr_lines, clear_all_chunks

    await insert_ocr_lines(
        pool, "doc-clear-test",
        {1: [{"text": "x", "box": [0, 0, 10, 10]}]},
    )
    assert (await get_ocr_lines(pool, "doc-clear-test", 1)) != []

    await clear_all_chunks(pool)
    assert (await get_ocr_lines(pool, "doc-clear-test", 1)) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
make db   # if db isn't already running
pytest tests/test_db.py::test_insert_and_get_ocr_lines tests/test_db.py::test_clear_all_chunks_also_clears_ocr_lines -v
```

Expected: FAIL with `ImportError` or `relation "ocr_lines" does not exist`.

- [ ] **Step 3: Update SCHEMA in `backend/db.py`**

Replace the SCHEMA constant (lines 7-33) with:

```python
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    parent_text TEXT,
    page_number INTEGER,
    chunk_type  TEXT NOT NULL DEFAULT 'text',
    image_data  BYTEA,
    image_mime  TEXT,
    embedding   vector(768) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_meta (
    doc_id      TEXT PRIMARY KEY,
    chunk_count INTEGER NOT NULL,
    k           INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ocr_lines (
    doc_id    TEXT       NOT NULL,
    page_num  INTEGER    NOT NULL,
    line_idx  INTEGER    NOT NULL,
    text      TEXT       NOT NULL,
    bbox      INTEGER[]  NOT NULL,
    PRIMARY KEY (doc_id, page_num, line_idx)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""
```

Append a new block to MIGRATION (inside the existing `DO $$ ... END $$;`, before the closing `END`):

```sql
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'ocr_lines'
    ) THEN
        CREATE TABLE ocr_lines (
            doc_id    TEXT       NOT NULL,
            page_num  INTEGER    NOT NULL,
            line_idx  INTEGER    NOT NULL,
            text      TEXT       NOT NULL,
            bbox      INTEGER[]  NOT NULL,
            PRIMARY KEY (doc_id, page_num, line_idx)
        );
    END IF;
```

- [ ] **Step 4: Add `insert_ocr_lines` and `get_ocr_lines`**

Append to `backend/db.py`:

```python
async def insert_ocr_lines(
    pool: asyncpg.Pool,
    doc_id: str,
    lines_by_page: dict[int, list[dict]],
) -> None:
    """Insert OCR'd lines with bboxes. lines_by_page maps page_num -> list of
    {"text": str, "box": [ymin, xmin, ymax, xmax]} (coords normalized 0-1000)."""
    rows = []
    for page_num, lines in lines_by_page.items():
        for idx, line in enumerate(lines):
            rows.append(
                (doc_id, page_num, idx, line["text"], list(line["box"]))
            )
    if not rows:
        return
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO ocr_lines (doc_id, page_num, line_idx, text, bbox) "
            "VALUES ($1, $2, $3, $4, $5)",
            rows,
        )


async def get_ocr_lines(
    pool: asyncpg.Pool, doc_id: str, page_num: int
) -> list[dict]:
    """Return [{line_idx, text, bbox}] for one page, ordered by line_idx.
    Empty list if page wasn't OCR'd."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT line_idx, text, bbox FROM ocr_lines "
            "WHERE doc_id = $1 AND page_num = $2 ORDER BY line_idx",
            doc_id,
            page_num,
        )
    return [
        {"line_idx": r["line_idx"], "text": r["text"], "bbox": list(r["bbox"])}
        for r in rows
    ]
```

- [ ] **Step 5: Extend `clear_all_chunks` to also clear ocr_lines**

Replace `clear_all_chunks` (lines 260-262):

```python
async def clear_all_chunks(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
        await conn.execute("TRUNCATE TABLE ocr_lines")
```

Also update `tests/conftest.py` so the per-test cleanup truncates `ocr_lines` too. Replace the cleanup block (lines 22-25):

```python
    yield p
    async with p.acquire() as conn:
        await conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
        await conn.execute("TRUNCATE TABLE doc_meta")
        await conn.execute("TRUNCATE TABLE ocr_lines")
    await p.close()
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: all `test_db.py` tests pass, including the two new ones.

- [ ] **Step 7: Commit**

```bash
git add backend/db.py tests/test_db.py tests/conftest.py
git commit -m "feat(db): add ocr_lines table and helpers for OCR bboxes"
```

---

## Task 2: `_render_page` helper

**Files:**
- Modify: `backend/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingest.py`:

```python
def test_render_page_produces_png_bytes():
    """Builds a minimal one-page PDF, then renders it via _render_page."""
    import fitz
    from backend.ingest import _render_page

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello OCR")
    pdf_bytes = doc.write()
    doc.close()

    img_bytes, mime = _render_page(pdf_bytes, page_num=1, dpi=200)
    assert mime == "image/png"
    # PNG magic bytes
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(img_bytes) > 1000
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pytest tests/test_ingest.py::test_render_page_produces_png_bytes -v
```

Expected: FAIL with `ImportError: cannot import name '_render_page'`.

- [ ] **Step 3: Implement `_render_page`**

Add to `backend/ingest.py`, near the top with other constants:

```python
_OCR_PAGE_DPI = 200
_OCR_EMPTY_PAGE_THRESHOLD = 50
```

Add the function (place near `_extract_images`):

```python
def _render_page(pdf_bytes: bytes, page_num: int, dpi: int) -> tuple[bytes, str]:
    """Render one PDF page to PNG bytes. page_num is 1-indexed."""
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png"), "image/png"
    finally:
        doc.close()
```

- [ ] **Step 4: Run it to verify it passes**

```bash
pytest tests/test_ingest.py::test_render_page_produces_png_bytes -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): add _render_page helper for OCR pipeline"
```

---

## Task 3: `_ocr_one_page` — Vision OCR with structured JSON

**Files:**
- Modify: `backend/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
import json as _json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


async def test_ocr_one_page_parses_valid_json():
    fake_response = SimpleNamespace(text=_json.dumps([
        {"text": "Hello world", "box": [10, 20, 30, 800]},
        {"text": "Second line", "box": [40, 20, 60, 800]},
    ]))
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG\r\n\x1a\n...", "image/png")),
        patch("backend.ingest._downsize_for_vision", return_value=(b"jpegbytes", "image/jpeg")),
        patch(
            "backend.ingest._client.aio.models.generate_content",
            new_callable=AsyncMock,
            return_value=fake_response,
        ),
    ):
        from backend.ingest import _ocr_one_page
        text, lines = await _ocr_one_page(b"fake-pdf", 1)
    assert text == "Hello world\nSecond line"
    assert len(lines) == 2
    assert lines[0]["text"] == "Hello world"
    assert lines[0]["box"] == [10, 20, 30, 800]


async def test_ocr_one_page_handles_empty_array():
    fake_response = SimpleNamespace(text="[]")
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG", "image/png")),
        patch("backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")),
        patch(
            "backend.ingest._client.aio.models.generate_content",
            new_callable=AsyncMock,
            return_value=fake_response,
        ),
    ):
        from backend.ingest import _ocr_one_page
        text, lines = await _ocr_one_page(b"fake-pdf", 1)
    assert text == ""
    assert lines == []


async def test_ocr_one_page_handles_malformed_json():
    fake_response = SimpleNamespace(text="not json at all {{{")
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG", "image/png")),
        patch("backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")),
        patch(
            "backend.ingest._client.aio.models.generate_content",
            new_callable=AsyncMock,
            return_value=fake_response,
        ),
    ):
        from backend.ingest import _ocr_one_page
        text, lines = await _ocr_one_page(b"fake-pdf", 1)
    assert text == ""
    assert lines == []


async def test_ocr_one_page_drops_lines_missing_box():
    fake_response = SimpleNamespace(text=_json.dumps([
        {"text": "Has box", "box": [0, 0, 10, 100]},
        {"text": "Missing box"},
        {"text": "Has box too", "box": [20, 0, 30, 100]},
    ]))
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG", "image/png")),
        patch("backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")),
        patch(
            "backend.ingest._client.aio.models.generate_content",
            new_callable=AsyncMock,
            return_value=fake_response,
        ),
    ):
        from backend.ingest import _ocr_one_page
        text, lines = await _ocr_one_page(b"fake-pdf", 1)
    # Text preserves all three lines; lines list drops the one without bbox
    assert "Has box" in text and "Missing box" in text and "Has box too" in text
    assert len(lines) == 2
    assert all("box" in line for line in lines)


async def test_ocr_one_page_render_failure_returns_empty():
    with patch("backend.ingest._render_page", side_effect=RuntimeError("render bomb")):
        from backend.ingest import _ocr_one_page
        text, lines = await _ocr_one_page(b"fake-pdf", 1)
    assert text == ""
    assert lines == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v -k "ocr_one_page"
```

Expected: FAIL with `ImportError: cannot import name '_ocr_one_page'`.

- [ ] **Step 3: Implement `_ocr_one_page`**

Add OCR prompt near the existing `_CAPTION_PROMPT` in `backend/ingest.py`:

```python
_OCR_PROMPT = (
    "Transcribe ALL visible text in this image (printed and handwritten) as JSON. "
    "Output an array of objects, one per line of text, in reading order: "
    '[{"text": "...", "box": [ymin, xmin, ymax, xmax]}, ...] '
    "Coordinates are normalized 0-1000. Preserve line breaks faithfully. "
    "Do not include commentary or markdown fences. "
    "If the image contains no readable text, return []."
)
```

Add the function (place near `_caption_image`):

```python
import json as _json_mod  # put near the top with other imports

async def _ocr_one_page(
    pdf_bytes: bytes, page_num: int
) -> tuple[str, list[dict]]:
    """Render a PDF page and OCR it via Gemini Vision with structured JSON output.

    Returns (full_text, lines) where lines is a list of
    {"text": str, "box": [ymin, xmin, ymax, xmax]} (coords 0-1000).
    On any failure (render error, malformed JSON, all-retries-exhausted) returns
    ("", []) so one bad page never aborts the ingest."""
    try:
        page_bytes, page_mime = _render_page(pdf_bytes, page_num, _OCR_PAGE_DPI)
    except Exception as exc:
        logger.warning("OCR render failed for page %d: %s", page_num, exc)
        return "", []

    send_bytes, send_mime = _downsize_for_vision(page_bytes)
    if not send_mime:
        send_mime = page_mime

    raw_text = ""
    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.generate_content(
                model=_VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=send_bytes, mime_type=send_mime),
                    _OCR_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            raw_text = (r.text or "").strip()
            break
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning(
                    "OCR call failed for page %d (%s); retrying in %.0fs",
                    page_num, exc, wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.warning(
                "OCR gave up for page %d after %d attempts: %s",
                page_num, attempt + 1, exc,
            )
            return "", []

    if not raw_text or raw_text.lower().strip(".!? \"'") == "unreadable":
        return "", []

    try:
        parsed = _json_mod.loads(raw_text)
    except Exception as exc:
        logger.warning("OCR JSON parse failed for page %d: %s", page_num, exc)
        return "", []

    if not isinstance(parsed, list):
        return "", []

    text_parts: list[str] = []
    lines: list[dict] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        line_text = entry.get("text")
        if not isinstance(line_text, str) or not line_text.strip():
            continue
        text_parts.append(line_text)
        box = entry.get("box")
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
        ):
            lines.append({"text": line_text, "box": [int(v) for v in box]})

    return "\n".join(text_parts), lines
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v -k "ocr_one_page"
```

Expected: all five new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): add _ocr_one_page Vision OCR with structured JSON"
```

---

## Task 4: `_ocr_empty_pages` — orchestrate per-page OCR

**Files:**
- Modify: `backend/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
async def test_ocr_empty_pages_only_ocrs_below_threshold():
    pages = [
        "Substantial first page text " * 20,   # > 50 chars → not OCR'd
        "",                                     # empty → OCR'd
        "tiny",                                 # < 50 → OCR'd
        "Plenty of words on this fourth page that is well above threshold",
    ]
    captured_pages: list[int] = []

    async def fake_ocr(_pdf, page_num):
        captured_pages.append(page_num)
        return f"OCR text page {page_num}", [{"text": f"OCR text page {page_num}", "box": [0, 0, 10, 100]}]

    with patch("backend.ingest._ocr_one_page", side_effect=fake_ocr):
        from backend.ingest import _ocr_empty_pages
        updated, ocr_set, lines_by_page = await _ocr_empty_pages(b"pdf", pages)

    assert sorted(captured_pages) == [2, 3]
    assert ocr_set == {2, 3}
    assert updated[0] == pages[0]
    assert updated[1] == "OCR text page 2"
    assert updated[2] == "OCR text page 3"
    assert updated[3] == pages[3]
    assert set(lines_by_page.keys()) == {2, 3}


async def test_ocr_empty_pages_failure_isolated():
    pages = ["", "", ""]

    async def fake_ocr(_pdf, page_num):
        if page_num == 2:
            return "", []  # simulate failure -> empty
        return f"page {page_num} text", [{"text": f"page {page_num} text", "box": [0, 0, 5, 50]}]

    with patch("backend.ingest._ocr_one_page", side_effect=fake_ocr):
        from backend.ingest import _ocr_empty_pages
        updated, ocr_set, lines_by_page = await _ocr_empty_pages(b"pdf", pages)

    assert updated[0] == "page 1 text"
    assert updated[1] == ""        # failed page kept empty
    assert updated[2] == "page 3 text"
    assert ocr_set == {1, 3}       # only successfully OCR'd pages in the set
    assert set(lines_by_page.keys()) == {1, 3}


async def test_ocr_empty_pages_no_empty_pages_no_calls():
    pages = ["Lots of text " * 10, "More text " * 20]
    fake_ocr = AsyncMock()
    with patch("backend.ingest._ocr_one_page", fake_ocr):
        from backend.ingest import _ocr_empty_pages
        updated, ocr_set, lines_by_page = await _ocr_empty_pages(b"pdf", pages)

    fake_ocr.assert_not_awaited()
    assert updated == pages
    assert ocr_set == set()
    assert lines_by_page == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v -k "ocr_empty_pages"
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_ocr_empty_pages`**

Add to `backend/ingest.py`:

```python
async def _ocr_empty_pages(
    pdf_bytes: bytes, pages: list[str]
) -> tuple[list[str], set[int], dict[int, list[dict]]]:
    """For each page in `pages` whose stripped text length is below the
    threshold, OCR it via Gemini Vision. Returns (updated_pages,
    ocr_page_nums, ocr_lines_by_page).

    - updated_pages: copy of `pages` with successfully-OCR'd entries replaced
      by the transcribed text; failed pages remain as their original empty
      text.
    - ocr_page_nums: 1-indexed pages where OCR returned non-empty text.
    - ocr_lines_by_page: {page_num: [{"text", "box"}, ...]} for pages with
      bbox lines, used to populate the ocr_lines table for the viewer.
    """
    targets = [
        i + 1
        for i, p in enumerate(pages)
        if len(p.strip()) <= _OCR_EMPTY_PAGE_THRESHOLD
    ]
    if not targets:
        return list(pages), set(), {}

    logger.info("Running OCR on %d empty/sparse page(s)", len(targets))
    results = await _gather_bounded(
        [_ocr_one_page(pdf_bytes, n) for n in targets],
        _GEMINI_CONCURRENCY,
    )

    updated = list(pages)
    ocr_set: set[int] = set()
    lines_by_page: dict[int, list[dict]] = {}

    for page_num, (text, lines) in zip(targets, results):
        if not text:
            continue
        updated[page_num - 1] = text
        ocr_set.add(page_num)
        if lines:
            lines_by_page[page_num] = lines

    logger.info(
        "OCR completed: %d/%d pages produced text",
        len(ocr_set), len(targets),
    )
    return updated, ocr_set, lines_by_page
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v -k "ocr_empty_pages"
```

Expected: all three new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): add _ocr_empty_pages orchestrator for per-page OCR"
```

---

## Task 5: `_extract_images(skip_pages=...)` — avoid double-extracting OCR'd pages

**Files:**
- Modify: `backend/ingest.py:206-294` (`_extract_images`)
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingest.py`:

```python
def test_extract_images_skip_pages_blocks_stage2():
    """Build a 3-page PDF where pages 2 and 3 have no embedded images and
    very little text — both would normally trigger stage-2 full-page render.
    With skip_pages={2}, only page 3 should produce a stage-2 render."""
    import fitz
    from backend.ingest import _extract_images

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "This page has plenty of substantive text " * 5)
    p2 = doc.new_page()  # empty / sparse → would trigger stage 2
    p3 = doc.new_page()  # empty / sparse → would trigger stage 2
    pdf_bytes = doc.write()
    doc.close()

    # Without skipping, both 2 and 3 would be rendered
    all_imgs = _extract_images(pdf_bytes)
    pages_extracted = {p for p, _, _ in all_imgs}
    assert 2 in pages_extracted and 3 in pages_extracted

    # With skip_pages={2}, only page 3 produces a stage-2 render
    skipped = _extract_images(pdf_bytes, skip_pages={2})
    pages_skipped = {p for p, _, _ in skipped}
    assert 2 not in pages_skipped
    assert 3 in pages_skipped


def test_extract_images_default_skip_pages_is_none():
    """Calling without skip_pages keeps existing behavior (regression guard)."""
    import fitz
    from backend.ingest import _extract_images

    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "x")
    pdf_bytes = doc.write()
    doc.close()

    # Should not raise; should return a list (may or may not have images)
    out = _extract_images(pdf_bytes)
    assert isinstance(out, list)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v -k "extract_images_skip"
```

Expected: FAIL with `TypeError: _extract_images() got an unexpected keyword argument 'skip_pages'`.

- [ ] **Step 3: Update `_extract_images` signature and stage-2 loop**

In `backend/ingest.py`, change the function signature (line 206):

```python
def _extract_images(
    pdf_bytes: bytes,
    skip_pages: set[int] | None = None,
) -> list[tuple[int, bytes, str]]:
```

Add at the top of the function body (after the docstring, before the existing `t0 = time.monotonic()`):

```python
    skip_pages = skip_pages or set()
```

In the stage-2 loop (around line 271), change the existing skip check:

```python
    # ---------- Stage 2: full-page render for sparse-text pages with no embedded images ----------
    for page_num, page in enumerate(doc, 1):
        if page_num in pages_with_extracted:
            continue
        if page_num in skip_pages:
            continue  # caller (e.g. OCR pipeline) already handled this page
        text_len = len(page.get_text().strip())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v -k "extract_images"
```

Expected: new tests PASS, existing image-extraction tests (if any) still pass.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): _extract_images accepts skip_pages to avoid OCR/image overlap"
```

---

## Task 6: Wire `ocr_scanned` through `run_ingest`

**Files:**
- Modify: `backend/ingest.py:375-508` (`run_ingest`)
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:

```python
async def test_run_ingest_default_ocr_off_no_change(pool):
    """Regression guard: ocr_scanned defaults to False; behavior unchanged."""
    fake_text = "This is real embedded text content. " * 30
    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", return_value=(fake_text, ["page1 text"])),
        patch("backend.ingest._extract_images", return_value=[]),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch("backend.ingest._ocr_empty_pages", new_callable=AsyncMock) as m_ocr,
    ):
        events = [e async for e in run_ingest(b"pdf-bytes")]
    m_ocr.assert_not_awaited()
    assert any(e["status"] == "done" for e in events)


async def test_run_ingest_ocr_on_no_empty_pages_skips_vision(pool):
    fake_pages = ["page one substantive text " * 10, "page two substantive text " * 10]
    fake_text = "\n\n".join(fake_pages)
    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", return_value=(fake_text, fake_pages)),
        patch("backend.ingest._extract_images", return_value=[]),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch("backend.ingest._ocr_one_page", new_callable=AsyncMock) as m_ocr_one,
    ):
        events = [e async for e in run_ingest(b"pdf-bytes", ocr_scanned=True)]
    m_ocr_one.assert_not_awaited()
    assert any(e["status"] == "done" for e in events)


async def test_run_ingest_ocr_on_with_empty_pages_calls_vision_and_inserts_lines(pool):
    fake_pages = ["", "Substantial typed text on this page " * 5, ""]
    fake_text = "\n\n".join(p for p in fake_pages if p.strip())

    async def fake_ocr(_pdf, page_num):
        return (
            f"OCR'd text page {page_num} " * 4,
            [{"text": f"OCR'd text page {page_num}", "box": [0, 0, 10, 100]}],
        )

    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", return_value=(fake_text, fake_pages)),
        patch("backend.ingest._extract_images", return_value=[]) as m_imgs,
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch("backend.ingest._ocr_one_page", side_effect=fake_ocr),
    ):
        events = [e async for e in run_ingest(b"pdf-bytes", ocr_scanned=True)]

    statuses = [e["status"] for e in events]
    assert "ocr" in statuses
    done = next(e for e in events if e["status"] == "done")
    assert done["chunk_count"] > 0

    # _extract_images received skip_pages={1, 3} (the OCR'd pages)
    _, kwargs = m_imgs.call_args
    assert kwargs.get("skip_pages") == {1, 3}

    # ocr_lines were persisted for both pages
    from backend.db import get_ocr_lines
    assert (await get_ocr_lines(pool, done["doc_id"], 1)) != []
    assert (await get_ocr_lines(pool, done["doc_id"], 3)) != []
    assert (await get_ocr_lines(pool, done["doc_id"], 2)) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v -k "run_ingest_ocr or run_ingest_default_ocr"
```

Expected: FAIL — `run_ingest` doesn't accept `ocr_scanned`, no `ocr` status emitted, etc.

- [ ] **Step 3: Update `run_ingest`**

In `backend/ingest.py`, change the function signature (line 375):

```python
async def run_ingest(
    pdf_bytes: bytes,
    process_images: bool = True,
    ocr_scanned: bool = False,
) -> AsyncGenerator[dict, None]:
```

Insert the OCR step after `_extract_text` and before chunking. Replace the existing block:

```python
    yield {"status": "extracting", "message": "Extracting text from PDF…"}
    text, pages = await loop.run_in_executor(None, _extract_text, pdf_bytes)

    yield {"status": "chunking", "message": "Splitting into chunks…"}
```

with:

```python
    yield {"status": "extracting", "message": "Extracting text from PDF…"}
    text, pages = await loop.run_in_executor(None, _extract_text, pdf_bytes)

    ocr_page_set: set[int] = set()
    ocr_lines_by_page: dict[int, list[dict]] = {}
    if ocr_scanned:
        empty_count = sum(
            1 for p in pages if len(p.strip()) <= _OCR_EMPTY_PAGE_THRESHOLD
        )
        if empty_count:
            yield {
                "status": "ocr",
                "message": f"OCRing {empty_count} scanned page(s)…",
            }
            pages, ocr_page_set, ocr_lines_by_page = await _ocr_empty_pages(
                pdf_bytes, pages
            )
            text = "\n\n".join(p for p in pages if p.strip())

    yield {"status": "chunking", "message": "Splitting into chunks…"}
```

Update the call to `_extract_images` (around line 438) to pass skip_pages:

```python
        images = await loop.run_in_executor(
            None,
            lambda: _extract_images(pdf_bytes, skip_pages=ocr_page_set),
        )
```

After `await db.insert_chunks(pool, doc_id, rows)` (around line 428), persist OCR lines:

```python
    if ocr_lines_by_page:
        await db.insert_ocr_lines(pool, doc_id, ocr_lines_by_page)
```

- [ ] **Step 4: Run all ingest tests**

```bash
pytest tests/test_ingest.py -v
```

Expected: all tests PASS, including new OCR run_ingest tests and pre-existing run_ingest tests.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): wire ocr_scanned param through run_ingest"
```

---

## Task 7: API — `ocr_scanned` form field + `/doc/ocr-lines/{page_num}`

**Files:**
- Modify: `backend/main.py:176-205` (`/ingest`), append new endpoint
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
async def test_ingest_endpoint_accepts_ocr_scanned_form_field():
    from unittest.mock import AsyncMock, patch
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async def fake_run_ingest(pdf_bytes, process_images=True, ocr_scanned=False):
        # Yield a minimal done event so the endpoint completes
        yield {
            "status": "done",
            "doc_id": "abc123",
            "chunk_count": 1,
            "k": 5,
            "message": "ok",
        }
        # Capture call args via the mock wrapper below

    with patch("backend.main.ingest.run_ingest", side_effect=fake_run_ingest) as m:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/ingest",
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
                data={"process_images": "false", "ocr_scanned": "true"},
            )
    assert resp.status_code == 200
    # Verify ocr_scanned was passed through
    _, kwargs = m.call_args
    assert kwargs.get("ocr_scanned") is True


async def test_doc_ocr_lines_endpoint_returns_lines(pool):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app, _state
    from backend.db import insert_ocr_lines

    _state["doc_id"] = "doc-api-ocr"
    await insert_ocr_lines(
        pool,
        "doc-api-ocr",
        {3: [
            {"text": "first line", "box": [10, 20, 30, 800]},
            {"text": "second line", "box": [40, 20, 60, 800]},
        ]},
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/doc/ocr-lines/3")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["text"] == "first line"
    assert body[0]["bbox"] == [10, 20, 30, 800]


async def test_doc_ocr_lines_endpoint_returns_empty_for_typed_page(pool):
    from httpx import AsyncClient, ASGITransport
    from backend.main import app, _state

    _state["doc_id"] = "doc-typed"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/doc/ocr-lines/1")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v -k "ocr_scanned or ocr_lines"
```

Expected: FAIL — endpoint doesn't exist; form field not accepted.

- [ ] **Step 3: Update `/ingest` and add `/doc/ocr-lines/{page_num}`**

In `backend/main.py`, change the `/ingest` signature (line 178):

```python
@app.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    process_images: bool = Form(True),
    ocr_scanned: bool = Form(False),
) -> StreamingResponse:
```

Pass it through in the `stream` closure (replace the `async for event` line ~192):

```python
            async for event in ingest.run_ingest(
                pdf_bytes,
                process_images=process_images,
                ocr_scanned=ocr_scanned,
            ):
```

Append a new endpoint near `/doc/images` (after line 113):

```python
@app.get("/doc/ocr-lines/{page_num}")
async def get_ocr_lines_for_page(page_num: int) -> list[dict]:
    """Return OCR'd lines + bboxes for one page, or [] if the page is not OCR'd
    (or if no doc is loaded). Viewer fetches this once per cited page."""
    if not _state["doc_id"]:
        return []
    pool = await db.get_pool()
    return await db.get_ocr_lines(pool, _state["doc_id"], page_num)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_api.py
git commit -m "feat(api): add ocr_scanned form field and /doc/ocr-lines endpoint"
```

---

## Task 8: Frontend upload modal — "OCR scanned pages" checkbox

**Files:**
- Modify: `frontend/index.html:138-148` (modal markup), `frontend/index.html:273` (FormData)

- [ ] **Step 1: Read the current modal markup and FormData lines**

```bash
grep -n "ingest-modal\|process_images\|ocr_scanned" frontend/index.html
```

Confirm the structure: modal title at line 138, "Text only" / "Process images" buttons at lines 147-148, FormData append at line 273.

- [ ] **Step 2: Add the OCR checkbox to the modal**

Edit `frontend/index.html`. Replace the modal body around lines 138-148 (the existing buttons block) with:

```html
    <div class="modal-title" id="ingest-modal-title">Process images in this PDF?</div>
    <div class="modal-body">
      <p style="margin: 0 0 12px;">
        Image processing extracts figures, captions them with Vision, and makes them retrievable.
      </p>
      <label style="display:flex; align-items:center; gap:8px; margin: 12px 0; cursor:pointer;">
        <input type="checkbox" id="ingest-modal-ocr">
        <span>Also OCR scanned pages (for handwritten or scanned PDFs — uses Vision per page)</span>
      </label>
    </div>
    <div class="modal-actions">
      <button class="modal-btn" id="ingest-modal-skip">Text only (faster)</button>
      <button class="modal-btn primary" id="ingest-modal-confirm">Process images</button>
    </div>
```

(Adjust the wrapping `<div>` only if needed — keep the existing modal class names. The key additions are the `<input type="checkbox" id="ingest-modal-ocr">` line and the surrounding label.)

- [ ] **Step 3: Read the OCR checkbox value and send it as a form field**

In `frontend/index.html`, find the upload code that builds FormData (around line 273). It currently has:

```javascript
    formData.append('process_images', processImages ? 'true' : 'false');
```

Add immediately after:

```javascript
    const ocrScanned = document.getElementById('ingest-modal-ocr').checked;
    formData.append('ocr_scanned', ocrScanned ? 'true' : 'false');
```

The "Text only" path should also send `ocr_scanned=false` (the checkbox would be unchecked, so this works automatically — no extra change needed).

- [ ] **Step 4: Verify in the browser**

```bash
make db && make dev
```

Open http://localhost:8000. Drag a PDF onto the sidebar. Confirm:
- The modal shows the new checkbox under the existing buttons.
- Toggle the checkbox; click "Process images" or "Text only".
- Open browser devtools Network tab; the `/ingest` request's form data should include `ocr_scanned=true` or `false` matching the checkbox.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html
git commit -m "feat(ui): add OCR scanned pages checkbox to upload modal"
```

---

## Task 9: Frontend viewer — bbox overlay for OCR'd pages

**Files:**
- Modify: `frontend/viewer.html` (around lines 320-380, the navigation/highlight phase)

- [ ] **Step 1: Read the current navigation block**

```bash
grep -n "tryHighlight\|page_number\|targetPg\|fullText\|pageData" frontend/viewer.html | head -40
```

Confirm where the per-page highlight logic lives (around lines 333-355 inside the main render block).

- [ ] **Step 2: Add a helper to render bbox overlays**

In `frontend/viewer.html`, near the existing `highlightItems` / `highlightByWords` helpers (around lines 73-122), append a new helper:

```javascript
  // Render translucent yellow rectangles for OCR'd lines whose text appears
  // in the cited chunk. Returns true if at least one box was drawn.
  function highlightOcrLines(lines, chunkText, ctx, vp, scale) {
    if (!lines || !lines.length || !chunkText) return false;
    const norm = s => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
    const chunkN = norm(chunkText);
    if (!chunkN) return false;

    const canvasW = vp.width;
    const canvasH = vp.height;
    let drew = false;
    ctx.fillStyle = 'rgba(255, 235, 0, 0.4)';
    for (const line of lines) {
      const lt = norm(line.text);
      if (lt.length < 3) continue;          // skip noise
      if (!chunkN.includes(lt)) continue;    // line not part of this chunk
      const [ymin, xmin, ymax, xmax] = line.bbox;
      // Coords are 0-1000 normalized; scale to canvas pixels.
      const top = (ymin / 1000) * canvasH;
      const left = (xmin / 1000) * canvasW;
      const height = ((ymax - ymin) / 1000) * canvasH;
      const width = ((xmax - xmin) / 1000) * canvasW;
      ctx.fillRect(left, top, width, height);
      drew = true;
    }
    return drew;
  }

  async function fetchOcrLines(pageNum) {
    try {
      const r = await fetch('/doc/ocr-lines/' + pageNum);
      if (!r.ok) return [];
      const body = await r.json();
      return Array.isArray(body) ? body : [];
    } catch { return []; }
  }
```

- [ ] **Step 3: Wire OCR overlay into the per-page highlight path**

In the navigation block (around lines 333-349), replace:

```javascript
        if (fullText) {
          const prevPd = targetPg > 1 ? pageData[targetPg - 2] : null; // [targetPg-2] = page N-1
          if (prevPd) await tryHighlightHead(prevPd.page, prevPd.vp, prevPd.scale, prevPd.overlay, fullText);
          await tryHighlight(pd.page, pd.vp, pd.scale, pd.overlay, fullText);
          const nextPd = pageData[targetPg]; // targetPg is 1-indexed; [targetPg] = page N+1
          if (nextPd) await tryHighlightTail(nextPd.page, nextPd.vp, nextPd.scale, nextPd.overlay, fullText);
        }
```

with:

```javascript
        if (fullText) {
          const prevPd = targetPg > 1 ? pageData[targetPg - 2] : null;
          const nextPd = pageData[targetPg];

          // For each candidate page, try OCR-bbox overlay first; fall back
          // to text-layer highlighting if the page wasn't OCR'd.
          async function highlightOne(pgData, mode) {
            if (!pgData) return;
            const lines = await fetchOcrLines(pgData.n);
            if (lines.length) {
              const ctx = pgData.overlay.getContext('2d');
              if (highlightOcrLines(lines, fullText, ctx, pgData.vp, pgData.scale)) {
                return; // OCR overlay drew something — done for this page
              }
              // Lines exist but none matched the chunk → leave page un-highlighted
              return;
            }
            // Typed page → existing text-layer highlight path
            if (mode === 'head') {
              await tryHighlightHead(pgData.page, pgData.vp, pgData.scale, pgData.overlay, fullText);
            } else if (mode === 'tail') {
              await tryHighlightTail(pgData.page, pgData.vp, pgData.scale, pgData.overlay, fullText);
            } else {
              await tryHighlight(pgData.page, pgData.vp, pgData.scale, pgData.overlay, fullText);
            }
          }

          await highlightOne(prevPd, 'head');
          await highlightOne(pd, 'main');
          await highlightOne(nextPd, 'tail');
        }
```

- [ ] **Step 4: Manual verification with a typed PDF (regression check)**

```bash
make dev
```

Upload a typed PDF (OCR off), ask a question, click a citation pill. Confirm the existing text-layer highlight still works exactly as before (the `fetchOcrLines` call returns `[]`, so the path falls back to `tryHighlight`).

- [ ] **Step 5: Commit**

```bash
git add frontend/viewer.html
git commit -m "feat(viewer): bbox overlay for OCR'd pages with text-layer fallback"
```

---

## Task 10: Integration test + manual smoke + README update

**Files:**
- Create: `tests/test_integration_ocr.py`
- Modify: `README.md`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration_ocr.py`:

```python
"""Integration test: real Gemini Vision OCR on a synthetic scanned page.

Gated on GEMINI_API_KEY being a real key (not the test placeholder). Skip
in CI by default; run locally with `pytest -m integration`.
"""
import os

import pytest

pytestmark = pytest.mark.integration


def _has_real_key() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return bool(key) and key != "test-placeholder"


@pytest.mark.skipif(not _has_real_key(), reason="real GEMINI_API_KEY required")
async def test_ocr_real_vision_on_synthetic_scan(pool):
    """Build a 2-page PDF where page 1 is rendered text (no text layer) and
    page 2 has embedded text. Run ingest with ocr_scanned=True and verify
    chunks were created for both."""
    import io
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    from backend.ingest import run_ingest

    # Page 1: render "PROJECT KICKOFF NOTES" as an image, embed as a single
    # page-sized image (no text layer).
    img = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    except OSError:
        font = ImageFont.load_default()
    draw.text((100, 200), "PROJECT KICKOFF NOTES", fill="black", font=font)
    draw.text((100, 320), "Owner: Alex Kim", fill="black", font=font)
    draw.text((100, 440), "Deadline: 2026-06-01", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    doc = fitz.open()
    page1 = doc.new_page(width=600, height=800)
    page1.insert_image(page1.rect, stream=buf.getvalue())
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "This is page two. Embedded text. Owner is Alex Kim and deadline is June.",
    )
    pdf_bytes = doc.write()
    doc.close()

    events = [
        e async for e in run_ingest(pdf_bytes, process_images=False, ocr_scanned=True)
    ]
    done = next(e for e in events if e["status"] == "done")
    assert done["chunk_count"] >= 2

    # ocr_lines should exist for page 1 (the rendered/scanned page)
    from backend.db import get_ocr_lines
    page1_lines = await get_ocr_lines(pool, done["doc_id"], 1)
    assert len(page1_lines) > 0
    # At least one line should mention something from the rendered text
    all_text = " ".join(L["text"].lower() for L in page1_lines)
    assert "kickoff" in all_text or "alex" in all_text or "kim" in all_text
```

- [ ] **Step 2: Run the integration test (only if you have a real GEMINI_API_KEY)**

```bash
GEMINI_API_KEY=your_real_key pytest tests/test_integration_ocr.py -v -m integration
```

Expected: PASS. If FAIL on the keyword-matching assertion, inspect `page1_lines` — Vision phrasing varies; adjust the assertion to keywords visible in your run if needed.

- [ ] **Step 3: Manual smoke test**

```bash
make db && make dev
```

Run the five smoke cases from the spec:

1. Typed PDF, OCR off → ingest matches baseline; chunk_count looks normal.
2. Same typed PDF, OCR on → same chunk_count; no `OCRing N scanned page(s)…` status (Vision didn't fire).
3. Scanned/handwritten PDF, OCR off → done event has `chunk_count: 0`; querying returns "I don't know".
4. Same scanned PDF, OCR on → status line shows `OCRing N scanned pages…`; chunks created; querying returns a grounded answer; clicking a citation pill opens the viewer with translucent yellow boxes over the cited lines on the rendered page.
5. Mixed PDF (typed + scanned), OCR on → typed-page citations highlight via text layer (existing yellow rectangles around words); scanned-page citations highlight via the new bbox overlay. Both work in the same viewer session.

Capture any visual issues (mis-aligned overlays, low Vision OCR accuracy) and note them as follow-ups in the spec's "Open questions" section if they need tuning.

- [ ] **Step 4: Update README**

In `README.md`, in the "Stack" table (around lines 7-20), add a row for OCR. Replace the row that begins `| PDF extraction |` with:

```markdown
| PDF extraction | PyMuPDF (`fitz`) — text + embedded images |
| OCR (opt-in) | Gemini 2.5 Flash with structured JSON — line-level transcription + bboxes for scanned/handwritten pages |
```

In the "Usage" section (around lines 71-80), update step 2:

```markdown
2. A modal asks whether to **Process images** (extract figures, caption with Gemini Vision, embed) or do a **Text only** ingest (faster — skips Vision entirely). A separate checkbox enables **OCR scanned pages** for handwritten or scanned PDFs — empty-text pages are rendered and transcribed via Gemini Vision; cited lines are highlighted in yellow in the viewer.
```

In the "Design decisions" section, append a new entry near the existing image-related decisions:

```markdown
**OCR via Gemini Vision (opt-in)**
Scanned and handwritten PDFs have no embedded text — PyMuPDF returns empty pages and the index ends up empty. When the user enables "OCR scanned pages" at upload, every page where PyMuPDF returned ≤ 50 chars is rendered at 200 DPI and sent to Gemini 2.5 Flash with `response_mime_type="application/json"` to get back structured `[{text, box}, ...]` data — text feeds the chunker normally, bboxes get stored in a separate `ocr_lines` table. Tesseract was rejected because it's poor at handwriting (the primary case here) and would still need Vision as an escalation, doubling complexity. The viewer transparently switches between text-layer highlighting (typed pages) and bbox-overlay highlighting (OCR'd pages) per page, so a mixed PDF works without a second code path.
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_ocr.py README.md
git commit -m "test+docs: integration test and README updates for OCR feature"
```

---

## Self-review

**Spec coverage check:**

| Spec section | Plan task |
|---|---|
| Upload modal toggle | Task 8 |
| Per-page detection (≤50 chars) | Task 4 |
| Mixed PDFs work transparently | Task 4 + Task 6 + Task 9 |
| Citations + bbox overlay | Task 9 |
| Why Vision over Tesseract/hybrid | README update in Task 10 |
| Architecture (pre-step) | Task 6 |
| `_render_page` | Task 2 |
| `_ocr_one_page` (returns text + lines) | Task 3 |
| `_ocr_empty_pages` | Task 4 |
| `_extract_images(skip_pages=...)` | Task 5 |
| `run_ingest(ocr_scanned=...)` + SSE event | Task 6 |
| `ocr_lines` table | Task 1 |
| `insert_ocr_lines` / `get_ocr_lines` | Task 1 |
| `clear_all_chunks` extension | Task 1 |
| `ocr_scanned` form field | Task 7 |
| `/doc/ocr-lines/{page_num}` endpoint | Task 7 |
| Viewer overlay | Task 9 |
| Per-page failure isolation | Task 3 + Task 4 |
| Retry on 429/503 | Task 3 |
| Empty/`unreadable`/malformed JSON handling | Task 3 |
| All unit tests from spec | Tasks 1, 3, 4, 5, 6, 7 |
| Integration test | Task 10 |
| Manual smoke test (5 cases) | Task 10 |

All spec requirements have a task. No placeholders in the plan. Type/signature consistency: `_ocr_one_page` returns `tuple[str, list[dict]]` everywhere it's referenced; `_ocr_empty_pages` returns `tuple[list[str], set[int], dict[int, list[dict]]]` consistently; DB helpers use `bbox: list[int]` consistently in storage and `box: list[int]` in the in-memory shape returned by `_ocr_one_page` (the conversion happens in `insert_ocr_lines`).
