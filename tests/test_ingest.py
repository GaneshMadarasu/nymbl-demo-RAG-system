import json as _json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from backend.ingest import _doc_id, run_ingest


def test_doc_id_is_deterministic():
    b = b"hello pdf content"
    assert _doc_id(b) == _doc_id(b)


def test_doc_id_differs_for_different_content():
    assert _doc_id(b"pdf one") != _doc_id(b"pdf two")


def test_doc_id_is_16_chars():
    assert len(_doc_id(b"anything")) == 16


async def test_run_ingest_yields_done_event(pool):
    fake_text = "This is a sentence about artificial intelligence. " * 20
    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", return_value=(fake_text, [])),
        patch("backend.ingest._extract_images", return_value=[]),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
    ):
        events = [e async for e in run_ingest(b"fake-pdf-bytes")]
    statuses = [e["status"] for e in events]
    assert "done" in statuses


async def test_run_ingest_done_event_has_chunk_count(pool):
    fake_text = "Sentence about topic. " * 30
    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", return_value=(fake_text, [])),
        patch("backend.ingest._extract_images", return_value=[]),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
    ):
        events = [e async for e in run_ingest(b"fake-pdf-bytes")]
    done = next(e for e in events if e["status"] == "done")
    assert done["chunk_count"] > 0
    assert "doc_id" in done
    assert "k" in done


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


async def test_ocr_one_page_parses_valid_json():
    fake_response = SimpleNamespace(
        text=_json.dumps(
            [
                {"text": "Hello world", "box": [10, 20, 30, 800]},
                {"text": "Second line", "box": [40, 20, 60, 800]},
            ]
        )
    )
    with (
        patch(
            "backend.ingest._render_page",
            return_value=(b"\x89PNG\r\n\x1a\n...", "image/png"),
        ),
        patch(
            "backend.ingest._downsize_for_vision",
            return_value=(b"jpegbytes", "image/jpeg"),
        ),
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
        patch(
            "backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")
        ),
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
        patch(
            "backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")
        ),
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
    fake_response = SimpleNamespace(
        text=_json.dumps(
            [
                {"text": "Has box", "box": [0, 0, 10, 100]},
                {"text": "Missing box"},
                {"text": "Has box too", "box": [20, 0, 30, 100]},
            ]
        )
    )
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG", "image/png")),
        patch(
            "backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")
        ),
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


async def test_ocr_empty_pages_only_ocrs_below_threshold():
    pages = [
        "Substantial first page text " * 20,  # > 50 chars → not OCR'd
        "",  # empty → OCR'd
        "tiny",  # < 50 → OCR'd
        "Plenty of words on this fourth page that is well above threshold",
    ]
    captured_pages: list[int] = []

    async def fake_ocr(_pdf, page_num):
        captured_pages.append(page_num)
        return f"OCR text page {page_num}", [
            {"text": f"OCR text page {page_num}", "box": [0, 0, 10, 100]}
        ]

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
        return f"page {page_num} text", [
            {"text": f"page {page_num} text", "box": [0, 0, 5, 50]}
        ]

    with patch("backend.ingest._ocr_one_page", side_effect=fake_ocr):
        from backend.ingest import _ocr_empty_pages

        updated, ocr_set, lines_by_page = await _ocr_empty_pages(b"pdf", pages)

    assert updated[0] == "page 1 text"
    assert updated[1] == ""  # failed page kept empty
    assert updated[2] == "page 3 text"
    assert ocr_set == {1, 3}  # only successfully OCR'd pages in the set
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
