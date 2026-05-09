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
