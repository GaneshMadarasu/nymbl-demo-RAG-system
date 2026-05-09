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


def test_extract_text_captures_highlight_underline_and_freetext_annotations():
    """Manual PDF markup (highlights, underlines, sticky notes, free-text
    comments) is appended to the page's body text so the chunker / embedder
    pick it up. Ink annotations are intentionally skipped (Path 3 future)."""
    import fitz
    from backend.ingest import _extract_text

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Project deadline is Friday this week.")
    page.insert_text((72, 130), "Owner is Alex Kim.")
    # Highlight overlaps the deadline text
    page.add_highlight_annot(fitz.Rect(72, 92, 290, 110))
    # Underline overlaps the owner text
    page.add_underline_annot(fitz.Rect(72, 122, 200, 140))
    # Sticky note (no underlying text — content is in the annotation itself)
    page.add_text_annot(fitz.Point(400, 100), "review by EOQ")
    # Free-text annotation
    page.add_freetext_annot(fitz.Rect(72, 200, 300, 230), "ship blocker")
    pdf_bytes = doc.write()
    doc.close()

    text, pages = _extract_text(pdf_bytes)
    page_text = pages[0]

    # Body text is preserved
    assert "Project deadline is Friday" in page_text
    assert "Owner is Alex Kim" in page_text
    # Annotation summary appended
    assert "[Annotations on this page:" in page_text
    assert "[highlight:" in page_text
    assert "[underlined:" in page_text
    assert "[note: review by EOQ]" in page_text
    assert "[note: ship blocker]" in page_text


async def test_detect_visual_markup_one_page_parses_valid_json():
    import json as _json

    fake = SimpleNamespace(
        text=_json.dumps(
            {
                "markup": [
                    {"type": "underline", "color": "red", "text": "deadline"},
                    {"type": "note", "color": "blue", "text": "review by Alex"},
                ]
            }
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
            return_value=fake,
        ),
    ):
        from backend.ingest import _detect_visual_markup_one_page

        items = await _detect_visual_markup_one_page(b"fake-pdf", 1)
    assert len(items) == 2
    assert items[0]["type"] == "underline"
    assert items[0]["color"] == "red"
    assert items[0]["text"] == "deadline"


async def test_detect_visual_markup_one_page_handles_empty_array():
    fake = SimpleNamespace(text='{"markup": []}')
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG", "image/png")),
        patch(
            "backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")
        ),
        patch(
            "backend.ingest._client.aio.models.generate_content",
            new_callable=AsyncMock,
            return_value=fake,
        ),
    ):
        from backend.ingest import _detect_visual_markup_one_page

        items = await _detect_visual_markup_one_page(b"fake-pdf", 1)
    assert items == []


async def test_detect_visual_markup_one_page_handles_malformed_json():
    fake = SimpleNamespace(text="not json {{{")
    with (
        patch("backend.ingest._render_page", return_value=(b"\x89PNG", "image/png")),
        patch(
            "backend.ingest._downsize_for_vision", return_value=(b"jpeg", "image/jpeg")
        ),
        patch(
            "backend.ingest._client.aio.models.generate_content",
            new_callable=AsyncMock,
            return_value=fake,
        ),
    ):
        from backend.ingest import _detect_visual_markup_one_page

        items = await _detect_visual_markup_one_page(b"fake-pdf", 1)
    assert items == []


def test_format_markup_summary_includes_color_when_non_default():
    from backend.ingest import _format_markup_summary

    items = [
        {"type": "underline", "color": "red", "text": "deadline"},
        {"type": "note", "color": "black", "text": "see appendix"},
    ]
    out = _format_markup_summary(items)
    assert "[underline in red: deadline]" in out
    assert "[note: see appendix]" in out  # black drops the color suffix
    assert "Visual markup on this page" in out


def test_format_markup_summary_empty_for_no_items():
    from backend.ingest import _format_markup_summary

    assert _format_markup_summary([]) == ""


async def test_collect_visual_markup_runs_on_pages_with_text():
    """Pages with non-empty body text get the markup pass; empty pages don't."""
    pages = ["page one body text", "", "page three body text"]
    captured: list[int] = []

    async def fake_detect(_pdf, page_num):
        captured.append(page_num)
        return (
            [{"type": "underline", "color": "red", "text": f"mark p{page_num}"}]
            if page_num == 1
            else []
        )

    with patch(
        "backend.ingest._detect_visual_markup_one_page", side_effect=fake_detect
    ):
        from backend.ingest import _collect_visual_markup

        out = await _collect_visual_markup(b"fake-pdf", pages)

    assert sorted(captured) == [1, 3]
    assert 1 in out
    assert 3 not in out  # page 3 returned empty list


def test_extract_text_no_annotations_unchanged():
    """Regression guard: PDFs without annotations produce the same body text
    as before (no spurious annotation tag appended)."""
    import fitz
    from backend.ingest import _extract_text

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Plain typed body text with no markup.")
    pdf_bytes = doc.write()
    doc.close()

    text, pages = _extract_text(pdf_bytes)
    assert "Plain typed body text" in pages[0]
    assert "Annotations on this page" not in pages[0]


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


async def test_ocr_one_page_returns_transcribed_text():
    fake_response = SimpleNamespace(text="Hello world\nSecond line")
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

        text = await _ocr_one_page(b"fake-pdf", 1)
    assert text == "Hello world\nSecond line"


async def test_ocr_one_page_handles_unreadable_response():
    fake_response = SimpleNamespace(text="unreadable")
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

        text = await _ocr_one_page(b"fake-pdf", 1)
    assert text == ""


async def test_ocr_one_page_render_failure_returns_empty():
    with patch("backend.ingest._render_page", side_effect=RuntimeError("render bomb")):
        from backend.ingest import _ocr_one_page

        text = await _ocr_one_page(b"fake-pdf", 1)
    assert text == ""


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
        return f"OCR text page {page_num}"

    with patch("backend.ingest._ocr_one_page", side_effect=fake_ocr):
        from backend.ingest import _ocr_empty_pages

        updated, ocr_set = await _ocr_empty_pages(b"pdf", pages)

    assert sorted(captured_pages) == [2, 3]
    assert ocr_set == {2, 3}
    assert updated[0] == pages[0]
    assert updated[1] == "OCR text page 2"
    assert updated[2] == "OCR text page 3"
    assert updated[3] == pages[3]


async def test_ocr_empty_pages_failure_isolated():
    pages = ["", "", ""]

    async def fake_ocr(_pdf, page_num):
        if page_num == 2:
            return ""  # simulate failure -> empty
        return f"page {page_num} text"

    with patch("backend.ingest._ocr_one_page", side_effect=fake_ocr):
        from backend.ingest import _ocr_empty_pages

        updated, ocr_set = await _ocr_empty_pages(b"pdf", pages)

    assert updated[0] == "page 1 text"
    assert updated[1] == ""  # failed page kept empty
    assert updated[2] == "page 3 text"
    assert ocr_set == {1, 3}  # only successfully OCR'd pages in the set


async def test_ocr_empty_pages_no_empty_pages_no_calls():
    pages = ["Lots of text " * 10, "More text " * 20]
    fake_ocr = AsyncMock()
    with patch("backend.ingest._ocr_one_page", fake_ocr):
        from backend.ingest import _ocr_empty_pages

        updated, ocr_set = await _ocr_empty_pages(b"pdf", pages)

    fake_ocr.assert_not_awaited()
    assert updated == pages
    assert ocr_set == set()


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
        patch(
            "backend.ingest._detect_visual_markup_one_page",
            new_callable=AsyncMock,
            return_value=[],
        ) as m_markup,
    ):
        events = [e async for e in run_ingest(b"pdf-bytes", ocr_scanned=True)]
    m_ocr_one.assert_not_awaited()
    # Markup detection is gated on detect_markup, NOT ocr_scanned
    m_markup.assert_not_awaited()
    assert any(e["status"] == "done" for e in events)


async def test_run_ingest_detect_markup_runs_vision_pass(pool):
    """detect_markup=True should fire the visual-markup pass independently
    of ocr_scanned."""
    fake_pages = ["page one substantive text " * 10, "page two substantive text " * 10]
    fake_text = "\n\n".join(fake_pages)

    async def fake_markup(_pdf, page_num):
        return [{"type": "underline", "color": "red", "text": f"mark p{page_num}"}]

    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", return_value=(fake_text, fake_pages)),
        patch("backend.ingest._extract_images", return_value=[]),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch(
            "backend.ingest._detect_visual_markup_one_page", side_effect=fake_markup
        ) as m_markup,
    ):
        events = [e async for e in run_ingest(b"pdf-bytes", detect_markup=True)]
    statuses = [e["status"] for e in events]
    assert "markup" in statuses
    assert m_markup.await_count == 2  # one call per text page


async def test_run_ingest_detect_markup_off_skips_vision(pool):
    """Default detect_markup=False keeps the markup pass dormant."""
    fake_pages = ["page one substantive text " * 10]
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
        patch(
            "backend.ingest._detect_visual_markup_one_page",
            new_callable=AsyncMock,
            return_value=[],
        ) as m_markup,
    ):
        events = [e async for e in run_ingest(b"pdf-bytes")]
    m_markup.assert_not_awaited()
    assert any(e["status"] == "done" for e in events)


async def test_run_ingest_ocr_on_with_empty_pages_calls_vision(pool):
    fake_pages = ["", "Substantial typed text on this page " * 5, ""]
    fake_text = "\n\n".join(p for p in fake_pages if p.strip())

    async def fake_ocr(_pdf, page_num):
        return f"OCR'd text page {page_num} " * 4

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
        patch(
            "backend.ingest._detect_visual_markup_one_page",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        events = [e async for e in run_ingest(b"pdf-bytes", ocr_scanned=True)]

    statuses = [e["status"] for e in events]
    assert "ocr" in statuses
    done = next(e for e in events if e["status"] == "done")
    assert done["chunk_count"] > 0

    # _extract_images received skip_pages={1, 3} (the OCR'd pages)
    _, kwargs = m_imgs.call_args
    assert kwargs.get("skip_pages") == {1, 3}
