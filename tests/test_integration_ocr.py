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
