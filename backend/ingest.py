import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from io import BytesIO

import fitz  # pymupdf
import tiktoken
from google import genai
from google.genai import types
from PIL import Image

from backend import db
from backend.chunks import chunk_text
from backend.config import settings

_tokenizer = tiktoken.get_encoding("cl100k_base")

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

_EMBED_MODEL = "gemini-embedding-2"
_VISION_MODEL = "gemini-2.5-flash"
_EMBED_DIM = 768
_MAX_RETRIES = 6
_RETRY_BASE = 2.0  # delays: 2, 4, 8, 16, 32, 64s

# Filter heuristics for image extraction
_MIN_IMAGE_PX = 200  # skip images smaller than this in either dimension
_MIN_IMAGE_BYTES = 5_000  # skip very small files (icons, dividers, decorative)
_SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_FULLPAGE_DPI = 150  # DPI for stage-2 full-page fallback render
# Stage 2 fires only on pages with no embedded images AND text length under this
# limit — heuristic for "this page is mostly a painting/scan with little text".
_PAGE_TEXT_FALLBACK_LIMIT = 200
# Vision doesn't need full resolution to caption an image. Resize to this max
# dimension and re-encode as JPEG before sending — cuts the upload payload
# by 10-20× for high-DPI page renders.
_CAPTION_MAX_DIM = 1024
_CAPTION_JPEG_QUALITY = 85
# Cap concurrent Gemini calls (caption + embed) to avoid 429s and the
# exponential-backoff retries that follow them. 8 is well below the per-minute
# limits of gemini-embedding-2 / gemini-2.5-flash and reduces overall wall time
# vs unlimited gather() once you have ~30+ images.
_GEMINI_CONCURRENCY = 8

_OCR_PAGE_DPI = 200
_OCR_EMPTY_PAGE_THRESHOLD = 50


_BLANK_PHRASES = (
    "no meaningful visual",
    "no meaningful content",
    "no visual content",
    "no actual artwork",
    "no artwork",
    "no diagram",
    "no chart",
    "no schematic",
    "no figure",
    "no image content",
    "no painting",
    "only text",
    "just text",
    "text only",
    "text-only",
    "contains only text",
    "is blank",
    "appears blank",
    "is empty",
    "appears empty",
    "blank page",
    "blank image",
    "blank background",
    "page contains text",
    "this is a page of text",
)


def _is_blank_caption(caption: str | None) -> bool:
    """Drop captions that indicate the 'image' is really a text page, blank
    background, or other non-artwork content. Vision is instructed to reply
    'blank' for these but doesn't always comply, so we also catch common
    verbose negations like 'contains only text'."""
    if not caption:
        return True
    s = caption.strip().lower()
    if not s or len(s) < 5:
        return True
    s_clean = s.strip(".!?\"' ")
    if (
        s_clean == "blank"
        or s_clean.startswith("blank ")
        or s_clean.startswith("blank,")
    ):
        return True
    return any(phrase in s for phrase in _BLANK_PHRASES)


_CAPTION_PROMPT = (
    "CRITICAL FIRST CHECK: If this image is any of the following, respond with EXACTLY "
    "the single word 'blank' as your entire response (no explanation, no other text):\n"
    "- A page of text from a book, document, or PDF (even if you can read the text)\n"
    "- A blank, white, solid-colored, or near-uniform background\n"
    "- A page number, header, footer, decorative border, or watermark\n"
    "- Any image without an actual artwork, photograph, diagram, chart, or figure\n\n"
    "Otherwise, describe the visual content in 1-2 concise sentences. "
    "For an artwork (painting, illustration, photograph), focus on the subject depicted, "
    "the style or medium, and notable visual details such as colors, composition, or mood — "
    "ignore page numbers, headers, footers, and surrounding caption text. "
    "For a diagram, chart, or schematic, focus on what it conveys and any axis labels or "
    "key annotations. Be specific and factual."
)


_OCR_PROMPT = (
    "Transcribe ALL visible text in this image verbatim (printed and handwritten). "
    "Preserve line breaks and paragraph structure. "
    "Do not add commentary, summaries, or explanations — output only the transcribed text. "
    "If the image contains no readable text, respond with exactly the word 'unreadable'."
)

_VISUAL_MARKUP_PROMPT = (
    "Look at this page and identify HAND-DRAWN markup added on top of the printed text. "
    "Hand-drawn markup includes: pen/pencil underlines (visible ink strokes, often imperfect), "
    "marker highlights (translucent yellow/pink/orange overlay on text), circles or boxes "
    "drawn around words, arrows pointing at text, stars or check marks added by hand, "
    "margin notes written in by hand, and hand-drawn strikethroughs. "
    "Do NOT report typographic emphasis (bold, italic, typeset underlines), bullets, page "
    "borders, or anything that's part of the original typesetting. "
    "For each item, output JSON: "
    '{"markup": [{"type": "underline" | "highlight" | "circle" | "arrow" | "star" | "note" | "strikethrough" | "box", '
    '"color": "red" | "blue" | "green" | "yellow" | "orange" | "black" | "purple" | "other", '
    '"text": "verbatim text being marked or handwritten"}, ...]} '
    'If the page has no hand-drawn markup, output {"markup": []}. '
    "Do not include commentary or markdown fences."
)


def _ext_to_mime(ext: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext.lower(), f"image/{ext.lower()}")


def _doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def compute_params(total_tokens: int) -> tuple[int, int]:
    """Return (chunk_size, k) scaled to document size."""
    if total_tokens < 10_000:  # < ~20 pages
        return 256, 5
    elif total_tokens < 50_000:  # 20–100 pages
        return 384, 8
    elif total_tokens < 200_000:  # 100–400 pages
        return 512, 12
    elif total_tokens < 500_000:  # 400–1000 pages
        return 768, 15
    else:  # 1000+ pages
        return 1024, 20


_ANNOT_HIGHLIGHT_LIKE = {"Highlight", "Underline", "Squiggly", "StrikeOut"}
_ANNOT_LABEL = {
    "Highlight": "highlight",
    "Underline": "underlined",
    "Squiggly": "squiggly",
    "StrikeOut": "strikethrough",
}


def _format_annotations(page) -> str:
    """Read PDF annotation objects on the page (Highlight, Underline,
    StrikeOut, FreeText, Text/sticky) and return a string summary suitable
    for appending to the page's body text. Returns "" when there are none.

    Each annotation enriches the chunk that contains it: a query like "what
    did I underline?" can match the bracketed `[underlined: ...]` tag in
    the chunk's parent_text, and the LLM has the underlying text in context."""
    annots = page.annots()
    if not annots:
        return ""
    parts: list[str] = []
    for annot in annots:
        atype = annot.type[1] if annot.type else ""
        if atype in _ANNOT_HIGHLIGHT_LIKE:
            try:
                underlying = page.get_text("text", clip=annot.rect).strip()
            except Exception:
                underlying = ""
            if underlying:
                parts.append(f"[{_ANNOT_LABEL[atype]}: {underlying}]")
        elif atype in ("FreeText", "Text"):
            content = (annot.info.get("content") or "").strip()
            if content:
                parts.append(f"[note: {content}]")
        # Ink (handwritten scribbles) carries no text content; recognising it
        # would need Vision OCR on the rasterised stroke layer — see README
        # "Future: scanned & flattened-ink annotations".
    if not parts:
        return ""
    return "\n\n[Annotations on this page: " + " ".join(parts) + "]"


def _extract_text(pdf_bytes: bytes) -> tuple[str, list[str]]:
    """Returns (full_text, per_page_texts) where per_page_texts is 1-indexed (index 0 unused).
    Each page's text includes a trailing summary of any PDF annotations
    (highlights, underlines, sticky notes, free-text comments) so manual
    markup becomes searchable through the existing chunker / embedder."""
    t0 = time.monotonic()
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    pages: list[str] = []
    annot_count = 0
    for page in doc:
        body = page.get_text().replace("\x00", "")
        annot_summary = _format_annotations(page)
        if annot_summary:
            body = (body + annot_summary).strip()
            annot_count += 1
        pages.append(body)
    doc.close()
    text = "\n\n".join(p for p in pages if p.strip())
    logger.info(
        "PyMuPDF extracted %d chars from %d pages in %.2fs (%d page(s) with annotations)",
        len(text),
        len(pages),
        time.monotonic() - t0,
        annot_count,
    )
    return text, pages  # pages[0] = page 1 text, pages[1] = page 2 text, ...


def _find_chunk_page(chunk: str, pages: list[str]) -> int:
    """Return the 1-indexed page number where this chunk starts.
    Normalises whitespace so chunker's sentence-joining matches page text."""
    norm = lambda s: " ".join(s.split())
    for anchor_len in (80, 50, 30):
        anchor = norm(chunk[:anchor_len])
        if not anchor:
            continue
        for page_num, pt in enumerate(pages, 1):
            if anchor in norm(pt):
                return page_num
    return 1  # fallback


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "429" in msg
        or "503" in msg
        or "ResourceExhausted" in type(exc).__name__
        or "ServiceUnavailable" in type(exc).__name__
        or "timed out" in msg.lower()
        or "UNAVAILABLE" in msg
    )


async def _embed_one(text: str) -> list[float]:
    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.embed_content(
                model=_EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=_EMBED_DIM,
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            return list(r.embeddings[0].values)
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning("Embedding failed (%s); retrying in %.0fs", exc, wait)
                await asyncio.sleep(wait)
                continue
            raise


def _render_page(pdf_bytes: bytes, page_num: int, dpi: int) -> tuple[bytes, str]:
    """Render one PDF page to PNG bytes. page_num is 1-indexed."""
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    try:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png"), "image/png"
    finally:
        doc.close()


def _extract_images(
    pdf_bytes: bytes,
    skip_pages: set[int] | None = None,
) -> list[tuple[int, bytes, str]]:
    """Extract paintings/figures from a PDF in two stages.

    Stage 1 — embedded images: walk every page's image XRefs and use the raw
    embedded bytes. Re-rendering the page bbox would also capture any text
    overlay on top of a background image, so we trust the embedded bytes and
    let Vision filter out blank/decorative ones via caption.

    Stage 2 — page-render fallback: for pages with no embedded images AND very
    little text, render the whole page. Catches scanned/full-bleed painting
    pages where PyMuPDF reports no embedded images at all.

    Returns list of (page_num, image_bytes, mime_type)."""
    skip_pages = skip_pages or set()
    t0 = time.monotonic()
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    images: list[tuple[int, bytes, str]] = []
    seen_xrefs: set[int] = set()
    pages_with_extracted: set[int] = set()
    stage1_count = 0
    stage2_count = 0

    # ---------- Stage 1: embedded images, raw bytes ----------
    for page_num, page in enumerate(doc, 1):
        for img_tuple in page.get_images(full=True):
            xref = img_tuple[0]
            if xref in seen_xrefs:
                pages_with_extracted.add(page_num)
                continue
            seen_xrefs.add(xref)

            try:
                extracted = doc.extract_image(xref)
            except Exception as exc:
                logger.warning(
                    "Failed to extract image xref=%d on page %d: %s",
                    xref,
                    page_num,
                    exc,
                )
                continue

            img_bytes = extracted.get("image")
            ext = (extracted.get("ext") or "").lower()
            width = extracted.get("width", 0)
            height = extracted.get("height", 0)
            if not img_bytes:
                continue
            if width < _MIN_IMAGE_PX or height < _MIN_IMAGE_PX:
                continue
            if len(img_bytes) < _MIN_IMAGE_BYTES:
                continue
            mime = _ext_to_mime(ext)
            if mime not in _SUPPORTED_MIMES:
                continue

            # Use raw embedded bytes — re-rendering the bbox captures any text
            # overlay on top of the image (page text, page number, captions),
            # which produced text-page screenshots when the embedded image
            # was actually a blank background. Vision will caption blank
            # backgrounds as "blank" and we filter those after captioning.
            images.append((page_num, img_bytes, mime))
            pages_with_extracted.add(page_num)
            stage1_count += 1

    # ---------- Stage 2: full-page render for sparse-text pages with no embedded images ----------
    for page_num, page in enumerate(doc, 1):
        if page_num in pages_with_extracted:
            continue
        if page_num in skip_pages:
            continue  # caller (e.g. OCR pipeline) already handled this page
        text_len = len(page.get_text().strip())
        if text_len > _PAGE_TEXT_FALLBACK_LIMIT:
            continue  # text-heavy page with no figures — skip
        try:
            pix = page.get_pixmap(dpi=_FULLPAGE_DPI)
            img_bytes = pix.tobytes("png")
        except Exception as exc:
            logger.warning("Full-page render failed for page %d: %s", page_num, exc)
            continue
        images.append((page_num, img_bytes, "image/png"))
        stage2_count += 1

    doc.close()
    logger.info(
        "Image extraction: stage1=%d (embedded), stage2=%d (page-render), total=%d in %.2fs",
        stage1_count,
        stage2_count,
        len(images),
        time.monotonic() - t0,
    )
    return images


def _downsize_for_vision(image_bytes: bytes) -> tuple[bytes, str]:
    """Resize to <=_CAPTION_MAX_DIM on longest side and re-encode as JPEG.
    Returns (bytes, mime). Falls back to the original on any error so
    captioning still proceeds — the downsize is a perf optimization, not
    a correctness requirement."""
    try:
        im = Image.open(BytesIO(image_bytes))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > _CAPTION_MAX_DIM:
            scale = _CAPTION_MAX_DIM / longest
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.LANCZOS,
            )
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=_CAPTION_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as exc:
        logger.warning("Image downsize failed (%s); using original bytes", exc)
        return image_bytes, ""


async def _caption_image(image_bytes: bytes, mime_type: str) -> str:
    """Caption an image using Gemini Vision. Retries on retryable errors."""
    # Downsize before sending — Vision doesn't need 4K resolution to caption
    # a painting, and high-DPI page renders are the bulk of the upload time.
    send_bytes, send_mime = _downsize_for_vision(image_bytes)
    if not send_mime:
        send_mime = mime_type
    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.generate_content(
                model=_VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=send_bytes, mime_type=send_mime),
                    _CAPTION_PROMPT,
                ],
            )
            return (r.text or "image").strip()
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning("Captioning failed (%s); retrying in %.0fs", exc, wait)
                await asyncio.sleep(wait)
                continue
            logger.warning("Captioning gave up after %d attempts: %s", attempt + 1, exc)
            return "image"


async def _ocr_one_page(pdf_bytes: bytes, page_num: int) -> str:
    """Render a PDF page and OCR it via Gemini Vision.

    Returns the transcribed text. On any failure (render error, "unreadable"
    response, all-retries-exhausted) returns "" so one bad page never aborts
    the ingest."""
    try:
        page_bytes, page_mime = _render_page(pdf_bytes, page_num, _OCR_PAGE_DPI)
    except Exception as exc:
        logger.warning("OCR render failed for page %d: %s", page_num, exc)
        return ""

    send_bytes, send_mime = _downsize_for_vision(page_bytes)
    if not send_mime:
        send_mime = page_mime

    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.generate_content(
                model=_VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=send_bytes, mime_type=send_mime),
                    _OCR_PROMPT,
                ],
            )
            raw_text = (r.text or "").strip()
            if not raw_text or raw_text.lower().strip(".!? \"'") == "unreadable":
                return ""
            return raw_text
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning(
                    "OCR call failed for page %d (%s); retrying in %.0fs",
                    page_num,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.warning(
                "OCR gave up for page %d after %d attempts: %s",
                page_num,
                attempt + 1,
                exc,
            )
            return ""
    return ""


async def _detect_visual_markup_one_page(pdf_bytes: bytes, page_num: int) -> list[dict]:
    """Render a page and ask Vision to identify hand-drawn markup (underlines,
    highlights, circles, margin notes) and the text being marked. Returns a
    list of {type, color, text} dicts; empty list on any failure (one bad
    page never aborts the ingest)."""
    import json as _json

    try:
        page_bytes, page_mime = _render_page(pdf_bytes, page_num, _OCR_PAGE_DPI)
    except Exception as exc:
        logger.warning("Markup render failed for page %d: %s", page_num, exc)
        return []

    send_bytes, send_mime = _downsize_for_vision(page_bytes)
    if not send_mime:
        send_mime = page_mime

    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.generate_content(
                model=_VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=send_bytes, mime_type=send_mime),
                    _VISUAL_MARKUP_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            raw = (r.text or "").strip()
            if not raw:
                return []
            try:
                parsed = _json.loads(raw)
            except Exception as exc:
                logger.warning(
                    "Markup JSON parse failed for page %d: %s", page_num, exc
                )
                return []
            if not isinstance(parsed, dict):
                return []
            items = parsed.get("markup")
            if not isinstance(items, list):
                return []
            out: list[dict] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                txt = it.get("text")
                ty = it.get("type")
                col = it.get("color")
                if (
                    isinstance(txt, str)
                    and txt.strip()
                    and isinstance(ty, str)
                    and ty.strip()
                ):
                    out.append(
                        {
                            "type": ty.strip(),
                            "color": (col or "").strip()
                            if isinstance(col, str)
                            else "",
                            "text": txt.strip(),
                        }
                    )
            return out
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning(
                    "Markup call failed for page %d (%s); retrying in %.0fs",
                    page_num,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            logger.warning(
                "Markup gave up for page %d after %d attempts: %s",
                page_num,
                attempt + 1,
                exc,
            )
            return []
    return []


def _format_markup_summary(items: list[dict]) -> str:
    """Format detected markup items into a bracketed string suitable for
    appending to the page text. Mirrors the layered-annotation summary so the
    LLM sees the same shape regardless of whether the source was a PDF
    annotation object or a Vision-detected hand-drawn mark."""
    if not items:
        return ""
    parts: list[str] = []
    for it in items:
        ty = (it.get("type") or "marked").strip().lower()
        col = (it.get("color") or "").strip().lower()
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        if col and col not in ("other", "black"):
            parts.append(f"[{ty} in {col}: {txt}]")
        else:
            parts.append(f"[{ty}: {txt}]")
    if not parts:
        return ""
    return "\n\n[Visual markup on this page: " + " ".join(parts) + "]"


async def _collect_visual_markup(
    pdf_bytes: bytes, pages: list[str]
) -> dict[int, list[dict]]:
    """Run hand-drawn markup detection on every page that has body text. The
    point of this pass is to catch markup BAKED INTO the page raster (Path 3),
    so it's complementary to `_format_annotations`'s layered-annotation reader.
    Pages with no body text are skipped — they go through OCR if requested."""
    targets = [i + 1 for i, p in enumerate(pages) if p.strip()]
    if not targets:
        return {}
    logger.info("Detecting visual markup on %d page(s)", len(targets))
    results = await _gather_bounded(
        [_detect_visual_markup_one_page(pdf_bytes, n) for n in targets],
        _GEMINI_CONCURRENCY,
    )
    out: dict[int, list[dict]] = {}
    for page_num, items in zip(targets, results):
        if items:
            out[page_num] = items
    logger.info(
        "Visual markup detected on %d/%d pages",
        len(out),
        len(targets),
    )
    return out


async def _ocr_empty_pages(
    pdf_bytes: bytes, pages: list[str]
) -> tuple[list[str], set[int]]:
    """For each page in `pages` whose stripped text length is below the
    threshold, OCR it via Gemini Vision. Returns (updated_pages,
    ocr_page_nums).

    - updated_pages: copy of `pages` with successfully-OCR'd entries replaced
      by the transcribed text; failed pages remain as their original empty
      text.
    - ocr_page_nums: 1-indexed pages where OCR returned non-empty text.
    """
    targets = [
        i + 1
        for i, p in enumerate(pages)
        if len(p.strip()) <= _OCR_EMPTY_PAGE_THRESHOLD
    ]
    if not targets:
        return list(pages), set()

    logger.info("Running OCR on %d empty/sparse page(s)", len(targets))
    results = await _gather_bounded(
        [_ocr_one_page(pdf_bytes, n) for n in targets],
        _GEMINI_CONCURRENCY,
    )

    updated = list(pages)
    ocr_set: set[int] = set()

    for page_num, text in zip(targets, results):
        if not text:
            continue
        updated[page_num - 1] = text
        ocr_set.add(page_num)

    logger.info(
        "OCR completed: %d/%d pages produced text",
        len(ocr_set),
        len(targets),
    )
    return updated, ocr_set


def _parent_texts(chunks: list[str]) -> list[str]:
    """Return adjacent-window text for each chunk (prev + self + next)."""
    result = []
    for i, chunk in enumerate(chunks):
        parts = []
        if i > 0:
            parts.append(chunks[i - 1])
        parts.append(chunk)
        if i < len(chunks) - 1:
            parts.append(chunks[i + 1])
        result.append(" ".join(parts))
    return result


async def _gather_bounded(coros, limit: int):
    """Run `coros` with at most `limit` in flight. Cuts 429s vs unlimited
    gather() and is reliably faster overall once the count exceeds ~limit*3."""
    sem = asyncio.Semaphore(limit)

    async def _run(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_run(c) for c in coros])


async def run_ingest(
    pdf_bytes: bytes,
    process_images: bool = True,
    ocr_scanned: bool = False,
) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()
    doc_id = _doc_id(pdf_bytes)
    loop = asyncio.get_event_loop()

    if await db.doc_exists(pool, doc_id):
        doc = await db.get_latest_doc(pool)
        logger.info("doc_id=%s already indexed — skipping re-embedding", doc_id)
        yield {
            "status": "done",
            "doc_id": doc_id,
            "chunk_count": doc["chunk_count"],
            "k": doc["k"],
            "cached": True,
            "message": "Document already indexed — loaded from cache",
        }
        return

    yield {"status": "extracting", "message": "Extracting text from PDF…"}
    text, pages = await loop.run_in_executor(None, _extract_text, pdf_bytes)

    ocr_page_set: set[int] = set()
    if ocr_scanned:
        empty_count = sum(
            1 for p in pages if len(p.strip()) <= _OCR_EMPTY_PAGE_THRESHOLD
        )
        if empty_count:
            yield {
                "status": "ocr",
                "message": f"OCRing {empty_count} scanned page(s)…",
            }
            pages, ocr_page_set = await _ocr_empty_pages(pdf_bytes, pages)
            text = "\n\n".join(p for p in pages if p.strip())

        # Path 3: detect hand-drawn markup baked into the page raster
        # (underlines, highlights, margin notes, circles). Layered PDF
        # annotations are already captured by `_format_annotations` inside
        # `_extract_text`; this pass catches the case where the markup is
        # flattened into the rendered page image.
        marked_pages = sum(1 for p in pages if p.strip())
        if marked_pages:
            yield {
                "status": "markup",
                "message": f"Scanning {marked_pages} page(s) for hand-drawn markup…",
            }
            markup_by_page = await _collect_visual_markup(pdf_bytes, pages)
            for page_num, items in markup_by_page.items():
                summary = _format_markup_summary(items)
                if summary:
                    pages[page_num - 1] = pages[page_num - 1] + summary
            if markup_by_page:
                text = "\n\n".join(p for p in pages if p.strip())

    yield {"status": "chunking", "message": "Splitting into chunks…"}
    total_tokens = len(_tokenizer.encode(text))
    chunk_size, k = compute_params(total_tokens)
    logger.info(
        "Document: %d tokens → chunk_size=%d, k=%d", total_tokens, chunk_size, k
    )
    chunks = await loop.run_in_executor(None, chunk_text, text, chunk_size)
    total = len(chunks)
    logger.info("Split into %d chunks", total)

    yield {"status": "clearing", "message": "Clearing previous document…"}
    await db.clear_all_chunks(pool)

    yield {
        "status": "embedding",
        "message": f"Embedding {total} chunks…",
        "progress": 0,
    }
    embeddings = list(
        await _gather_bounded([_embed_one(c) for c in chunks], _GEMINI_CONCURRENCY)
    )

    parents = _parent_texts(chunks)
    page_nums = [_find_chunk_page(c, pages) for c in chunks]
    rows = [
        (i, chunk, parent, emb, page_num)
        for i, (chunk, parent, emb, page_num) in enumerate(
            zip(chunks, parents, embeddings, page_nums)
        )
    ]
    await db.insert_chunks(pool, doc_id, rows)
    text_chunk_count = total

    image_chunk_count = 0
    images: list = []
    if process_images:
        yield {
            "status": "extracting_images",
            "message": "Extracting images from PDF…",
        }
        images = await loop.run_in_executor(
            None,
            lambda: _extract_images(pdf_bytes, skip_pages=ocr_page_set),
        )
    else:
        logger.info("Image processing skipped (text-only ingest)")
    if images:
        yield {
            "status": "captioning",
            "message": f"Captioning {len(images)} images…",
        }
        captions = list(
            await _gather_bounded(
                [_caption_image(b, m) for _, b, m in images], _GEMINI_CONCURRENCY
            )
        )

        # Drop blank-captioned images (decorative backgrounds, page renders
        # with text overlay, etc.) before we spend embed calls on them.
        keep = [i for i, cap in enumerate(captions) if not _is_blank_caption(cap)]
        if len(keep) < len(images):
            logger.info(
                "Filtered %d/%d image(s) with blank captions",
                len(images) - len(keep),
                len(images),
            )
            images = [images[i] for i in keep]
            captions = [captions[i] for i in keep]

    if images:
        yield {"status": "embedding_images", "message": "Embedding image captions…"}
        img_embeddings = list(
            await _gather_bounded(
                [_embed_one(c) for c in captions], _GEMINI_CONCURRENCY
            )
        )

        def _adjacent_page_text(page_num: int) -> str | None:
            # Include prev/this/next page text so facing-page layouts (image on
            # one page, title text on the opposite page) still match the title
            # back to the right image.
            if not pages or not (0 < page_num <= len(pages)):
                return None
            lo, hi = max(1, page_num - 1), min(len(pages), page_num + 1)
            joined = " ".join(p for p in pages[lo - 1 : hi] if p.strip())
            return joined or None

        image_rows = [
            (
                text_chunk_count + i,
                captions[i],
                _adjacent_page_text(images[i][0]),
                img_embeddings[i],
                images[i][0],  # page number
                images[i][1],  # image bytes
                images[i][2],  # mime type
            )
            for i in range(len(images))
        ]
        await db.insert_image_chunks(pool, doc_id, image_rows)
        image_chunk_count = len(images)
        logger.info("Inserted %d image chunks", image_chunk_count)

    total = text_chunk_count + image_chunk_count
    await db.upsert_doc_meta(pool, doc_id, total, k)

    yield {
        "status": "done",
        "doc_id": doc_id,
        "chunk_count": total,
        "image_count": image_chunk_count,
        "k": k,
        "message": f"Done — {text_chunk_count} text + {image_chunk_count} image chunks stored",
    }
