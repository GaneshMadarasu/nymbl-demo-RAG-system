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


def _extract_text(pdf_bytes: bytes) -> tuple[str, list[str]]:
    """Returns (full_text, per_page_texts) where per_page_texts is 1-indexed (index 0 unused)."""
    t0 = time.monotonic()
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text().replace("\x00", "") for page in doc]
    doc.close()
    text = "\n\n".join(p for p in pages if p.strip())
    logger.info(
        "PyMuPDF extracted %d chars from %d pages in %.2fs",
        len(text),
        len(pages),
        time.monotonic() - t0,
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


def _extract_images(pdf_bytes: bytes) -> list[tuple[int, bytes, str]]:
    """Extract paintings/figures from a PDF in two stages.

    Stage 1 — embedded images: walk every page's image XRefs and use the raw
    embedded bytes. Re-rendering the page bbox would also capture any text
    overlay on top of a background image, so we trust the embedded bytes and
    let Vision filter out blank/decorative ones via caption.

    Stage 2 — page-render fallback: for pages with no embedded images AND very
    little text, render the whole page. Catches scanned/full-bleed painting
    pages where PyMuPDF reports no embedded images at all.

    Returns list of (page_num, image_bytes, mime_type)."""
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


async def _caption_image(image_bytes: bytes, mime_type: str) -> str:
    """Caption an image using Gemini Vision. Retries on retryable errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.generate_content(
                model=_VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
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


async def run_ingest(pdf_bytes: bytes) -> AsyncGenerator[dict, None]:
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
    embeddings = list(await asyncio.gather(*[_embed_one(c) for c in chunks]))

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

    yield {"status": "extracting_images", "message": "Extracting images from PDF…"}
    images = await loop.run_in_executor(None, _extract_images, pdf_bytes)
    image_chunk_count = 0
    if images:
        yield {
            "status": "captioning",
            "message": f"Captioning {len(images)} images…",
        }
        captions = list(
            await asyncio.gather(*[_caption_image(b, m) for _, b, m in images])
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
        img_embeddings = list(await asyncio.gather(*[_embed_one(c) for c in captions]))

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
