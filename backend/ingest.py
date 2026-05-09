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

_CAPTION_PROMPT = (
    "Describe this image from a PDF document in 1-2 concise sentences. "
    "Begin by stating the type of image (chart, diagram, photo, screenshot, table, illustration). "
    "Mention any visible text, axis labels, or data shown. Be specific and factual."
)


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
    """Extract images from a PDF. Returns list of (page_num, image_bytes, mime_type).
    Filters out small/decorative images and unsupported formats."""
    t0 = time.monotonic()
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    images: list[tuple[int, bytes, str]] = []
    seen: set[int] = set()  # dedupe by xref across pages
    for page_num, page in enumerate(doc, 1):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
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
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
            }.get(ext, f"image/{ext}")
            if mime not in _SUPPORTED_MIMES:
                continue
            images.append((page_num, img_bytes, mime))
    doc.close()
    logger.info(
        "PyMuPDF extracted %d filtered images in %.2fs",
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

        yield {"status": "embedding_images", "message": "Embedding image captions…"}
        img_embeddings = list(await asyncio.gather(*[_embed_one(c) for c in captions]))

        image_rows = [
            (
                text_chunk_count + i,
                captions[i],
                None,
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
