import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types

from backend import db
from backend.chunks import chunk_text
from backend.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

_EXTRACT_MODEL = "gemini-2.5-flash"
_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIM = 768
_MAX_RETRIES = 6
_RETRY_BASE = 2.0  # delays: 2, 4, 8, 16, 32, 64s


def _doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


async def _extract_text(pdf_bytes: bytes) -> str:
    t0 = time.monotonic()
    for attempt in range(_MAX_RETRIES):
        try:
            response = await _client.aio.models.generate_content(
                model=_EXTRACT_MODEL,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    "Extract all text from this PDF as clean plaintext. Preserve paragraph structure. Remove page numbers, headers, and footers.",
                ],
                config=types.GenerateContentConfig(max_output_tokens=65536),
            )
            text = response.text or ""
            logger.info(
                "Gemini extracted %d chars in %.2fs", len(text), time.monotonic() - t0
            )
            return text
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning("Extraction failed (%s); retrying in %.0fs", exc, wait)
                await asyncio.sleep(wait)
                continue
            raise


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


async def run_ingest(pdf_bytes: bytes) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()
    doc_id = _doc_id(pdf_bytes)
    loop = asyncio.get_event_loop()

    yield {"status": "extracting", "message": "Extracting text with Gemini…"}
    text = await _extract_text(pdf_bytes)

    yield {"status": "chunking", "message": "Splitting into chunks…"}
    chunks = await loop.run_in_executor(None, chunk_text, text)
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

    rows = [(i, text, emb) for i, (text, emb) in enumerate(zip(chunks, embeddings))]
    await db.insert_chunks(pool, doc_id, rows)

    yield {
        "status": "done",
        "doc_id": doc_id,
        "chunk_count": total,
        "message": f"Done — {total} chunks stored",
    }
