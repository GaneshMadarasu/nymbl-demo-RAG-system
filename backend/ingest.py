import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from io import BytesIO

import fitz  # pymupdf
from google import genai
from google.genai import types

from backend import db
from backend.chunks import chunk_text
from backend.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

_BATCH_SIZE = 20
_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIM = 768
_MAX_RETRIES = 6
_RETRY_BASE = 2.0  # seconds; delays: 2, 4, 8, 16, 32, 64


def _doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _extract_text(pdf_bytes: bytes) -> str:
    t0 = time.monotonic()
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n\n".join(p for p in pages if p.strip())
    logger.info(
        "PyMuPDF extracted %d chars from %d pages in %.2fs",
        len(text),
        len(pages),
        time.monotonic() - t0,
    )
    return text


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "ResourceExhausted" in type(exc).__name__


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
            if attempt < _MAX_RETRIES - 1 and _is_rate_limit(exc):
                wait = _RETRY_BASE**attempt
                logger.warning("Embedding rate-limited; retrying in %.0fs", wait)
                await asyncio.sleep(wait)
                continue
            raise


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    return list(await asyncio.gather(*[_embed_one(t) for t in texts]))


async def run_ingest(pdf_bytes: bytes) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()
    doc_id = _doc_id(pdf_bytes)

    yield {"status": "extracting", "message": "Extracting text from PDF…"}
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _extract_text, pdf_bytes)

    yield {"status": "chunking", "message": "Splitting into chunks…"}
    chunks = chunk_text(text)
    total = len(chunks)
    logger.info("Split into %d chunks", total)

    yield {"status": "clearing", "message": "Clearing previous document…"}
    await db.clear_all_chunks(pool)

    # Embed and insert each batch immediately — avoids accumulating all
    # embeddings in RAM before the first DB write (critical for large docs).
    for i in range(0, total, _BATCH_SIZE):
        batch_texts = chunks[i : i + _BATCH_SIZE]
        embeddings = await _embed_batch(batch_texts)
        batch_rows = [
            (i + j, chunk_text_val, emb)
            for j, (chunk_text_val, emb) in enumerate(zip(batch_texts, embeddings))
        ]
        await db.insert_chunks(pool, doc_id, batch_rows)
        done_so_far = min(i + _BATCH_SIZE, total)
        yield {
            "status": "embedding",
            "message": f"Embedding {done_so_far}/{total}…",
            "progress": done_so_far / total,
        }

    yield {
        "status": "done",
        "doc_id": doc_id,
        "chunk_count": total,
        "message": f"Done — {total} chunks stored",
    }
