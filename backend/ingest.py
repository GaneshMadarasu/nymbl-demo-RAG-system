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
_EMBED_DIM = 768
_MAX_RETRIES = 6
_RETRY_BASE = 2.0  # delays: 2, 4, 8, 16, 32, 64s


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


def _extract_text(pdf_bytes: bytes) -> str:
    t0 = time.monotonic()
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n\n".join(p for p in pages if p.strip()).replace("\x00", "")
    logger.info(
        "PyMuPDF extracted %d chars from %d pages in %.2fs",
        len(text),
        len(pages),
        time.monotonic() - t0,
    )
    return text


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
    text = await loop.run_in_executor(None, _extract_text, pdf_bytes)

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
    rows = [
        (i, chunk, parent, emb)
        for i, (chunk, parent, emb) in enumerate(zip(chunks, parents, embeddings))
    ]
    await db.insert_chunks(pool, doc_id, rows)
    await db.upsert_doc_meta(pool, doc_id, total, k)

    yield {
        "status": "done",
        "doc_id": doc_id,
        "chunk_count": total,
        "k": k,
        "message": f"Done — {total} chunks stored",
    }
