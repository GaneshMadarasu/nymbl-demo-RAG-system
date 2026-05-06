import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator

import google.generativeai as genai

from backend import db
from backend.chunks import chunk_text
from backend.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.gemini_api_key)
_gen_model = genai.GenerativeModel("gemini-1.5-flash")

_BATCH_SIZE = 20


def _doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


async def _extract_text(pdf_bytes: bytes) -> str:
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()
    response = await loop.run_in_executor(
        None,
        lambda: _gen_model.generate_content(
            [
                "Extract all text from this document. Preserve paragraph structure.",
                {"mime_type": "application/pdf", "data": pdf_bytes},
            ]
        ),
    )
    logger.info("Gemini text extraction took %.2fs", time.monotonic() - t0)
    return response.text


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: genai.embed_content(
            model="models/gemini-embedding-004",
            content=texts,
            task_type="retrieval_document",
        ),
    )
    return result["embedding"]


async def run_ingest(pdf_bytes: bytes) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()
    doc_id = _doc_id(pdf_bytes)

    yield {"status": "extracting", "message": "Extracting text from PDF…"}
    text = await _extract_text(pdf_bytes)
    logger.info("Extracted %d characters", len(text))

    yield {"status": "chunking", "message": "Splitting into chunks…"}
    chunks = chunk_text(text)
    total = len(chunks)
    logger.info("Split into %d chunks", total)

    yield {"status": "clearing", "message": "Clearing previous document…"}
    await db.clear_all_chunks(pool)

    rows: list[tuple[int, str, list[float]]] = []
    for i in range(0, total, _BATCH_SIZE):
        batch_texts = chunks[i : i + _BATCH_SIZE]
        embeddings = await _embed_batch(batch_texts)
        for j, (chunk_text_val, emb) in enumerate(zip(batch_texts, embeddings)):
            rows.append((i + j, chunk_text_val, emb))
        done_so_far = min(i + _BATCH_SIZE, total)
        yield {
            "status": "embedding",
            "message": f"Embedding {done_so_far}/{total}…",
            "progress": done_so_far / total,
        }

    t0 = time.monotonic()
    await db.insert_chunks(pool, doc_id, rows)
    logger.info("Inserted %d chunks in %.2fs", total, time.monotonic() - t0)

    yield {
        "status": "done",
        "doc_id": doc_id,
        "chunk_count": total,
        "message": f"Done — {total} chunks stored",
    }
