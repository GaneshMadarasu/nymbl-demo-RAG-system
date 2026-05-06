import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from io import BytesIO

import fitz  # pymupdf
from fastembed import TextEmbedding

from backend import db
from backend.chunks import chunk_text

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20
_EMBED_DIM = 768
_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

_embedder: TextEmbedding | None = None


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(_EMBED_MODEL_NAME)
    return _embedder


def _doc_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


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


def _embed_batch_sync(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    return [list(v) for v in embedder.embed(texts)]


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_batch_sync, texts)


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
