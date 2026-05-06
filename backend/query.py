import asyncio
import logging
import time
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types

from backend import db
from backend.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

_GEN_MODEL = "gemini-2.5-flash"
_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIM = 768

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer using ONLY the provided context.\n"
    "Give thorough, detailed answers — explain concepts fully, include relevant examples "
    "or steps from the context, and expand on any related points the context covers.\n"
    "If the context doesn't contain enough information, respond with exactly: \"I don't know.\"\n"
    "Cite sources as [Chunk N] inline throughout your answer."
)


_MAX_RETRIES = 6
_RETRY_BASE = 2.0


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


async def _embed_query(text: str) -> list[float]:
    for attempt in range(_MAX_RETRIES):
        try:
            r = await _client.aio.models.embed_content(
                model=_EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=_EMBED_DIM,
                    task_type="RETRIEVAL_QUERY",
                ),
            )
            return list(r.embeddings[0].values)
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1 and _is_retryable(exc):
                wait = _RETRY_BASE**attempt
                logger.warning("Query embed failed (%s); retrying in %.0fs", exc, wait)
                await asyncio.sleep(wait)
                continue
            raise


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n".join(f'[Chunk {r["chunk_index"]}]: "{r["text"]}"' for r in chunks)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


async def run_query(
    question: str, doc_id: str, k: int = 8
) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()

    t0 = time.monotonic()
    query_emb = await _embed_query(question)
    chunks = await db.search_chunks(pool, doc_id, query_emb, k=k)
    logger.info(
        "Retrieval took %.2fs, got %d chunks", time.monotonic() - t0, len(chunks)
    )

    if not chunks:
        yield {"type": "token", "text": "I don't know."}
        yield {"type": "done"}
        return

    yield {
        "type": "sources",
        "chunks": [
            {"chunk_index": c["chunk_index"], "text": c["text"]} for c in chunks
        ],
    }

    prompt = build_prompt(question, chunks)

    t0 = time.monotonic()
    async for chunk in await _client.aio.models.generate_content_stream(
        model=_GEN_MODEL,
        contents=prompt,
    ):
        if chunk.text:
            yield {"type": "token", "text": chunk.text}
    logger.info("Gemini answer took %.2fs", time.monotonic() - t0)

    yield {"type": "done"}
