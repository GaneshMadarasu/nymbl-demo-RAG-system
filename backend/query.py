import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator

from google import genai
from google.genai import types

from backend import db
from backend.config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.gemini_api_key)

_GEN_MODEL = "gemini-2.5-flash"
_RERANK_MODEL = "gemini-2.0-flash-lite"
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


async def _rerank(question: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Ask Gemini to re-rank chunks by relevance; falls back to original order on failure."""
    if len(chunks) <= top_n:
        return chunks
    snippets = "\n\n".join(f"[{i}]: {c['text'][:400]}" for i, c in enumerate(chunks))
    prompt = (
        "Rank these text chunks by relevance to the question. "
        "Return ONLY a JSON array of indices, most relevant first, e.g. [2,0,1].\n\n"
        f"Question: {question}\n\nChunks:\n{snippets}"
    )
    try:
        r = await _client.aio.models.generate_content(
            model=_RERANK_MODEL, contents=prompt
        )
        match = re.search(r"\[[\d,\s]+\]", r.text or "")
        if not match:
            return chunks[:top_n]
        indices = json.loads(match.group())
        seen: set[int] = set()
        result: list[dict] = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(chunks) and i not in seen:
                result.append(chunks[i])
                seen.add(i)
        for i, c in enumerate(chunks):
            if i not in seen:
                result.append(c)
        return result[:top_n]
    except Exception as exc:
        logger.warning("Re-ranking failed (%s); using retrieval order", exc)
        return chunks[:top_n]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n".join(
        f'[Chunk {r["chunk_index"]}]: "{r.get("parent_text") or r["text"]}"'
        for r in chunks
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


async def run_query(
    question: str, doc_id: str, k: int = 8
) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()

    t0 = time.monotonic()
    query_emb = await _embed_query(question)
    raw_chunks = await db.search_chunks(pool, doc_id, query_emb, question=question, k=k)
    retrieval_elapsed = time.monotonic() - t0

    if raw_chunks:
        sims = [c["similarity"] for c in raw_chunks]
        logger.info(
            "Retrieval took %.2fs — %d chunks, avg_sim=%.3f, min_sim=%.3f",
            retrieval_elapsed,
            len(raw_chunks),
            sum(sims) / len(sims),
            min(sims),
        )

    if not raw_chunks:
        yield {"type": "token", "text": "I don't know."}
        yield {"type": "done"}
        return

    rerank_top = max(1, k // 2)
    t_rerank = time.monotonic()
    chunks = await _rerank(question, raw_chunks, top_n=rerank_top)
    logger.info(
        "Re-ranking took %.2fs — kept %d/%d chunks",
        time.monotonic() - t_rerank,
        len(chunks),
        len(raw_chunks),
    )

    yield {
        "type": "sources",
        "chunks": [
            {
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "similarity": round(float(c["similarity"]), 3),
            }
            for c in chunks
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
