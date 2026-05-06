import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import google.generativeai as genai

from backend import db
from backend.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.gemini_api_key)
_gen_model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer ONLY using the provided context.\n"
    "If the context doesn't contain enough information, respond with exactly: \"I don't know.\"\n"
    "Cite sources as [Chunk N] inline."
)


async def _embed_query(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: genai.embed_content(
            model="models/gemini-embedding-004",
            content=text,
            task_type="retrieval_query",
        ),
    )
    return result["embedding"]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n".join(f'[Chunk {r["chunk_index"]}]: "{r["text"]}"' for r in chunks)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


async def run_query(question: str, doc_id: str) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()

    t0 = time.monotonic()
    query_emb = await _embed_query(question)
    chunks = await db.search_chunks(pool, doc_id, query_emb, k=5)
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
    response = _gen_model.generate_content_async(prompt, stream=True)
    async for chunk in await response:
        if chunk.text:
            yield {"type": "token", "text": chunk.text}
    logger.info("Gemini answer took %.2fs", time.monotonic() - t0)

    yield {"type": "done"}
