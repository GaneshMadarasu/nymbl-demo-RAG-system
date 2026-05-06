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
    "You are a document Q&A assistant. Answer ONLY using the provided context.\n"
    "If the context doesn't contain enough information, respond with exactly: \"I don't know.\"\n"
    "Cite sources as [Chunk N] inline."
)


async def _embed_query(text: str) -> list[float]:
    r = await _client.aio.models.embed_content(
        model=_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=_EMBED_DIM,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return list(r.embeddings[0].values)


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
    async for chunk in await _client.aio.models.generate_content_stream(
        model=_GEN_MODEL,
        contents=prompt,
    ):
        if chunk.text:
            yield {"type": "token", "text": chunk.text}
    logger.info("Gemini answer took %.2fs", time.monotonic() - t0)

    yield {"type": "done"}
