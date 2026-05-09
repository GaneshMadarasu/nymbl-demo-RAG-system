import asyncio
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
_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIM = 768

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer using ONLY the provided context.\n"
    "Match answer length to the question: brief factual questions get concise answers; "
    "complex or open-ended questions get thorough, detailed answers. "
    "If the user explicitly asks for a short, brief, or summary answer, be concise. "
    "For detailed questions, cover all relevant points fully, include examples and steps from the context, "
    "and use markdown headings, bullet lists, and bold text to structure long answers.\n"
    "If the context doesn't contain enough information, respond with exactly: \"I don't know.\"\n"
    "Cite sources as [Chunk N] inline throughout your answer whenever you use information from that chunk. "
    "Only cite the exact [Chunk N] numbers listed in the Context section. Never invent or interpolate chunk numbers.\n"
    "When you reference an artwork, painting, figure, or diagram that is provided as an image chunk "
    "(one labeled '(image, page N)' in the Context), format the artwork's title or main visual subject "
    "as a markdown link to image:N — for example: **[Bust portrait of a man](image:5)** [Chunk 5]. "
    "Use this format ONLY for image chunks. The frontend renders these links as clickable image previews."
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


_PRONOUN_RE = re.compile(
    r"\b(it|this|that|they|them|its|their|these|those|he|she|we|here|there)\b", re.I
)


async def _rewrite_query(question: str, history: list[dict]) -> str:
    """Make a follow-up question self-contained by resolving pronouns/references."""
    if not history or not _PRONOUN_RE.search(question):
        return question
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    )
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        "Rewrite the follow-up question below so it is fully self-contained — "
        "replace all pronouns and vague references with their explicit referents. "
        "Return ONLY the rewritten question, no explanation.\n\n"
        f"Follow-up question: {question}"
    )
    try:
        r = await _client.aio.models.generate_content(model=_GEN_MODEL, contents=prompt)
        rewritten = (r.text or question).strip()
        logger.info("Query rewritten: %r → %r", question, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("Query rewrite failed (%s); using original", exc)
        return question


_CHUNK_REF_RE = re.compile(r"\[Chunk\s+\d+\]", re.I)


def _history_section(history: list[dict]) -> str:
    if not history:
        return ""
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {_CHUNK_REF_RE.sub('', m['content']).strip()}"
        for m in history[-6:]
    )
    return f"\n\nConversation so far:\n{history_text}"


def build_prompt(question: str, chunks: list[dict], history: list[dict]) -> str:
    # Use sequential 1-N labels so the model can't cite out-of-range DB indexes.
    context = "\n".join(
        f'[Chunk {i + 1}]: "{r.get("parent_text") or r["text"]}"'
        for i, r in enumerate(chunks)
    )
    return f"{SYSTEM_PROMPT}{_history_section(history)}\n\nContext:\n{context}\n\nQuestion: {question}"


def _build_multimodal_contents(
    question: str,
    chunks: list[dict],
    history: list[dict],
    images: dict[int, dict],
) -> list:
    """Build a Gemini contents list interleaving text and image parts.
    `images` maps chunk_index → {image_data, image_mime, caption, page_number}.
    Image parts are inserted right after their [Chunk N] caption so the model
    sees the picture in context."""
    parts: list = [f"{SYSTEM_PROMPT}{_history_section(history)}\n\nContext:"]
    for i, c in enumerate(chunks):
        label = i + 1
        if c.get("chunk_type") == "image" and c["chunk_index"] in images:
            img = images[c["chunk_index"]]
            parts.append(
                f'[Chunk {label}] (image, page {img["page_number"]}): "{img["caption"]}"'
            )
            parts.append(
                types.Part.from_bytes(
                    data=img["image_data"], mime_type=img["image_mime"]
                )
            )
        else:
            text = c.get("parent_text") or c["text"]
            parts.append(f'[Chunk {label}]: "{text}"')
    parts.append(f"\nQuestion: {question}")
    return parts


async def run_query(
    question: str, doc_id: str, k: int = 8, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    history = history or []
    pool = await db.get_pool()

    if history and _PRONOUN_RE.search(question):
        yield {"type": "status", "text": "Rewriting query…"}
    retrieval_question = await _rewrite_query(question, history)

    yield {"type": "status", "text": "Embedding query…"}
    t0 = time.monotonic()
    query_emb = await _embed_query(retrieval_question)

    yield {"type": "status", "text": "Searching database…"}
    raw_chunks = await db.search_chunks(
        pool, doc_id, query_emb, question=retrieval_question, k=k
    )
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

    yield {
        "type": "sources",
        "chunks": [
            {
                "position": i + 1,  # matches [Chunk N] in the prompt (1-indexed)
                "chunk_index": c["chunk_index"],
                "chunk_type": c.get("chunk_type", "text"),
                "text": c["text"],
                "similarity": round(float(c["similarity"]), 3),
                "rrf_score": round(float(c["rrf_score"]), 6),
                "page_number": c.get("page_number"),
            }
            for i, c in enumerate(raw_chunks)
        ],
    }

    # Fetch image bytes for any image chunks in the retrieved set.
    images: dict[int, dict] = {}
    for c in raw_chunks:
        if c.get("chunk_type") == "image":
            img = await db.get_chunk_image(pool, doc_id, c["chunk_index"])
            if img:
                images[c["chunk_index"]] = img
    if images:
        logger.info("Passing %d image(s) to Gemini for multimodal answer", len(images))

    contents = _build_multimodal_contents(question, raw_chunks, history, images)

    t0 = time.monotonic()
    async for chunk in await _client.aio.models.generate_content_stream(
        model=_GEN_MODEL,
        contents=contents,
    ):
        if chunk.text:
            yield {"type": "token", "text": chunk.text}
    logger.info("Gemini answer took %.2fs", time.monotonic() - t0)

    yield {"type": "done"}
