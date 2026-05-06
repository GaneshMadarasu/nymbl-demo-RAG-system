import asyncio
import asyncpg

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    async with _pool_lock:
        if _pool is None:
            from backend.config import settings

            try:
                _pool = await asyncpg.create_pool(settings.database_url)
                async with _pool.acquire() as conn:
                    await conn.execute(SCHEMA)
            except Exception:
                _pool = None
                raise
    return _pool


def _vec(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def insert_chunks(
    pool: asyncpg.Pool,
    doc_id: str,
    chunks: list[tuple[int, str, list[float]]],
) -> None:
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, text, embedding) "
            "VALUES ($1, $2, $3, $4::vector)",
            [(doc_id, idx, text, _vec(emb)) for idx, text, emb in chunks],
        )


async def search_chunks(
    pool: asyncpg.Pool,
    doc_id: str,
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, chunk_index, text, "
            "1 - (embedding <=> $1::vector) AS similarity "
            "FROM chunks WHERE doc_id = $2 "
            "ORDER BY embedding <=> $1::vector LIMIT $3",
            _vec(query_embedding),
            doc_id,
            k,
        )
    return [dict(r) for r in rows]


async def clear_all_chunks(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")


async def get_doc_info(pool: asyncpg.Pool, doc_id: str) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS chunk_count FROM chunks WHERE doc_id = $1",
            doc_id,
        )
    if row and row["chunk_count"] > 0:
        return {"chunk_count": row["chunk_count"], "embedding_dim": 768}
    return None
