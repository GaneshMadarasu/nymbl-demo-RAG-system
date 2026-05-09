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
    parent_text TEXT,
    page_number INTEGER,
    chunk_type  TEXT NOT NULL DEFAULT 'text',
    image_data  BYTEA,
    image_mime  TEXT,
    embedding   vector(768) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_meta (
    doc_id      TEXT PRIMARY KEY,
    chunk_count INTEGER NOT NULL,
    k           INTEGER NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""

# Adds columns/indexes that may be missing on existing installations.
MIGRATION = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'parent_text'
    ) THEN
        ALTER TABLE chunks ADD COLUMN parent_text TEXT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'tsv'
    ) THEN
        ALTER TABLE chunks ADD COLUMN tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;
    END IF;
    CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin(tsv);
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'page_number'
    ) THEN
        ALTER TABLE chunks ADD COLUMN page_number INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'chunk_type'
    ) THEN
        ALTER TABLE chunks ADD COLUMN chunk_type TEXT NOT NULL DEFAULT 'text';
        ALTER TABLE chunks ADD COLUMN image_data BYTEA;
        ALTER TABLE chunks ADD COLUMN image_mime TEXT;
    END IF;
END
$$;
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
                    await conn.execute(MIGRATION)
            except Exception:
                _pool = None
                raise
    return _pool


def _vec(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def insert_chunks(
    pool: asyncpg.Pool,
    doc_id: str,
    chunks: list[tuple[int, str, str | None, list[float], int | None]],
) -> None:
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, text, parent_text, embedding, page_number) "
            "VALUES ($1, $2, $3, $4, $5::vector, $6)",
            [
                (doc_id, idx, text, parent_text, _vec(emb), page_num)
                for idx, text, parent_text, emb, page_num in chunks
            ],
        )


async def insert_image_chunks(
    pool: asyncpg.Pool,
    doc_id: str,
    chunks: list[tuple[int, str, str | None, list[float], int | None, bytes, str]],
) -> None:
    """Insert image chunks: (idx, caption, parent_text, embedding, page_num, image_bytes, mime)"""
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, text, parent_text, embedding, page_number, "
            "chunk_type, image_data, image_mime) "
            "VALUES ($1, $2, $3, $4, $5::vector, $6, 'image', $7, $8)",
            [
                (doc_id, idx, text, parent_text, _vec(emb), page_num, img_bytes, mime)
                for idx, text, parent_text, emb, page_num, img_bytes, mime in chunks
            ],
        )


_HYBRID_SQL = """
WITH
dense AS (
    SELECT chunk_index, text, parent_text, page_number, chunk_type,
           1 - (embedding <=> $1::vector) AS similarity,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rank
    FROM chunks WHERE doc_id = $2
    ORDER BY embedding <=> $1::vector LIMIT $3 * 2
),
sparse AS (
    SELECT chunk_index, text, parent_text, page_number, chunk_type,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC
           ) AS rank
    FROM chunks
    WHERE doc_id = $2
      AND tsv @@ plainto_tsquery('english', $4)
    ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC
    LIMIT $3 * 2
),
fused AS (
    SELECT
        COALESCE(d.chunk_index, s.chunk_index)   AS chunk_index,
        COALESCE(d.text, s.text)                 AS text,
        COALESCE(d.parent_text, s.parent_text)   AS parent_text,
        COALESCE(d.page_number, s.page_number)   AS page_number,
        COALESCE(d.chunk_type, s.chunk_type)     AS chunk_type,
        COALESCE(d.similarity, 0.0)              AS similarity,
        COALESCE(1.0 / (60.0 + d.rank), 0.0) +
        COALESCE(1.0 / (60.0 + s.rank), 0.0)    AS rrf_score
    FROM dense d
    FULL OUTER JOIN sparse s ON d.chunk_index = s.chunk_index
)
SELECT chunk_index, text, parent_text, page_number, chunk_type, similarity, rrf_score
FROM fused
ORDER BY rrf_score DESC
LIMIT $3
"""

_DENSE_SQL = """
SELECT chunk_index, text, parent_text, page_number, chunk_type,
       1 - (embedding <=> $1::vector) AS similarity,
       1 - (embedding <=> $1::vector) AS rrf_score
FROM chunks WHERE doc_id = $2
ORDER BY embedding <=> $1::vector LIMIT $3
"""


async def search_chunks(
    pool: asyncpg.Pool,
    doc_id: str,
    query_embedding: list[float],
    question: str = "",
    k: int = 5,
) -> list[dict]:
    vec = _vec(query_embedding)
    async with pool.acquire() as conn:
        if question.strip():
            rows = await conn.fetch(_HYBRID_SQL, vec, doc_id, k, question)
        else:
            rows = await conn.fetch(_DENSE_SQL, vec, doc_id, k)
    return [dict(r) for r in rows]


async def get_chunk_text(
    pool: asyncpg.Pool, doc_id: str, chunk_index: int
) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT text, page_number FROM chunks WHERE doc_id = $1 AND chunk_index = $2",
            doc_id,
            chunk_index,
        )
    return {"text": row["text"], "page_number": row["page_number"]} if row else None


async def list_doc_images(pool: asyncpg.Pool, doc_id: str) -> list[dict]:
    """List all image chunks for a doc with caption + page-text (no bytes)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT chunk_index, page_number, text AS caption, parent_text AS page_text "
            "FROM chunks WHERE doc_id = $1 AND chunk_type = 'image' "
            "ORDER BY chunk_index",
            doc_id,
        )
    return [dict(r) for r in rows]


async def get_chunk_image(
    pool: asyncpg.Pool, doc_id: str, chunk_index: int
) -> dict | None:
    """Fetch image bytes + metadata for an image chunk."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT image_data, image_mime, text, page_number FROM chunks "
            "WHERE doc_id = $1 AND chunk_index = $2 AND chunk_type = 'image'",
            doc_id,
            chunk_index,
        )
    if not row or not row["image_data"]:
        return None
    return {
        "image_data": bytes(row["image_data"]),
        "image_mime": row["image_mime"],
        "caption": row["text"],
        "page_number": row["page_number"],
    }


async def doc_exists(pool: asyncpg.Pool, doc_id: str) -> bool:
    async with pool.acquire() as conn:
        meta = await conn.fetchrow("SELECT 1 FROM doc_meta WHERE doc_id = $1", doc_id)
        if not meta:
            return False
        chunks = await conn.fetchrow(
            "SELECT 1 FROM chunks WHERE doc_id = $1 LIMIT 1", doc_id
        )
    return chunks is not None


async def upsert_doc_meta(
    pool: asyncpg.Pool, doc_id: str, chunk_count: int, k: int
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO doc_meta (doc_id, chunk_count, k) VALUES ($1, $2, $3) "
            "ON CONFLICT (doc_id) DO UPDATE SET chunk_count = $2, k = $3, created_at = now()",
            doc_id,
            chunk_count,
            k,
        )


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


async def get_latest_doc(pool: asyncpg.Pool) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT doc_id, chunk_count, k FROM doc_meta ORDER BY created_at DESC LIMIT 1"
        )
    if row:
        return {
            "doc_id": row["doc_id"],
            "chunk_count": row["chunk_count"],
            "k": row["k"],
        }
    return None
