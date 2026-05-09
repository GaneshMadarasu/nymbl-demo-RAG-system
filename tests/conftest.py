import os

# Set before any backend import so config.py doesn't sys.exit in CI/test env.
# Real GEMINI_API_KEY in .env overrides this; all Gemini calls are mocked in tests anyway.
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder")
os.environ.setdefault("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragdb")

import pytest
import asyncpg
from backend.config import settings


@pytest.fixture
async def pool():
    from backend.db import SCHEMA, MIGRATION

    p = await asyncpg.create_pool(settings.database_url)
    async with p.acquire() as conn:
        await conn.execute(SCHEMA)
        await conn.execute(MIGRATION)
    yield p
    async with p.acquire() as conn:
        await conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
        await conn.execute("TRUNCATE TABLE doc_meta")
        await conn.execute("TRUNCATE TABLE ocr_lines")
    await p.close()
