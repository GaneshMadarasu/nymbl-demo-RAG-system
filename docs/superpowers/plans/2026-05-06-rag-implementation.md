# RAG System Implementation Plan

> **Status: Historical — superseded by current implementation.** This document records the implementation plan as written on **2026-05-06**. The shipped code has diverged: SDK migrated from `google-generativeai==0.8.3` to `google-genai==1.73.1`; embedding model upgraded `gemini-embedding-004` → `gemini-embedding-2`; generation/vision model upgraded `gemini-1.5-flash` → `gemini-2.5-flash`; retrieval `k` is now adaptive (5–20 via `compute_params`) instead of fixed at 5; the frontend is split into three HTML files (`index.html`, `viewer.html`, `image-viewer.html`) plus additional endpoints not in this plan. For the current state see [README.md](../../../README.md) and [docs/ARCHITECTURE.md](../../ARCHITECTURE.md). Preserved here as a record of the original implementation intent.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-document RAG web app that ingests a PDF, stores chunks + embeddings in pgvector, and answers questions via a dark-indigo split-pane chat UI with inline citations.

**Architecture:** FastAPI serves a single HTML frontend and four REST endpoints; ingestion calls Gemini 1.5 Flash for text extraction and `gemini-embedding-004` for embeddings, storing them in Postgres pgvector; queries embed the question, retrieve top-5 chunks by cosine similarity, and stream a grounded Gemini answer back via SSE.

**Tech Stack:** Python 3.11+, FastAPI, asyncpg, google-generativeai, tiktoken, pgvector (Docker), pytest + pytest-asyncio + httpx

---

## File Map

```
nymbl-demo-RAG-system/
├── backend/
│   ├── __init__.py          empty
│   ├── config.py            env var loading + fail-fast validation
│   ├── db.py                asyncpg pool, schema init, CRUD (insert/search/clear)
│   ├── chunks.py            pure text chunker (512 tokens, 64 overlap)
│   ├── ingest.py            PDF → Gemini text → chunks → embeddings → DB (async generator)
│   └── main.py              FastAPI app: /ingest /query /doc/info /health + serves frontend
├── frontend/
│   └── index.html           complete dark-indigo split-pane UI (vanilla HTML/CSS/JS)
├── tests/
│   ├── __init__.py          empty
│   ├── conftest.py          async DB pool fixture + cleanup
│   ├── test_db.py           insert, search, clear (real pgvector)
│   ├── test_chunks.py       pure unit tests for chunker
│   ├── test_ingest.py       ingest pipeline with mocked Gemini
│   ├── test_query.py        query pipeline with mocked Gemini
│   └── test_api.py          FastAPI endpoints via httpx AsyncClient
├── docker-compose.yml       Postgres 16 + pgvector, health-checked
├── pytest.ini               asyncio_mode = auto
├── .env.example
├── requirements.txt
└── README.md
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `docker-compose.yml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `pytest.ini`
- Create: `backend/__init__.py`
- Create: `backend/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d ragdb"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
asyncpg==0.29.0
google-generativeai==0.8.3
tiktoken==0.7.0
python-dotenv==1.0.1
python-multipart==0.0.9
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

- [ ] **Step 3: Create `.env.example`**

```
GEMINI_API_KEY=your_key_here
DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 5: Create `backend/__init__.py` and `tests/__init__.py`** (both empty files)

- [ ] **Step 6: Create `backend/config.py`**

```python
import os
import sys
from dotenv import load_dotenv

load_dotenv()


class _Settings:
    gemini_api_key: str
    database_url: str

    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.database_url = os.getenv("DATABASE_URL", "")
        missing = [k for k, v in vars(self).items() if not v]
        if missing:
            print(f"ERROR: missing required env vars: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)


settings = _Settings()
```

- [ ] **Step 7: Create `tests/conftest.py`**

```python
import os

# Set before any backend import so config.py doesn't sys.exit in CI/test env.
# Real GEMINI_API_KEY in .env overrides this; all Gemini calls are mocked in tests anyway.
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder")
os.environ.setdefault("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragdb")

import pytest
import asyncpg
from backend.config import settings
from backend.db import SCHEMA


@pytest.fixture
async def pool():
    p = await asyncpg.create_pool(settings.database_url)
    async with p.acquire() as conn:
        await conn.execute(SCHEMA)
    yield p
    async with p.acquire() as conn:
        await conn.execute("TRUNCATE TABLE chunks RESTART IDENTITY")
    await p.close()
```

- [ ] **Step 8: Start Postgres and install deps**

```bash
docker compose up -d
# wait for health check to pass (~10s)
pip install -r requirements.txt
```

Expected: `docker compose ps` shows `db` as `healthy`.

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml requirements.txt .env.example pytest.ini backend/__init__.py backend/config.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold project with docker, deps, config, test fixtures"
```

---

## Task 2: Database Layer

**Files:**
- Create: `backend/db.py`
- Create: `tests/test_db.py`

> **Requires:** Docker Compose DB running (`docker compose up -d`)

- [ ] **Step 1: Write failing tests in `tests/test_db.py`**

```python
import pytest
from backend.db import insert_chunks, search_chunks, clear_all_chunks, get_doc_info


async def test_insert_and_search(pool):
    await insert_chunks(pool, "doc1", [(0, "The cat sat on the mat.", [0.1] * 768)])
    results = await search_chunks(pool, "doc1", [0.1] * 768, k=1)
    assert len(results) == 1
    assert results[0]["text"] == "The cat sat on the mat."
    assert results[0]["chunk_index"] == 0


async def test_search_returns_most_similar_first(pool):
    await insert_chunks(pool, "doc2", [
        (0, "chunk zero", [1.0] + [0.0] * 767),
        (1, "chunk one",  [0.0] + [1.0] + [0.0] * 766),
    ])
    results = await search_chunks(pool, "doc2", [1.0] + [0.0] * 767, k=2)
    assert results[0]["chunk_index"] == 0


async def test_clear_all_removes_everything(pool):
    await insert_chunks(pool, "doc3", [(0, "text", [0.2] * 768)])
    await clear_all_chunks(pool)
    results = await search_chunks(pool, "doc3", [0.2] * 768, k=1)
    assert results == []


async def test_get_doc_info_returns_count(pool):
    await insert_chunks(pool, "doc4", [(0, "a", [0.3] * 768), (1, "b", [0.4] * 768)])
    info = await get_doc_info(pool, "doc4")
    assert info is not None
    assert info["chunk_count"] == 2
    assert info["embedding_dim"] == 768


async def test_get_doc_info_returns_none_for_missing(pool):
    assert await get_doc_info(pool, "no_such_doc") is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_db.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `backend.db` does not exist yet.

- [ ] **Step 3: Create `backend/db.py`**

```python
import asyncpg

_pool: asyncpg.Pool | None = None

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
    if _pool is None:
        from backend.config import settings
        _pool = await asyncpg.create_pool(settings.database_url)
        async with _pool.acquire() as conn:
            await conn.execute(SCHEMA)
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_db.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/db.py tests/test_db.py
git commit -m "feat(db): add asyncpg pool, schema init, and chunk CRUD"
```

---

## Task 3: Text Chunker

**Files:**
- Create: `backend/chunks.py`
- Create: `tests/test_chunks.py`

- [ ] **Step 1: Write failing tests in `tests/test_chunks.py`**

```python
from backend.chunks import chunk_text


def test_short_text_produces_one_chunk():
    result = chunk_text("Hello world. This is a short sentence.", max_tokens=512)
    assert len(result) == 1
    assert "Hello world" in result[0]


def test_long_text_splits_into_multiple_chunks():
    sentence = "The quick brown fox jumps over the lazy dog. " * 30
    result = chunk_text(sentence, max_tokens=100, overlap=10)
    assert len(result) > 1


def test_overlap_means_second_chunk_shares_tokens_with_first():
    # Build ~120 tokens of text then check overlap
    text = " ".join([f"word{i}." for i in range(80)])
    chunks = chunk_text(text, max_tokens=60, overlap=15)
    assert len(chunks) >= 2
    # Last token(s) of chunk 0 should appear in start of chunk 1
    end_of_first = chunks[0].split()[-5:]
    start_of_second = chunks[1].split()[:20]
    shared = set(end_of_first) & set(start_of_second)
    assert len(shared) > 0


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_no_chunk_is_empty_string():
    text = "Sentence one. Sentence two. Sentence three."
    result = chunk_text(text, max_tokens=512)
    assert all(c.strip() for c in result)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_chunks.py -v
```

Expected: `ImportError` — `backend.chunks` does not exist yet.

- [ ] **Step 3: Create `backend/chunks.py`**

```python
import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_tokens: list[int] = []

    for sentence in sentences:
        sentence_tokens = _enc.encode(sentence)
        if len(current_tokens) + len(sentence_tokens) > max_tokens:
            if current_tokens:
                chunks.append(_enc.decode(current_tokens))
            current_tokens = current_tokens[-overlap:] + sentence_tokens
        else:
            current_tokens.extend(sentence_tokens)

    if current_tokens:
        decoded = _enc.decode(current_tokens)
        if decoded.strip():
            chunks.append(decoded)

    return [c for c in chunks if c.strip()]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_chunks.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/chunks.py tests/test_chunks.py
git commit -m "feat(chunks): add sentence-aware text chunker with overlap"
```

---

## Task 4: Ingest Pipeline

**Files:**
- Create: `backend/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests in `tests/test_ingest.py`**

```python
import pytest
from unittest.mock import AsyncMock, patch
from backend.ingest import _doc_id, run_ingest


def test_doc_id_is_deterministic():
    b = b"hello pdf content"
    assert _doc_id(b) == _doc_id(b)


def test_doc_id_differs_for_different_content():
    assert _doc_id(b"pdf one") != _doc_id(b"pdf two")


def test_doc_id_is_16_chars():
    assert len(_doc_id(b"anything")) == 16


async def test_run_ingest_yields_done_event(pool):
    fake_text = "This is a sentence about artificial intelligence. " * 20

    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", new_callable=AsyncMock, return_value=fake_text),
        patch("backend.ingest._embed_batch", new_callable=AsyncMock, return_value=[[0.1] * 768] * 5),
    ):
        events = [e async for e in run_ingest(b"fake-pdf-bytes")]

    statuses = [e["status"] for e in events]
    assert "done" in statuses


async def test_run_ingest_done_event_has_chunk_count(pool):
    fake_text = "Sentence about topic. " * 30

    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.ingest._extract_text", new_callable=AsyncMock, return_value=fake_text),
        patch("backend.ingest._embed_batch", new_callable=AsyncMock, return_value=[[0.1] * 768] * 5),
    ):
        events = [e async for e in run_ingest(b"fake-pdf-bytes")]

    done = next(e for e in events if e["status"] == "done")
    assert done["chunk_count"] > 0
    assert "doc_id" in done
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ingest.py -v
```

Expected: `ImportError` — `backend.ingest` does not exist yet.

- [ ] **Step 3: Create `backend/ingest.py`**

```python
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
        lambda: _gen_model.generate_content([
            "Extract all text from this document. Preserve paragraph structure.",
            {"mime_type": "application/pdf", "data": pdf_bytes},
        ]),
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): add PDF ingestion pipeline with Gemini + pgvector"
```

---

## Task 5: Query Pipeline

**Files:**
- Create: `backend/query.py`
- Create: `tests/test_query.py`

- [ ] **Step 1: Write failing tests in `tests/test_query.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.query import build_prompt, run_query, SYSTEM_PROMPT


def test_build_prompt_includes_system_prompt():
    chunks = [{"chunk_index": 0, "text": "AI is transformative."}]
    prompt = build_prompt("What is AI?", chunks)
    assert SYSTEM_PROMPT in prompt


def test_build_prompt_formats_chunk_references():
    chunks = [
        {"chunk_index": 3, "text": "Deep learning uses neural networks."},
        {"chunk_index": 17, "text": "Transformers changed NLP."},
    ]
    prompt = build_prompt("What changed NLP?", chunks)
    assert "[Chunk 3]" in prompt
    assert "Deep learning uses neural networks." in prompt
    assert "[Chunk 17]" in prompt
    assert "What changed NLP?" in prompt


def test_build_prompt_includes_question():
    chunks = [{"chunk_index": 0, "text": "some text"}]
    question = "What are the key findings?"
    prompt = build_prompt(question, chunks)
    assert question in prompt


async def test_run_query_yields_sources_and_done(pool):
    from backend.db import insert_chunks
    import backend.query as bq

    await insert_chunks(pool, "qdoc1", [(0, "AI is great for research.", [0.1] * 768)])

    mock_chunk = MagicMock()
    mock_chunk.text = "AI is indeed great."

    # generate_content_async is called as: `await model.generate_content_async(prompt, stream=True)`
    # so the mock must return a coroutine that resolves to an async iterable.
    async def _fake_response(*args, **kwargs):
        async def _iter():
            yield mock_chunk
        return _iter()

    with (
        patch("backend.query.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.query._embed_query", new_callable=AsyncMock, return_value=[0.1] * 768),
        patch.object(bq._gen_model, "generate_content_async", side_effect=_fake_response),
    ):
        events = [e async for e in run_query("What is AI?", "qdoc1")]

    types = [e["type"] for e in events]
    assert "sources" in types
    assert "token" in types
    assert "done" in types


async def test_run_query_returns_i_dont_know_when_no_chunks(pool):
    with (
        patch("backend.query.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch("backend.query._embed_query", new_callable=AsyncMock, return_value=[0.5] * 768),
    ):
        events = [e async for e in run_query("What is the meaning of life?", "empty_doc")]

    token_events = [e for e in events if e["type"] == "token"]
    assert any("I don't know" in e["text"] for e in token_events)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_query.py -v
```

Expected: `ImportError` — `backend.query` does not exist yet.

- [ ] **Step 3: Create `backend/query.py`**

```python
import asyncio
import logging
import time
from collections.abc import AsyncGenerator  # noqa: F401 — used in type hints

import google.generativeai as genai

from backend import db
from backend.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.gemini_api_key)
_gen_model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer ONLY using the provided context.\n"
    'If the context doesn\'t contain enough information, respond with exactly: "I don\'t know."\n'
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
    context = "\n".join(
        f'[Chunk {r["chunk_index"]}]: "{r["text"]}"' for r in chunks
    )
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


async def run_query(question: str, doc_id: str) -> AsyncGenerator[dict, None]:
    pool = await db.get_pool()

    t0 = time.monotonic()
    query_emb = await _embed_query(question)
    chunks = await db.search_chunks(pool, doc_id, query_emb, k=5)
    logger.info("Retrieval took %.2fs, got %d chunks", time.monotonic() - t0, len(chunks))

    if not chunks:
        yield {"type": "token", "text": "I don't know."}
        yield {"type": "done"}
        return

    yield {
        "type": "sources",
        "chunks": [{"chunk_index": c["chunk_index"], "text": c["text"]} for c in chunks],
    }

    prompt = build_prompt(question, chunks)

    t0 = time.monotonic()
    response = _gen_model.generate_content_async(prompt, stream=True)
    async for chunk in await response:
        if chunk.text:
            yield {"type": "token", "text": chunk.text}
    logger.info("Gemini answer took %.2fs", time.monotonic() - t0)

    yield {"type": "done"}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_query.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/query.py tests/test_query.py
git commit -m "feat(query): add retrieval + Gemini answer pipeline with SSE streaming"
```

---

## Task 6: FastAPI App

**Files:**
- Create: `backend/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests in `tests/test_api.py`**

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch


async def test_health():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_query_without_doc_returns_400():
    from backend.main import app, _state
    _state["doc_id"] = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/query", json={"question": "what is this?"})
    assert r.status_code == 400


async def test_ingest_rejects_non_pdf():
    from backend.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
    assert r.status_code == 400


async def test_doc_info_no_doc():
    from backend.main import app, _state
    _state["doc_id"] = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/doc/info")
    assert r.status_code == 200
    assert r.json()["loaded"] is False


async def test_doc_info_with_doc():
    from backend.main import app, _state
    _state["doc_id"] = "abc123"
    _state["chunk_count"] = 42
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/doc/info")
    assert r.status_code == 200
    data = r.json()
    assert data["loaded"] is True
    assert data["chunk_count"] == 42
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_api.py -v
```

Expected: `ImportError` — `backend.main` does not exist yet.

- [ ] **Step 3: Create `backend/main.py`**

```python
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend import ingest, query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="DocRAG")

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"

_state: dict = {"doc_id": None, "chunk_count": 0}


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/doc/info")
async def doc_info() -> dict:
    if not _state["doc_id"]:
        return {"loaded": False}
    return {
        "loaded": True,
        "doc_id": _state["doc_id"],
        "chunk_count": _state["chunk_count"],
        "embedding_dim": 768,
    }


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()

    async def stream():
        async for event in ingest.run_ingest(pdf_bytes):
            if event.get("status") == "done":
                _state["doc_id"] = event["doc_id"]
                _state["chunk_count"] = event["chunk_count"]
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_doc(req: QueryRequest) -> StreamingResponse:
    if not _state["doc_id"]:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    async def stream():
        async for event in query.run_query(req.question, _state["doc_id"]):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_api.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run all tests to confirm nothing is broken**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py tests/test_api.py
git commit -m "feat(api): add FastAPI app with ingest, query, doc info, and health endpoints"
```

---

## Task 7: Frontend

**Files:**
- Create: `frontend/index.html`

No unit tests for this task — verify visually by running the server and using the app.

- [ ] **Step 1: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DocRAG</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0a0a14;
      --sidebar: #11112b;
      --chat-bg: #0d0d22;
      --bubble: #141432;
      --user-bubble: #1e1e42;
      --accent: #818cf8;
      --accent-dim: #4f5899;
      --border: #2a2a4a;
      --text: #e2e8f0;
      --muted: #64748b;
      --input-bg: #11112b;
      --pill-bg: #1e3a5f;
      --radius: 8px;
    }

    html, body { height: 100%; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; font-size: 14px; }

    .app { display: flex; height: 100vh; overflow: hidden; }

    /* ── Sidebar ── */
    .sidebar {
      width: 300px; min-width: 260px; max-width: 340px;
      background: var(--sidebar);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column; gap: 16px;
      padding: 20px 16px;
      overflow-y: auto;
    }

    .logo {
      font-size: 13px; font-weight: 700; letter-spacing: 2px;
      color: var(--accent); padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }

    .upload-zone {
      border: 1.5px dashed var(--border);
      border-radius: var(--radius);
      padding: 28px 16px;
      text-align: center;
      color: var(--muted);
      cursor: pointer;
      transition: border-color .2s, background .2s;
      line-height: 1.6;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(129,140,248,.05);
      color: var(--accent);
    }
    .upload-zone .icon { font-size: 28px; display: block; margin-bottom: 8px; }

    .progress-area { display: none; flex-direction: column; gap: 8px; }
    .progress-area.visible { display: flex; }
    .progress-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
    .progress-fill { height: 100%; background: var(--accent); border-radius: 2px; width: 0%; transition: width .3s; }
    .progress-msg { font-size: 12px; color: var(--muted); }

    .doc-card { display: none; background: #1a1a3e; border-radius: var(--radius); padding: 12px; }
    .doc-card.visible { display: block; }
    .doc-card .label { font-size: 10px; letter-spacing: 1.5px; color: var(--accent); margin-bottom: 6px; }
    .doc-card .doc-name { font-weight: 600; font-size: 13px; margin-bottom: 4px; word-break: break-all; }
    .doc-card .doc-meta { font-size: 11px; color: var(--muted); margin-bottom: 10px; }
    .clear-btn {
      font-size: 11px; color: var(--muted); background: none;
      border: 1px solid var(--border); border-radius: 4px;
      padding: 4px 10px; cursor: pointer; width: 100%;
      transition: color .2s, border-color .2s;
    }
    .clear-btn:hover { color: #f87171; border-color: #f87171; }

    .sidebar-error { font-size: 12px; color: #f87171; background: rgba(248,113,113,.1); border-radius: 4px; padding: 8px; display: none; }
    .sidebar-error.visible { display: block; }

    /* ── Chat panel ── */
    .chat { flex: 1; display: flex; flex-direction: column; background: var(--chat-bg); min-width: 0; }

    .messages {
      flex: 1; overflow-y: auto; padding: 24px 20px; display: flex;
      flex-direction: column; gap: 16px;
    }
    .messages:empty::after {
      content: "Upload a PDF, then ask a question.";
      color: var(--muted); font-size: 13px;
      display: block; text-align: center; margin-top: 20vh;
    }

    .msg { display: flex; flex-direction: column; max-width: 75%; }
    .msg.user { align-self: flex-end; align-items: flex-end; }
    .msg.assistant { align-self: flex-start; align-items: flex-start; }

    .bubble {
      padding: 10px 14px; border-radius: var(--radius); line-height: 1.6;
      white-space: pre-wrap; word-break: break-word;
    }
    .msg.user .bubble { background: var(--user-bubble); color: var(--muted); }
    .msg.assistant .bubble {
      background: var(--bubble);
      border-left: 3px solid var(--accent);
    }

    .citations { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
    .pill {
      background: var(--pill-bg); color: var(--accent);
      border-radius: 20px; padding: 2px 10px; font-size: 11px;
      cursor: pointer; position: relative; transition: background .2s;
      user-select: none;
    }
    .pill:hover { background: #2a4a6f; }
    .tooltip {
      display: none; position: absolute; bottom: calc(100% + 6px); left: 50%;
      transform: translateX(-50%);
      background: #1e1e42; border: 1px solid var(--border);
      border-radius: 6px; padding: 10px 12px;
      font-size: 11px; color: var(--text); line-height: 1.5;
      width: 280px; white-space: normal; z-index: 10;
      box-shadow: 0 8px 24px rgba(0,0,0,.5);
    }
    .pill:hover .tooltip { display: block; }

    .error-bubble {
      align-self: flex-start;
      background: rgba(248,113,113,.1);
      border-left: 3px solid #f87171;
      color: #f87171;
      padding: 10px 14px;
      border-radius: var(--radius);
      font-size: 13px;
    }

    /* ── Input bar ── */
    .input-bar {
      display: flex; gap: 8px; padding: 16px 20px;
      border-top: 1px solid var(--border); background: var(--chat-bg);
    }
    .input-bar input {
      flex: 1; background: var(--input-bg); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 10px 14px;
      color: var(--text); font-size: 14px; outline: none;
      transition: border-color .2s;
    }
    .input-bar input:focus { border-color: var(--accent); }
    .input-bar input::placeholder { color: var(--muted); }
    .input-bar input:disabled { opacity: .5; cursor: not-allowed; }
    .send-btn {
      background: var(--accent); color: #0a0a14;
      border: none; border-radius: var(--radius); padding: 10px 18px;
      font-size: 16px; font-weight: 700; cursor: pointer;
      transition: background .2s; flex-shrink: 0;
    }
    .send-btn:hover:not(:disabled) { background: #a5b4fc; }
    .send-btn:disabled { opacity: .4; cursor: not-allowed; }

    .spinner {
      display: inline-block; width: 14px; height: 14px;
      border: 2px solid var(--border); border-top-color: var(--accent);
      border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div class="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="logo">DOCRAG</div>

    <div class="upload-zone" id="upload-zone">
      <input type="file" id="file-input" accept=".pdf" hidden>
      <span class="icon">📄</span>
      <strong>Drop PDF here</strong><br>or click to upload
    </div>

    <div class="progress-area" id="progress-area">
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
      <div class="progress-msg" id="progress-msg">Starting…</div>
    </div>

    <div class="doc-card" id="doc-card">
      <div class="label">LOADED DOCUMENT</div>
      <div class="doc-name" id="doc-name"></div>
      <div class="doc-meta" id="doc-meta"></div>
      <button class="clear-btn" id="clear-btn">Clear document</button>
    </div>

    <div class="sidebar-error" id="sidebar-error"></div>
  </aside>

  <!-- Chat -->
  <main class="chat">
    <div class="messages" id="messages"></div>
    <div class="input-bar">
      <input type="text" id="query-input" placeholder="Upload a PDF first" disabled>
      <button class="send-btn" id="send-btn" disabled>→</button>
    </div>
  </main>
</div>

<script>
  const $ = id => document.getElementById(id);

  // ── State ──────────────────────────────────────────────
  let currentDoc = null;      // { name, chunkCount }
  let queryActive = false;

  // ── SSE reader (works with POST responses) ─────────────
  async function readSSE(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw) { try { onEvent(JSON.parse(raw)); } catch (_) {} }
        }
      }
    }
  }

  // ── Upload ─────────────────────────────────────────────
  const uploadZone = $('upload-zone');
  const fileInput  = $('file-input');

  uploadZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) handleUpload(e.target.files[0]); });
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.pdf')) handleUpload(file);
    else showSidebarError('Please drop a PDF file.');
  });

  async function handleUpload(file) {
    hideSidebarError();
    showProgress(0, 'Starting…');
    setUploadZoneVisible(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/ingest', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      await readSSE(res, event => {
        if (event.status === 'done') {
          currentDoc = { name: file.name, chunkCount: event.chunk_count, docId: event.doc_id };
          showDocCard(file.name, event.chunk_count);
          hideProgress();
          enableInput();
        } else {
          const pct = event.progress ? Math.round(event.progress * 100) : null;
          showProgress(pct, event.message);
        }
      });
    } catch (err) {
      hideProgress();
      setUploadZoneVisible(true);
      showSidebarError(err.message);
    }
  }

  // ── Query ──────────────────────────────────────────────
  const queryInput = $('query-input');
  const sendBtn    = $('send-btn');

  sendBtn.addEventListener('click', sendQuery);
  queryInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) sendQuery(); });

  async function sendQuery() {
    const q = queryInput.value.trim();
    if (!q || queryActive || !currentDoc) return;

    queryActive = true;
    queryInput.disabled = true;
    sendBtn.disabled = true;
    queryInput.value = '';

    appendUserMessage(q);
    const assistantBubble = appendAssistantMessage();
    let accumulated = '';
    let sources = [];

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Query failed');
      }

      await readSSE(res, event => {
        if (event.type === 'sources') {
          sources = event.chunks;
        } else if (event.type === 'token') {
          accumulated += event.text;
          assistantBubble.textContent = accumulated;
        } else if (event.type === 'done') {
          renderFinalAnswer(assistantBubble, accumulated, sources);
        }
      });
    } catch (err) {
      assistantBubble.closest('.msg')?.remove();
      appendErrorBubble(err.message);
    } finally {
      queryActive = false;
      queryInput.disabled = false;
      sendBtn.disabled = false;
      queryInput.focus();
      scrollToBottom();
    }
  }

  // ── Message rendering ──────────────────────────────────
  function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg user';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    msg.appendChild(bubble);
    $('messages').appendChild(msg);
    scrollToBottom();
    return bubble;
  }

  function appendAssistantMessage() {
    const msg = document.createElement('div');
    msg.className = 'msg assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = '<span class="spinner"></span>';
    msg.appendChild(bubble);
    $('messages').appendChild(msg);
    scrollToBottom();
    return bubble;
  }

  function renderFinalAnswer(bubble, text, chunks) {
    // Replace [Chunk N] with pill badges
    const escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    const withPills = escaped.replace(/\[Chunk (\d+)\]/g, (_, n) => {
      const chunk = chunks.find(c => c.chunk_index === parseInt(n));
      const tipText = chunk ? chunk.text.replace(/"/g, '&quot;') : '';
      return `<span class="pill">§ Chunk ${n}<span class="tooltip">${tipText}</span></span>`;
    });
    bubble.innerHTML = withPills;

    if (chunks.length > 0) {
      const cites = document.createElement('div');
      cites.className = 'citations';
      const seen = new Set();
      chunks.forEach(c => {
        if (seen.has(c.chunk_index)) return;
        seen.add(c.chunk_index);
        const pill = document.createElement('span');
        pill.className = 'pill';
        pill.innerHTML = `§ Chunk ${c.chunk_index}<span class="tooltip">${escapeHtml(c.text)}</span>`;
        cites.appendChild(pill);
      });
      bubble.closest('.msg').appendChild(cites);
    }
    scrollToBottom();
  }

  function appendErrorBubble(msg) {
    const el = document.createElement('div');
    el.className = 'error-bubble';
    el.textContent = `Error: ${msg}`;
    $('messages').appendChild(el);
    scrollToBottom();
  }

  function scrollToBottom() {
    const m = $('messages');
    m.scrollTop = m.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // ── Sidebar helpers ────────────────────────────────────
  function showProgress(pct, msg) {
    $('progress-area').classList.add('visible');
    $('progress-msg').textContent = msg || '';
    if (pct !== null) $('progress-fill').style.width = pct + '%';
  }
  function hideProgress() { $('progress-area').classList.remove('visible'); }
  function setUploadZoneVisible(v) { uploadZone.style.display = v ? '' : 'none'; }
  function showDocCard(name, count) {
    $('doc-name').textContent = name;
    $('doc-meta').textContent = `${count} chunks · 768-dim`;
    $('doc-card').classList.add('visible');
  }
  function hideDocCard() { $('doc-card').classList.remove('visible'); }
  function showSidebarError(msg) {
    const el = $('sidebar-error');
    el.textContent = msg;
    el.classList.add('visible');
  }
  function hideSidebarError() { $('sidebar-error').classList.remove('visible'); }
  function enableInput() {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.placeholder = 'Ask a question…';
    queryInput.focus();
  }

  // ── Clear ──────────────────────────────────────────────
  $('clear-btn').addEventListener('click', () => {
    currentDoc = null;
    hideDocCard();
    setUploadZoneVisible(true);
    hideSidebarError();
    fileInput.value = '';
    queryInput.disabled = true;
    sendBtn.disabled = true;
    queryInput.placeholder = 'Upload a PDF first';
    $('messages').innerHTML = '';
  });

  // ── Init ───────────────────────────────────────────────
  async function init() {
    try {
      const res = await fetch('/doc/info');
      const info = await res.json();
      if (info.loaded) {
        currentDoc = { docId: info.doc_id, chunkCount: info.chunk_count };
        showDocCard(info.doc_id, info.chunk_count);
        setUploadZoneVisible(false);
        enableInput();
      }
    } catch (_) { /* server not ready yet — leave default state */ }
  }

  init();
</script>
</body>
</html>
```

- [ ] **Step 2: Start the server and verify visually**

```bash
uvicorn backend.main:app --reload
```

Open http://localhost:8000 — verify:
- Dark indigo split-pane layout loads
- Sidebar shows upload zone
- Chat panel input is disabled with "Upload a PDF first" placeholder
- Drag a real PDF onto the upload zone and confirm progress bar animates
- After ingest completes, confirm doc card appears and input is enabled
- Ask a question and confirm streaming answer with `§ Chunk N` pills
- Hover over a pill and confirm raw chunk text tooltip appears
- Click "Clear document" and confirm the UI resets

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): add dark-indigo split-pane chat UI with SSE streaming and citations"
```

---

## Task 8: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update `README.md`**

```markdown
# DocRAG

Single-document RAG system: upload a PDF, ask questions, get grounded answers with citations.

## Stack

| Layer | Technology |
|-------|-----------|
| PDF extraction | Gemini 1.5 Flash |
| Embeddings | `gemini-embedding-004` (768-dim) |
| Answering | Gemini 1.5 Flash (streamed) |
| Vector store | Postgres 16 + pgvector (HNSW) |
| Backend | FastAPI + asyncpg |
| Frontend | Single-file HTML/CSS/JS |

## Setup

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose
- A Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY
```

### 3. Start the database

```bash
docker compose up -d
```

Wait ~10 seconds for the health check to pass: `docker compose ps` should show `db` as `healthy`.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Run

```bash
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

## Usage

1. Drag a PDF onto the sidebar or click to upload
2. Wait for the ingestion progress bar to complete
3. Type a question and press Enter or click →
4. The answer streams back with `§ Chunk N` citation pills — hover for raw text

If the document doesn't contain enough information to answer, the system responds: *"I don't know."*

## Running tests

> Requires the Docker Compose database to be running.

```bash
pytest -v
```

## Design decisions

- **Gemini for extraction**: handles complex PDF layouts including images without local dependencies
- **`gemini-embedding-004`**: 768-dim embeddings, same provider as answering — consistent similarity space
- **pgvector HNSW index**: fast approximate nearest-neighbour search, good default for demo scale
- **512-token chunks with 64-token overlap**: balances retrieval precision with context continuity
- **Single HTML file**: no build tools or Node.js required — serves directly from FastAPI
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, usage, and design decisions"
```

---

## Final Verification

- [ ] Run full test suite: `pytest -v` — all tests pass
- [ ] Cold start: `docker compose down -v && docker compose up -d`, restart server, upload a PDF, ask a question
- [ ] Confirm "I don't know" response: ask a question completely unrelated to the uploaded PDF
- [ ] Confirm citation pills: hover each pill and verify the tooltip shows the correct chunk text
