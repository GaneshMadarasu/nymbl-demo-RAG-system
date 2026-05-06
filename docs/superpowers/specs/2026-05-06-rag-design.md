# RAG System Design

**Date:** 2026-05-06  
**Project:** nymbl-demo-RAG-system  
**Status:** Approved

---

## Overview

A single-document RAG system: ingest one PDF, chunk and embed it, store in a vector database, and answer user questions grounded in the document with explicit chunk citations. Delivered as a web app with a dark-indigo split-pane chat UI.

---

## Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| PDF extraction | Gemini 1.5 Flash | No local dependency, handles complex layouts and images |
| Embeddings | `gemini-embedding-004` (768-dim) | Consistent with answering provider, high quality |
| Answering | Gemini 1.5 Flash | Fast streaming, same API key as embeddings |
| Vector store | Postgres 16 + pgvector (HNSW) | Robust, SQL-queryable, easy Docker setup |
| Backend | FastAPI (Python) | Async, streaming SSE support, serves static frontend |
| Frontend | Single HTML/CSS/JS file | No build tools, portable, fully controllable design |
| Infrastructure | Docker Compose | One-command database startup |

---

## Project Structure

```
nymbl-demo-RAG-system/
├── backend/
│   ├── main.py           # FastAPI app — serves frontend + REST endpoints
│   ├── ingest.py         # PDF → Gemini text → chunks → embeddings → pgvector
│   ├── query.py          # Embed query → pgvector top-k → Gemini answer + citations
│   ├── db.py             # pgvector connection, schema init (asyncpg)
│   └── config.py         # Settings loaded from env vars, fails fast if missing
├── frontend/
│   └── index.html        # Dark-indigo split-pane UI, vanilla JS, SSE streaming
├── docker-compose.yml    # Postgres 16 + pgvector, health-checked, volume-mounted
├── .env.example
├── requirements.txt
└── README.md
```

---

## Data Model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,        -- SHA-256 of file content
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

One document at a time. Ingesting a new PDF deletes all rows for the previous `doc_id`.

---

## REST API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve `frontend/index.html` |
| `POST` | `/ingest` | Upload PDF, run full pipeline |
| `GET` | `/ingest/status` | SSE stream of ingest progress events |
| `POST` | `/query` | Ask a question, stream answer via SSE |
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |

---

## Ingestion Pipeline

1. Receive PDF via `POST /ingest` (multipart form)
2. Send PDF bytes to Gemini 1.5 Flash with prompt: `"Extract all text from this document. Preserve paragraph structure."`
3. Split extracted text into chunks: **512 tokens, 64-token overlap**, split on sentence boundaries using regex
4. Embed each chunk with `gemini-embedding-004` → `vector(768)`
5. Batch-insert into `chunks` table (delete previous doc rows first)
6. Progress events streamed via SSE: `"Extracting text…"`, `"Chunking…"`, `"Embedding 47/142…"`, `"Done — 142 chunks stored"`

---

## Query Pipeline

1. Receive question via `POST /query` → embed with `gemini-embedding-004`
2. Cosine similarity search: `SELECT ... ORDER BY embedding <=> $1 LIMIT 5`
3. Build prompt:

```
You are a document Q&A assistant. Answer ONLY using the provided context.
If the context doesn't contain enough information, respond with exactly: "I don't know."
Cite sources as [Chunk N] inline.

Context:
[Chunk 3]: "..."
[Chunk 12]: "..."
...

Question: {user_question}
```

4. Stream Gemini 1.5 Flash response back to client via SSE
5. Client renders tokens as they arrive; parses `[Chunk N]` references into pill badges

---

## Frontend UI

**Layout:** Dark-indigo split pane.

**Sidebar (35%):**
- Drag-and-drop / click-to-upload PDF zone
- Live ingestion progress bar (SSE)
- Doc info card: filename, chunk count, embedding dim
- "Clear document" button

**Chat panel (65%):**
- Message thread: user messages right-aligned (muted), assistant answers left-aligned with indigo left-border
- `[Chunk N]` citations rendered as pill badges; clicking expands a tooltip with raw chunk text
- Streaming: answer types out token-by-token
- Input bar pinned to bottom; `Enter` or send button to submit
- Input disabled with spinner during active query
- "I don't know" displayed as plain answer text

**Error states:**
- Upload failure → inline error in sidebar, drop zone resets
- Query failure → error bubble in chat thread, input re-enabled
- No doc loaded → input placeholder reads "Upload a PDF first", input disabled

---

## Chunking Strategy

- **Size:** 512 tokens (≈ 380 words) — large enough for context, small enough for precise retrieval
- **Overlap:** 64 tokens — preserves continuity across chunk boundaries
- **Boundary detection:** Split on sentence-ending punctuation before exceeding token limit
- **Tokenizer:** `tiktoken` `cl100k_base` for token counting — approximates Gemini's SentencePiece tokenizer, accurate enough for a demo

---

## "I Don't Know" Guarantee

The system prompt explicitly instructs Gemini to return `"I don't know."` verbatim when the retrieved chunks lack sufficient evidence. No post-processing required — the response is streamed and displayed as-is.

---

## Hygiene & DX

**Environment:**
```
GEMINI_API_KEY=your_key_here
DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb
```
`config.py` calls `sys.exit(1)` at startup if either var is missing.

**Run commands:**
```bash
cp .env.example .env          # fill in GEMINI_API_KEY
docker compose up -d          # start postgres + pgvector
pip install -r requirements.txt
uvicorn backend.main:app --reload   # app at http://localhost:8000
```

**Logging:** Python `logging` with structured messages at: ingest start/complete, chunk count, query received, retrieval latency (ms), Gemini call duration (ms). No secrets in log output.

**Error responses:** `{"error": "plain English message"}` to client; full traceback logged server-side only.

---

## Out of Scope

- Multi-document support
- User authentication
- Persistent chat history across sessions
- Reranking (cross-encoder)
- Hybrid search (BM25 + vector)
