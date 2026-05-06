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
- A Gemini API key

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
