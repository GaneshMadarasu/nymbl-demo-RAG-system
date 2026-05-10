<!-- generated-by: gsd-doc-writer -->
# Getting Started

This guide walks a new developer or evaluator from a fresh clone to asking the
first question against a PDF, in roughly five minutes. For deeper material see
[`README.md`](../README.md), [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), and
[`docs/CONFIGURATION.md`](CONFIGURATION.md).

## Prerequisites

You need three things installed locally before running anything:

- **Python 3.11 or newer** — the FastAPI backend and ingest pipeline run on the
  host Python, not inside the container.
- **Docker** — used to run the Postgres 16 + pgvector container declared in
  `docker-compose.yml`. Docker Desktop or any Docker Engine with the
  `docker compose` plugin works.
- **A Gemini API key** — required for embeddings, vision captioning, and answer
  generation. You can grab one from Google AI Studio.

## Installation steps

1. **Clone the repository and enter it:**
   ```bash
   git clone <repository-url>
   cd nymbl-demo-RAG-system
   ```

2. **Run the one-shot setup target.** This copies `.env.example` to `.env` (if
   one does not exist) and installs Python dependencies from
   `requirements.txt`:
   ```bash
   make setup
   ```

3. **Edit `.env` and set your Gemini API key.** Open the file and replace the
   placeholder:
   ```bash
   GEMINI_API_KEY=your_key_here
   ```
   Leave `DATABASE_URL` unchanged — it is already pointed at the
   docker-compose Postgres
   (`postgresql://rag:rag@localhost:5432/ragdb`). See
   [`docs/CONFIGURATION.md`](CONFIGURATION.md) for the full list of supported
   variables.

4. **Start Postgres + pgvector** in the background:
   ```bash
   make db
   ```
   This runs `docker compose up -d` against the `pgvector/pgvector:pg16` image
   defined in `docker-compose.yml`.

## First run

Start the FastAPI dev server with auto-reload:

```bash
make dev
```

This runs `uvicorn backend.main:app --reload` and binds to
`http://localhost:8000`. Open that URL in your browser. You should see the
**NYMBL - DOCRAG** sidebar on the left, an upload zone, and an empty chat with
the placeholder *"Upload a PDF, then ask a question."*

### Walk through your first query

1. **Drag a PDF onto the "Drop PDF here" zone**, or click it and pick a file.
   The zone only accepts `.pdf`.
2. **Pick an ingest mode in the modal** that pops up
   (titled *"Process images in this PDF?"*). It exposes four controls:
   - **Text** — text-only ingest. Fastest path. No vision calls.
   - **+Images** — text plus image extraction, Gemini Vision captioning, and
     image-aware retrieval.
   - **Hand Written?** *(checkbox)* — runs Vision OCR on pages whose extracted
     text is sparse. Use for scanned or handwritten PDFs.
   - **Markings?** *(checkbox)* — runs a separate Vision pass to detect
     hand-drawn markup (underlines, highlights, circles, arrows, margin notes)
     and folds the result into page text. An optional **Pages to scan** field
     accepts a range like `1-50, 100, 200-220` to limit the scan; leave it
     blank to scan all pages.
3. **Watch the progress bar** in the sidebar. The server streams stage names
   over Server-Sent Events: `extracting → (ocr) → (markup) → chunking →
   clearing → embedding → (extracting_images → captioning → embedding_images)
   → done`.
4. **Ask your first question** in the chat input at the bottom. Hit Enter or
   click the send arrow. Tokens stream back live.
5. **Click a citation pill** in the answer:
   - Purple pills (e.g. `§ 3`) open the PDF in a side viewer (`/viewer`)
     scrolled to the cited chunk's page.
   - Orange image pills open the captioned figure in the image viewer
     (`/image-viewer`).

That is the full end-to-end flow.

## Common setup issues

- **`GEMINI_API_KEY` is still the placeholder.** The ingest stream will fail at
  the `embedding` stage. Open `.env`, set a real key, and restart `make dev`.
- **Postgres is not running.** Symptoms include connection-refused errors when
  you upload a PDF. Run `make db` and confirm with `docker compose ps` that
  the `db` service is up. The container exposes port `5432` on the host, so
  another local Postgres on `5432` will conflict — stop it or change the
  mapping in `docker-compose.yml`.
- **Wrong Python version.** `pip install -r requirements.txt` may fail or pull
  incompatible wheels on Python < 3.11. Check `python --version` before
  running `make setup`.
- **Docker Compose plugin missing.** `make db` calls `docker compose` (with a
  space, the v2 plugin). If you only have legacy `docker-compose`, install
  Docker Desktop or the `docker-compose-plugin` package.
- **Port 8000 already in use.** Another local service is bound to the uvicorn
  default port. Stop it, or run uvicorn manually on a different port:
  `uvicorn backend.main:app --reload --port 8001`.

## Make targets cheat sheet

| Command       | What it does                                                                  |
| ------------- | ----------------------------------------------------------------------------- |
| `make setup`  | Copy `.env.example` → `.env` (if absent) and `pip install -r requirements.txt` |
| `make db`     | Start the `pgvector/pgvector:pg16` Postgres container via `docker compose`    |
| `make dev`    | Run `uvicorn backend.main:app --reload`                                       |
| `make test`   | Run the pytest suite                                                          |
| `make logs`   | `tail -f logs/app.log`                                                        |
| `make stop`   | `docker compose down` — stop the database                                     |
| `make reset`  | `make stop` followed by `make db` — fresh DB process, same volume             |
| `make clean`  | Remove `__pycache__/`, `.pytest_cache/`, and `logs/`                          |

## Next steps

- **Architecture:** [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — how ingestion,
  hybrid retrieval, and answer generation fit together.
- **Configuration:** [`docs/CONFIGURATION.md`](CONFIGURATION.md) — every
  environment variable, defaults, and per-environment overrides.
- **Project overview and API surface:** [`README.md`](../README.md) —
  endpoints, repository layout, and key design decisions.
