<!-- generated-by: gsd-doc-writer -->
# Testing

How to run, write, and reason about the test suite for this project.

## Test framework and setup

- **Framework:** [pytest](https://pytest.org) `8.3.3` with [pytest-asyncio](https://pytest-asyncio.readthedocs.io) `0.24.0` (from `requirements.txt`).
- **Async mode:** `asyncio_mode = auto` is set in `pytest.ini`. Every `async def test_*` is awaited automatically — no per-test `@pytest.mark.asyncio` decorator is required.
- **HTTP client for FastAPI:** [httpx](https://www.python-httpx.org/) `0.27.2` via `AsyncClient` + `ASGITransport` (in-process; no live server needed).
- **Test runner config (`pytest.ini`):**
  ```ini
  [pytest]
  asyncio_mode = auto
  ```

### Prerequisites (must be true before `make test`)

1. **Postgres + pgvector must be running.** The shared `pool` fixture in `tests/conftest.py` opens an `asyncpg` pool against `DATABASE_URL` and runs the schema/migration DDL on it. If the database is not up, every async test that requests `pool` will fail at connection time. Start it with:
   ```bash
   make db          # docker compose up -d (pgvector/pgvector:pg16)
   ```
2. **Python deps installed.** Run `make setup` once to copy `.env.example` to `.env` and `pip install -r requirements.txt`.
3. **`GEMINI_API_KEY` is auto-stubbed.** `tests/conftest.py` sets `GEMINI_API_KEY=test-placeholder` (and `DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb`) via `os.environ.setdefault` *before* importing `backend.config`, so the suite never sys.exits on missing config. All Gemini calls in unit tests are mocked. A real key in `.env` only matters for the gated integration test (see below).

## Running tests

```bash
make test                       # alias for: pytest
pytest                          # full suite from repo root
pytest tests/test_chunks.py     # one file
pytest tests/test_db.py::test_insert_and_search   # one test
pytest -k "ocr"                 # match by name pattern
pytest -m integration           # only the gated real-Vision integration test
pytest -v                       # verbose names
pytest -x                       # stop on first failure
```

There is no watch-mode script wired up; re-run `pytest` after edits.

### Resetting database state between runs

The `pool` fixture truncates `chunks` and `doc_meta` (with identity reset) on teardown, so tests that share the DB do not bleed into each other within a run. If a crashed run leaves stale rows you want gone, recycle the volume:

```bash
make reset       # docker compose down && docker compose up -d
```

## Test layout

All tests live in `tests/` (flat, one file per backend module under test). The package marker file `tests/__init__.py` exists but is empty.

| File | What it covers | Needs DB? |
|---|---|---|
| `tests/conftest.py` | Shared `pool` fixture; sets safe env-var placeholders before backend import. | — |
| `tests/test_chunks.py` | `backend.chunks.chunk_text` — short/long inputs, overlap behavior, empty input, no-empty-chunks invariant, newline-only splits for handwritten OCR. | No |
| `tests/test_db.py` | `backend.db.insert_chunks` / `search_chunks` / `clear_all_chunks` / `get_doc_info` — vector insert + cosine search, hybrid (vector + question text) path, full clear, doc metadata lookup. | **Yes** |
| `tests/test_ingest.py` | `backend.ingest` — `_doc_id` determinism + length, `run_ingest` end-to-end with mocked extractors/embedder, `_extract_text` annotation capture (highlight/underline/sticky/free-text), `_extract_images` `skip_pages` plumbing, `_render_page` PNG output, `_ocr_one_page` + `_ocr_empty_pages` (threshold logic, failure isolation, no-op when no empty pages), `_detect_visual_markup_one_page` (valid JSON, empty array, malformed JSON), `_format_markup_summary`, `_parse_page_ranges` (singles, ranges, clamping, garbage), `_collect_visual_markup` (text-page gating + `target_pages` filter), and the `detect_markup` / `ocr_scanned` toggles on `run_ingest`. | **Yes** (subset using `pool`) |
| `tests/test_query.py` | `backend.query.build_prompt` (system prompt, sequential `[Chunk N]` labels, question echo) and `run_query` (yields `sources` / `token` / `done` events; "I don't know" path when retrieval is empty). | **Yes** |
| `tests/test_api.py` | FastAPI HTTP surface — `GET /health`, `GET /doc/info` (loaded vs unloaded), `POST /query` 400 with no doc loaded, `POST /ingest` rejects non-PDF MIME and forwards the `ocr_scanned` form field through to `ingest.run_ingest`. | No (Gemini + ingest mocked; no `pool`) |
| `tests/test_integration_ocr.py` | Real Gemini Vision OCR against a synthetic 2-page PDF (rendered-text page 1 + embedded-text page 2). Marked `pytest.mark.integration` and skipped unless `GEMINI_API_KEY` is set to a real key. | **Yes** |

## Fixtures

There is exactly one shared fixture, defined in `tests/conftest.py`:

### `pool` (async)

- Creates an `asyncpg` pool against `settings.database_url`.
- Acquires a connection and runs `backend.db.SCHEMA` + `backend.db.MIGRATION` (idempotent — safe to re-run).
- Yields the pool to the test.
- On teardown, runs `TRUNCATE TABLE chunks RESTART IDENTITY` and `TRUNCATE TABLE doc_meta`, then closes the pool.

Tests that touch the database simply add `pool` to the signature: `async def test_x(pool): ...`. Tests that don't (e.g. `test_chunks.py`, most of `test_api.py`) omit it and run without a DB connection.

## Writing new tests

### Naming and discovery

- File names follow `tests/test_<module>.py`, mirroring the backend module under test (`backend/db.py` → `tests/test_db.py`).
- Test functions are named `test_<behavior_in_words>` — descriptive sentences, not `test_1`. Examples from the suite: `test_short_text_produces_one_chunk`, `test_run_ingest_ocr_on_with_empty_pages_calls_vision`, `test_query_without_doc_returns_400`.
- pytest's default discovery is used (no custom `testpaths` or `python_files` overrides), so anything matching `test_*.py` under `tests/` is picked up automatically.

### Async tests

Just write `async def`. `asyncio_mode = auto` handles the rest:

```python
async def test_insert_and_search(pool):
    await insert_chunks(pool, "doc1", [(0, "hello", None, [0.1] * 768, None)])
    results = await search_chunks(pool, "doc1", [0.1] * 768, k=1)
    assert results[0]["text"] == "hello"
```

### Mocking external services

The suite mocks Gemini at the SDK boundary using `unittest.mock.patch` and `AsyncMock`. Common patterns from `tests/test_ingest.py` and `tests/test_query.py`:

- **Embeddings:** `patch("backend.ingest._embed_one", new_callable=AsyncMock, return_value=[0.1] * 768)`
- **PDF extraction:** `patch("backend.ingest._extract_text", return_value=(fake_text, fake_pages))`
- **Pool injection:** `patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool)` so the production code reuses the test's truncatable pool.
- **Streaming generation:** patch `_client.aio.models.generate_content_stream` with a `side_effect` that returns an async iterator yielding `MagicMock(text="...")` chunks.
- **Vision / OCR responses:** wrap a fake `text` in `types.SimpleNamespace(text=...)` and patch `_client.aio.models.generate_content`.

### Building real PDFs in tests

Several `test_ingest.py` tests build PDFs in-memory with `pymupdf` (`import fitz`) — `doc.new_page()`, `page.insert_text(...)`, `page.add_highlight_annot(...)`, `doc.write()` — to exercise `_extract_text`, `_extract_images`, and `_render_page` against the real library rather than mocks. Prefer this when the behavior under test is "what does pymupdf give us"; prefer mocks when the behavior is "what does our code do given an extractor result".

### Testing the FastAPI surface

`test_api.py` runs the app in-process. There is no live server:

```python
from httpx import AsyncClient, ASGITransport
from backend.main import app

async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
```

For endpoints that read `_state` (the in-memory "currently loaded doc" dict in `backend.main`), set the relevant keys directly before issuing the request — see `test_doc_info_with_doc` and `test_query_without_doc_returns_400`.

## Coverage

**No coverage threshold is configured.** There is no `coverage`, `pytest-cov`, `.coveragerc`, or `.nycrc` in the repo, and `pytest.ini` only sets `asyncio_mode`. To collect coverage ad-hoc:

```bash
pip install pytest-cov
pytest --cov=backend --cov-report=term-missing
```

## CI integration

**No CI pipeline is configured in this repo.** There is no `.github/workflows/` directory, no `.gitlab-ci.yml`, and no other CI config file. Tests run only when a developer runs `make test` (or `pytest`) locally. The integration test in `test_integration_ocr.py` is gated on a real `GEMINI_API_KEY` and self-skips otherwise; its docstring notes "Skip in CI by default" — wiring a CI job would inherit that gating for free.

## Markers

Only one custom marker is in use:

| Marker | Where | Behavior |
|---|---|---|
| `integration` | `tests/test_integration_ocr.py` (module-level `pytestmark`) | Skipped unless `GEMINI_API_KEY` is set to a non-placeholder value. Run explicitly with `pytest -m integration`. |

Note: `integration` is not registered in `pytest.ini`, so pytest will print a `PytestUnknownMarkWarning` when the suite runs. Adding `markers = integration: ...` under `[pytest]` would silence it.

## Coverage gaps and known limitations

Surfaces that the current suite does *not* exercise:

- **Streaming over SSE:** `POST /ingest` and `POST /query` stream NDJSON / SSE events to the client in production, but `test_api.py` does not consume the streamed body — it only asserts on status codes and form-field plumbing. The streaming protocol is exercised indirectly through `run_ingest` / `run_query` event-list tests.
- **`backend.config`:** No direct tests. The module's `sys.exit` on missing `GEMINI_API_KEY` is bypassed in tests via the `os.environ.setdefault` calls in `conftest.py`.
- **Frontend:** `frontend/` is a single-file HTML/CSS/JS bundle with no JS tests in the repo.
- **Real Postgres failure modes:** Connection drops, transaction rollback, and pgvector index degradation paths are not simulated.
- **End-to-end `/ingest` → `/query`:** No test uploads a real PDF through the HTTP layer and then queries it. The closest equivalent is the gated `test_integration_ocr.py`, which calls `run_ingest` directly rather than through `POST /ingest`.

## Next steps

- See [`GETTING-STARTED.md`](GETTING-STARTED.md) for first-time setup (Python install, `make db`, `make setup`).
- See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the local dev loop and the full list of `make` targets.
- See [`CONFIGURATION.md`](CONFIGURATION.md) for `DATABASE_URL` and `GEMINI_API_KEY` semantics.
