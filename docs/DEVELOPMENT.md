<!-- generated-by: gsd-doc-writer -->

# Development

This is the working developer's guide to DocRAG: where things live, how the
day-to-day loop works, and the exact files to touch when you want to add an
endpoint, change retrieval, change chunking, add an env var, or add a `make`
target. Generic Python advice is intentionally omitted — every section points
at a specific file, line range, or shell command in this repository.

For first-time setup (prerequisites, install, first run) see `README.md`. For
the system-level picture see `docs/ARCHITECTURE.md`. For env-var and tuning
knobs see `docs/CONFIGURATION.md`.

## Directory Structure

```
.
├── Makefile               # dev loop entrypoints (setup / db / dev / test / logs / reset / clean)
├── docker-compose.yml     # Postgres 16 + pgvector container (image: pgvector/pgvector:pg16)
├── requirements.txt       # pinned Python deps (FastAPI, asyncpg, google-genai, pymupdf, tiktoken, pytest)
├── pytest.ini             # asyncio_mode = auto — every async test runs without a decorator
├── .env.example           # canonical env vars (GEMINI_API_KEY, DATABASE_URL)
├── .env                   # local secrets, gitignored
├── backend/
│   ├── __init__.py
│   ├── config.py          # _Settings — loads .env, sys.exits if a required var is missing
│   ├── main.py            # FastAPI app, all HTTP routes, lifespan hook, in-memory _state
│   ├── ingest.py          # run_ingest() async generator — extract / OCR / caption / chunk / embed / insert
│   ├── chunks.py          # chunk_text() — token-aware sentence/newline splitter with overlap
│   ├── query.py           # run_query() async generator — rewrite / embed / search / stream answer
│   └── db.py              # asyncpg pool, SCHEMA + MIGRATION DDL, _HYBRID_SQL retrieval, all CRUD
├── frontend/
│   ├── index.html         # chat + upload UI (single file, vanilla JS)
│   ├── viewer.html        # PDF viewer (served at /viewer)
│   └── image-viewer.html  # image-chunk preview (served at /image-viewer)
├── tests/
│   ├── conftest.py        # sets test env vars before backend import; provides `pool` fixture
│   ├── test_api.py        # FastAPI route tests via httpx
│   ├── test_chunks.py     # chunk_text() unit tests
│   ├── test_db.py         # asyncpg integration tests against the local Postgres
│   ├── test_ingest.py     # run_ingest() with Gemini calls mocked
│   ├── test_integration_ocr.py
│   └── test_query.py      # run_query() with Gemini calls mocked
├── docs/                  # ARCHITECTURE.md, CONFIGURATION.md, DEVELOPMENT.md (this file), …
└── logs/                  # app.log written by RotatingFileHandler in backend/main.py
```

The `backend/` package is intentionally flat. There is no `routes/`, `services/`,
or `models/` layer — `main.py` holds every HTTP handler and delegates to two
async generators (`ingest.run_ingest`, `query.run_query`). New code should
follow that shape rather than introduce new sub-packages.

## Dev Loop

The `Makefile` is the canonical dev interface — every target is one or two
lines, so you can also run the underlying command directly if you prefer.

| Target | Command it runs | Use it for |
| --- | --- | --- |
| `make setup` | `cp -n .env.example .env` then `pip install -r requirements.txt` | One-time install. Preserves an existing `.env`. |
| `make db` | `docker compose up -d` | Start the `pgvector/pgvector:pg16` container in the background. |
| `make stop` | `docker compose down` | Stop the Postgres container (keeps the `pgdata` volume). |
| `make dev` | `uvicorn backend.main:app --reload` | Run the API with hot-reload on `backend/**` changes. Defaults to `http://127.0.0.1:8000`. |
| `make test` | `pytest` | Full test suite. Some tests require Postgres — run `make db` first. |
| `make logs` | `tail -f logs/app.log` | Tail the app log. Useful in a second terminal while `make dev` runs. |
| `make clean` | Removes `__pycache__/`, `.pytest_cache/`, `logs/` | Wipe Python caches and logs (does not touch the DB). |
| `make reset` | `docker compose down` then `docker compose up -d` | Restart the DB container. Note: this does **not** delete the `pgdata` volume. |

### Recommended terminal layout

1. **Terminal A** — `make db` (once), then `make dev`. Uvicorn auto-reloads on
   any save inside `backend/`. Static frontend files in `frontend/` are served
   by `FileResponse` and reload on hard refresh in the browser; `viewer.html`
   and `image-viewer.html` are served with `Cache-Control: no-cache,
   no-store, must-revalidate` (`backend/main.py:74-78`, `116-121`) so you can
   iterate on them without bumping a version.
2. **Terminal B** — `make logs`. Logs are written by a `RotatingFileHandler`
   (5 MB × 3 backups) configured in `backend/main.py:14-27`. The same
   formatter is wired to a `StreamHandler`, so anything you see in Terminal A
   is also in `logs/app.log`.
3. **Terminal C** — `make test` while you iterate.

### Truly nuking the DB

`make reset` only restarts the container. If you actually want to drop all
data (or the schema is broken), remove the volume:

```bash
docker compose down -v   # the -v flag deletes the pgdata volume
make db                  # bring it back up empty
```

The schema is recreated on the next request via `db.get_pool()`, which executes
both `SCHEMA` and `MIGRATION` inside the pool-init lock (`backend/db.py:74-88`).
You can also clear just the chunks (preserve the volume) with the
`DELETE /doc` endpoint or by calling `db.clear_all_chunks(pool)` directly.

## Where to Add a New Endpoint

All HTTP handlers live in `backend/main.py`. The pattern is:

1. Define a `pydantic.BaseModel` for any non-trivial JSON request body
   (see `QueryRequest` at `backend/main.py:226-228`).
2. Decorate an `async def` handler with `@app.get|post|put|delete(...)`.
3. If the handler needs DB access, call `pool = await db.get_pool()` at the top
   of the function. The pool is module-level cached behind an `asyncio.Lock`
   in `backend/db.py:4-5,74-88` — you do **not** need to manage it yourself.
4. If the handler needs current-document state, read from the module-level
   `_state` dict at `backend/main.py:35`. It holds `doc_id`, `chunk_count`,
   and the adaptive top-`k`. The first thing every doc-specific handler does
   is guard `if not _state["doc_id"]: raise HTTPException(...)` —
   see `get_chunk` (`main.py:96-103`) for the canonical shape.
5. For long-running operations that emit progress, return a `StreamingResponse`
   over `text/event-stream` and yield `f"data: {json.dumps(event)}\n\n"`. The
   two existing examples are `/ingest` (`main.py:176-212`) and `/query`
   (`main.py:231-248`). Catch exceptions inside the inner `stream()` so a
   crash mid-stream still emits a single `error` event to the client instead of
   tearing down the connection silently.
6. The `HTTPException` handler at `main.py:61-66` converts every raised
   `HTTPException` into a JSON `{"error": detail}` body and logs the
   `method path → status detail` line. Prefer raising `HTTPException` over
   returning custom error JSON so this single chokepoint stays effective.

The full route list as of this writing (from `backend/main.py`):

```
GET    /                       — serve frontend/index.html
GET    /viewer                 — serve frontend/viewer.html
GET    /image-viewer           — serve frontend/image-viewer.html
GET    /health                 — {"status": "ok"}
GET    /doc/info               — current doc metadata or {"loaded": false}
GET    /doc/pdf                — the original uploaded PDF bytes
GET    /doc/chunk/{chunk_index}
GET    /doc/images             — list image chunks for fuzzy-link rendering
GET    /image/{chunk_index}
GET    /image/{chunk_index}/meta
POST   /ingest                 — multipart upload, SSE stream of progress events
POST   /query                  — JSON body, SSE stream of token + sources events
DELETE /doc                    — TRUNCATE the chunks table and reset _state
```

If your endpoint needs new DB queries, add the SQL constant + `async def`
helper to `backend/db.py` (next section); if it needs new ingest or retrieval
logic, edit the corresponding generator (sections after that).

## Where to Change Retrieval

Retrieval is split across two files:

- **`backend/db.py`** — the SQL. Two query plans live here:
  - `_HYBRID_SQL` (`db.py:129-166`) — the production path. A single CTE that
    runs dense pgvector cosine search and sparse `to_tsvector('english')` BM25
    in parallel, fuses them with Reciprocal Rank Fusion (`1 / (60 + rank)`),
    and returns the top-`k`. The `60` RRF constant is hard-coded — change it
    here if you want different fusion weighting. Each side fetches `LIMIT $3 * 2`
    (i.e. `2k`) candidates before the join, which is also the place to widen
    or narrow the candidate pool.
  - `_DENSE_SQL` (`db.py:168-174`) — fallback for empty-string queries.
    `search_chunks` (`db.py:177-190`) routes to hybrid when
    `question.strip()` is truthy, dense-only otherwise.
- **`backend/query.py`** — the orchestration around it. `run_query` (`query.py:155-227`)
  is an async generator that:
  1. Optionally rewrites the question to be self-contained when the user
     asks a follow-up with pronouns (`_PRONOUN_RE` at `query.py:74-76`,
     `_rewrite_query` at `79-100`).
  2. Embeds the (possibly rewritten) question with `gemini-embedding-2` at
     768 dims (`_embed_query`, `query.py:53-71`) — this dimension is fixed
     by the `vector(768)` column type in `db.py:20`.
  3. Calls `db.search_chunks(...)`.
  4. Emits a `sources` SSE event with the retrieved chunks.
  5. Builds a multimodal Gemini prompt that interleaves text chunks and
     image bytes (`_build_multimodal_contents`, `query.py:125-152`) and
     streams the answer tokens.

To change retrieval behavior:

| Want to change | Edit |
| --- | --- |
| The hybrid SQL (RRF constant, candidate pool size, dense/sparse weighting) | `_HYBRID_SQL` in `backend/db.py:129-166` |
| Whether queries run hybrid or dense-only | `search_chunks` routing in `backend/db.py:177-190` |
| The query embedding model or dimensionality | `_EMBED_MODEL` / `_EMBED_DIM` in `backend/query.py:18-19` (must match `vector(N)` in `db.py:20`) |
| The retry / backoff policy for embed and Gemini calls | `_MAX_RETRIES`, `_RETRY_BASE`, `_is_retryable` in `backend/query.py:37-50` |
| Pronoun-rewrite trigger or prompt | `_PRONOUN_RE` and `_rewrite_query` in `backend/query.py:74-100` |
| How chunks are presented to the LLM (prompt template, image interleaving) | `SYSTEM_PROMPT` (`query.py:21-35`), `build_prompt` / `_build_multimodal_contents` (`query.py:116-152`) |
| The default top-`k` when the doc has no stored value | The `8` literals in `backend/main.py:35,46,205,221` (and `backend/ingest.py` `compute_params`, see next section) |

The adaptive `k` is set at ingest time, not query time — it's stored in
`doc_meta.k` and read into `_state["k"]` at lifespan startup
(`main.py:38-55`) and after each successful ingest (`main.py:202-205`).

## Where to Change Chunking

`backend/chunks.py` holds the entire chunker — 48 lines, one function:

```python
def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
```

It splits on `(?<=[.!?])\s+|\n+` (sentence boundaries **or** raw newlines —
the newline branch is the trick that keeps OCR'd handwriting and
unpunctuated bullet lists from collapsing into one giant chunk;
`chunks.py:11-15`), greedily fills a buffer until adding the next sentence
would exceed `max_tokens`, then carries trailing sentences forward until the
overlap budget is exhausted. Token counting uses the `cl100k_base` tiktoken
encoding (`chunks.py:4`).

To change chunking:

- **The split regex** — edit `chunks.py:15`. If you add new boundary types
  (e.g. semicolons, headings) keep both look-behind alternation and the raw
  newline branch.
- **`max_tokens` / `overlap` defaults** — these defaults are only used when a
  caller doesn't pass values. The actual production values are computed by
  `compute_params` in `backend/ingest.py:161-172`, which also sets the
  retrieval top-`k` based on document size:

  | Doc size (tokens) | chunk_size | k  |
  | --- | --- | --- |
  | < 10,000        | 256 | 5  |
  | 10,000–50,000   | 384 | 8  |
  | 50,000–200,000  | 512 | 12 |
  | 200,000–500,000 | 768 | 15 |
  | ≥ 500,000       | 1024 | 20 |

  Tune the breakpoints there if you want different behavior for a particular
  document size class.
- **Adding a new chunk type** (text and image are the only two today) — add
  the new `chunk_type` value, then plumb it through `db.py` (the column
  default is `'text'` at `db.py:17`; the SQL CTEs select `chunk_type` and
  `query.py` already branches on it at `query.py:138`).

Tests for the chunker live in `tests/test_chunks.py` and run without
Postgres, so they're the fastest feedback loop when you change `chunks.py`.

## How to Add a New Env Var

Two files, in this order:

1. **`backend/config.py`** — add a class-level annotation and a matching
   `os.getenv(...)` line in `__init__`:

   ```python
   class _Settings:
       gemini_api_key: str
       database_url: str
       my_new_var: str               # 1. add the annotation

       def __init__(self) -> None:
           self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
           self.database_url = os.getenv("DATABASE_URL", "")
           self.my_new_var = os.getenv("MY_NEW_VAR", "")  # 2. read it
           ...
   ```

   The required-var check at `config.py:15-23` is automatic: it iterates
   `__class__.__annotations__` and `sys.exit(1)`s if any annotated attribute
   is empty after init. **A var is required iff it has a class annotation.**
   Optional vars should be assigned in `__init__` without an annotation, e.g.
   `self.optional_thing = os.getenv("OPTIONAL_THING") or "default"`.

2. **`.env.example`** — append the new var with a placeholder value so the
   next contributor's `make setup` (which runs `cp -n .env.example .env`) sees
   it. Keep the file two-column-clean — one `KEY=value` per line, no quotes,
   no comments interleaved with values.

3. **`tests/conftest.py`** — if the var is required (annotated), add a
   matching `os.environ.setdefault("MY_NEW_VAR", "test-placeholder")` before
   `from backend.config import settings` so test runs don't `sys.exit`
   (`tests/conftest.py:5-6` is the existing pattern).

4. **`docs/CONFIGURATION.md`** — document the new var in the env-vars table.
   This is part of the same change; CI doc-verification will flag mismatches.

Then read the value via `from backend.config import settings; settings.my_new_var`
anywhere downstream. Do **not** call `os.getenv` directly outside `config.py`
— the centralised `_Settings` class is what makes the required-var startup
check work.

## How to Add a New Make Target

The `Makefile` is the dev surface area — keep targets short, declarative, and
prefix-friendly.

1. **Declare the target as `.PHONY`** at the top. The current declaration is:

   ```makefile
   .PHONY: setup db stop dev test logs clean reset
   ```

   Add your target name to that list. This prevents Make from confusing the
   target with a same-named file on disk.

2. **Add the recipe.** One command per logical step. Existing targets are
   1–2 lines each — match that scale rather than embedding shell logic.

   ```makefile
   shell:
   	psql $$DATABASE_URL
   ```

   (Inside a recipe, double `$$` to emit a literal `$` for the shell.)

3. **Compose with existing targets** when there's a dependency. `make reset`
   does this today (`reset: stop db`) — list prerequisites after the colon
   and Make runs them in order.

4. **Document it.** Update the dev-loop table earlier in this file. Targets
   that aren't in this table get rediscovered as folklore.

Tabs vs spaces matter: Makefile recipes must be indented with a literal `TAB`,
not spaces. If your editor is configured to expand tabs in `Makefile`, the
recipe will fail with `*** missing separator`.

## Tests

`pytest.ini` sets `asyncio_mode = auto` (`pytest.ini:1-2`), so every `async
def` test function is collected and awaited automatically — no per-test
`@pytest.mark.asyncio` decorator needed.

The `pool` fixture in `tests/conftest.py:13-25` opens a real `asyncpg` pool
against `DATABASE_URL`, runs `SCHEMA` + `MIGRATION` to ensure the table is
present, yields the pool to the test, then `TRUNCATE`s `chunks` and
`doc_meta` and closes the pool. So:

- DB-touching tests (`test_db.py`, `test_ingest.py`, `test_integration_ocr.py`,
  `test_api.py`) **require** `make db` to be running first.
- Pure unit tests (`test_chunks.py`) do not need Postgres.
- Gemini calls are always mocked in tests — `conftest.py:5-6` sets a
  placeholder `GEMINI_API_KEY` before `backend.config` is imported.

To run a single test file or test:

```bash
pytest tests/test_chunks.py             # one file
pytest tests/test_query.py::test_name   # one test
pytest -k chunk                         # by keyword
pytest -x                               # stop on first failure
```

There is no `make test:watch` target — install `pytest-watch` (`pip install
pytest-watch && ptw`) locally if you want one, or add a target following the
recipe in the previous section.

## Code Style

There is no committed linter or formatter config (no `pyproject.toml`,
`.ruff.toml`, `setup.cfg` `[flake8]`, `.pre-commit-config.yaml`, or
`.editorconfig` in the repo). Match the existing style by reading
`backend/db.py` and `backend/main.py`:

- Type hints on all function signatures, including `-> None` for
  side-effecting helpers.
- `from __future__ import annotations` is **not** used; the codebase targets
  Python 3.11+ (the `X | None` PEP 604 syntax appears throughout, e.g.
  `backend/db.py:4`, `backend/db.py:195`).
- Module-level constants are SCREAMING_SNAKE_CASE and prefixed with `_` when
  they are not part of the public surface (e.g. `_HYBRID_SQL`,
  `_GEMINI_CONCURRENCY`, `_BLANK_PHRASES` in `backend/ingest.py`).
- Module-level state caches use a leading underscore and are guarded with an
  `asyncio.Lock` when initialised lazily (see `_pool` / `_pool_lock` in
  `backend/db.py:4-5`).
- Docstrings on non-trivial helpers explain *why* (the chunker comment at
  `backend/chunks.py:11-15` and the annotation handler at
  `backend/ingest.py:184-214` are good examples). One-line obvious helpers
  go uncommented.

If you prefer Black/Ruff, run them locally — just don't commit a config file
without discussing the formatting strategy first, since it would reformat
the whole tree.

## Branch Conventions

No convention is documented in this repo (no `CONTRIBUTING.md`, no
`.github/PULL_REQUEST_TEMPLATE.md`, no GitHub Actions workflows under
`.github/workflows/`). The default branch is `main`. Recent commits follow
[Conventional Commits](https://www.conventionalcommits.org/) prefixes (`feat`,
`fix`, `chore`, `docs`, `perf`) — match that style for new commits.

## PR Process

There is no automated PR template or required CI check at the time of writing.
Suggested checklist before opening a PR:

- [ ] `make test` passes locally (with `make db` running).
- [ ] If you changed `backend/config.py`, you also updated `.env.example`,
      `tests/conftest.py`, and `docs/CONFIGURATION.md`.
- [ ] If you changed `_HYBRID_SQL` or chunking, you re-ingested at least
      one PDF locally and ran a representative query.
- [ ] Conventional-commit style commit message.
- [ ] Doc updates included in the same PR (this repo treats `docs/` as
      part of the code, not a separate stream).
