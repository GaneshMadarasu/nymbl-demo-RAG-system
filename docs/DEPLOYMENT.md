<!-- generated-by: gsd-doc-writer -->

# Deployment

DocRAG is shipped as a **demo / local-development project**. The repository
contains everything needed to run the full stack on a developer laptop, but it
does **not** contain a production Dockerfile for the FastAPI app, a CI/CD
pipeline, or any cloud-platform deploy config. This document describes the
present deployment reality first, then sketches what a production deployment
would require so an operator can plan one.

## Deployment Targets

### Currently supported: local / demo

The repository ships with a single deployment target — a developer machine.
Two pieces are run side by side:

| Component | How it runs | Config file |
| --- | --- | --- |
| Postgres 16 + `pgvector` | `docker compose up -d` (`make db`) | `docker-compose.yml` |
| FastAPI app (Uvicorn) | `uvicorn backend.main:app --reload` (`make dev`) | `Makefile`, `requirements.txt` |
| Frontend | Static files served by FastAPI from `frontend/index.html` | `backend/main.py:69-72` |

The `docker-compose.yml` provisions only the database service:

```yaml
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
```

Schema and indexes (`chunks`, `doc_meta`, the HNSW vector index, the GIN tsv
index) are created on first connection by `backend/db.py`'s `SCHEMA` /
`MIGRATION` blocks, so no separate migration step is required.

### Not in this repository

There is **no** production deployment config in the repo. The following are
absent and would need to be added before a real production deploy:

- No `Dockerfile` for the FastAPI app
- No `docker-compose.prod.yml`, `fly.toml`, `vercel.json`, `netlify.toml`,
  `railway.json`, or `serverless.yml`
- No Kubernetes manifests or Helm chart
- No reverse-proxy / TLS config (nginx, Caddy, Traefik)
- No managed-Postgres provisioning (Terraform, Pulumi, etc.)

<!-- VERIFY: any cloud provider, region, or hosting platform is intentional, since the project has not chosen one -->

## Build Pipeline

There is **no CI/CD pipeline** in this repository. The `.github/` directory
does not exist, and no other workflow runner config (GitLab CI, CircleCI,
Buildkite, Jenkins) is present.

The "build" today is the local install path:

```bash
make setup    # cp -n .env.example .env; pip install -r requirements.txt
make db       # docker compose up -d
make dev      # uvicorn backend.main:app --reload
```

Tests are run manually with `make test` (`pytest`). See `docs/TESTING.md`
when generated for the full test surface. <!-- VERIFY: TESTING.md path if generated to a non-default location -->

To productionize the build, an operator would need to add at minimum:

1. A `Dockerfile` that installs `requirements.txt` into a Python 3.11 base
   image, copies `backend/` and `frontend/`, and runs
   `uvicorn backend.main:app --host 0.0.0.0 --port 8000` (no `--reload`).
2. A CI workflow that runs `pytest` and builds the image on push to `main`.
3. An image registry (GHCR, ECR, GCR, Docker Hub) to store the built image.

<!-- VERIFY: image registry, base image choice, exact uvicorn worker count, and CI provider — none of these have been chosen -->

## Environment Setup

Production deployments need the same two environment variables documented in
`docs/CONFIGURATION.md`:

| Variable | Required | Notes for production |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Must be supplied via the platform's secret manager — never baked into the image or committed to a `.env` file. The app exits with code `1` at startup if it is missing or empty (`backend/config.py:18-23`). |
| `DATABASE_URL` | Yes | Should point at a managed Postgres 16 with the `pgvector` extension installed. The bundled `rag:rag@localhost:5432/ragdb` credentials in `.env.example` are for local development only. |

`backend/config.py` calls `python-dotenv`'s `load_dotenv()` at import time,
which silently no-ops if no `.env` file is present — so injecting both vars
through the platform's environment / secrets is sufficient.

### Database notes for production

- The `pgvector/pgvector:pg16` image pins both Postgres 16 and `pgvector`. A
  managed Postgres needs the `pgvector` extension available (most modern
  managed offerings support it; verify before choosing one).
- `backend/db.py` runs `CREATE EXTENSION IF NOT EXISTS vector` on first
  connection (`db.py:8`). The connecting role must have permission to create
  extensions, or the extension must be pre-created by an admin.
- Embeddings and image bytes live in the `chunks` table — image data is
  stored as `BYTEA` inline. Plan for non-trivial row sizes if many figures
  are ingested.
- The current `docker-compose.yml` uses a named volume `pgdata` for
  durability across `docker compose down`. A production database needs its
  own backup / PITR strategy independent of the app.

<!-- VERIFY: managed Postgres provider (RDS, Cloud SQL, Supabase, Neon, etc.) and pgvector availability for that provider -->

### Files written at runtime

The app writes outside the database in two places that production deployments
must account for:

| Path | Purpose | Source |
| --- | --- | --- |
| `logs/app.log` (relative to project root) | Rotating log file, 5 MB × 3 backups | `backend/main.py:14-22` |
| `<tempdir>/docrag_current.pdf` | The most recently ingested PDF, served by `GET /doc/pdf` | `backend/main.py:33` |

In a containerized deployment these become ephemeral on container restart
unless a volume is mounted. The PDF being ephemeral is the looser constraint
(re-uploading replaces it), but the logs should ideally be shipped to a
log aggregator instead of relying on the rotating file.

<!-- VERIFY: log aggregation choice (CloudWatch, Datadog, Loki, etc.) — not configured in this repo -->

## Rollback Procedure

There is no rollback automation in this repository because there is no
deployment automation. If you stand up a production deployment, the rollback
procedure depends on the platform you choose:

- **Container-based deployments** — redeploy the previous image tag. Tag
  every release with a git SHA or semver so the previous version is
  addressable.
- **Database state** — re-deploys are safe in both directions because the
  schema is idempotent (`CREATE TABLE IF NOT EXISTS`, `DO $$ ... $$`
  migrations in `backend/db.py:36-71`). However, schema changes you add in
  the future may require their own forward-only migrations; plan accordingly.
- **Document state** — a rollback does **not** invalidate ingested data. The
  `chunks` and `doc_meta` rows persist. To wipe document state without
  touching the schema, call `DELETE /doc` against the running app or run
  `db.clear_all_chunks(pool)` in a maintenance script.

<!-- VERIFY: exact rollback steps depend on the deployment platform, which has not been chosen -->

## Monitoring

No application performance monitoring (APM), error tracking, or distributed
tracing library is wired up. `requirements.txt` does not include Sentry,
Datadog, New Relic, OpenTelemetry, or similar. The observability surface
today consists of:

| Channel | What it shows | Source |
| --- | --- | --- |
| `logs/app.log` (rotating, 5 MB × 3) | Full app log: request errors, ingest progress, query timings, "Restored doc_id" on startup | `backend/main.py:14-28` |
| stdout / stderr | Same content as the file log; useful when running under a process manager that captures stdout | `backend/main.py:24-27` |
| `GET /health` | Returns `{"status": "ok"}` — a trivial liveness probe with no dependency checks | `backend/main.py:156-158` |
| `GET /doc/info` | Reports whether a doc is loaded plus chunk count and embedding dim — useful as a smoke test after a deploy | `backend/main.py:161-170` |
| HTTP status codes | `HTTPException`s are logged at WARNING level via the custom handler | `backend/main.py:61-66` |

For production, an operator would typically add:

- A real APM (Sentry for errors, OpenTelemetry / Datadog / Honeycomb for
  traces) wrapping the FastAPI app and the `asyncpg` pool.
- A richer health check that pings the database (the current `/health`
  does not touch Postgres or Gemini).
- Alerting on Gemini 429 / 5xx rates — the ingest pipeline already logs
  these, and `_GEMINI_CONCURRENCY` / `_MARKUP_CONCURRENCY` in
  `backend/ingest.py` exist specifically to throttle them; surfacing them
  in dashboards lets you tune those constants with data.

<!-- VERIFY: monitoring vendor, dashboard URLs, and alerting policy — none configured -->

## Security Considerations for Production

Several defaults in this repository are demo-grade and must be hardened
before exposing the app to the public internet:

- **Database credentials.** `docker-compose.yml` hard-codes `rag` / `rag`
  as the Postgres user and password. Use a managed Postgres or rotate to
  long random credentials before deploying.
- **No authentication on the API.** Every route in `backend/main.py` is
  unauthenticated — anyone who can reach the host can upload, query, or
  delete the active document (`POST /ingest`, `POST /query`, `DELETE /doc`).
  Add an auth layer (API key middleware, OAuth proxy, network ACL) before
  exposing it.
- **No CORS configuration.** FastAPI's CORS middleware is not registered.
  If the frontend is served from a different origin than the API, that
  must be added.
- **TLS is not handled by the app.** Uvicorn is started without `--ssl-*`
  flags. Terminate TLS at a reverse proxy (nginx, Caddy, ALB, Cloud Run
  ingress) in front of the app.
- **Single-document state.** The app keeps a single in-memory `_state` dict
  with the active `doc_id` (`backend/main.py:35`). Two concurrent uploads
  will fight over it. The current design is single-tenant by construction.
- **Upload limit.** `_MAX_UPLOAD_BYTES = 500 * 1024 * 1024` (500 MB) caps
  individual PDF uploads (`backend/main.py:173`). Lower this on the
  reverse proxy as well to fail-fast before bytes hit the app.

<!-- VERIFY: production auth model, TLS termination point, and tenancy strategy — not present in the demo -->

## Summary: What Production Would Require

A non-exhaustive checklist if you decide to deploy this beyond a demo:

1. Add a `Dockerfile` for the FastAPI app and a CI workflow that builds and
   pushes it.
2. Provision a managed Postgres 16 with `pgvector` enabled; set
   `DATABASE_URL` via the platform's secret manager.
3. Inject `GEMINI_API_KEY` from a secret manager — never commit it.
4. Put the app behind a reverse proxy that terminates TLS and adds an auth
   layer.
5. Mount a volume for `logs/` (or, better, ship logs to an aggregator) and
   for the temp PDF directory if you want it to survive restarts.
6. Add a real `/health` that exercises the database and add APM /
   error-tracking.
7. Decide on a tenancy model — the current single-`_state` design assumes
   one user / one document at a time.
