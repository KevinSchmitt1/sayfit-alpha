# SW Engineering & MLOps Layer

> **Covers:** API · Containerization · CI/CD · Testing · Observability
> **Merge target:** this file is one section of the shared team CLAUDE.md

---

## Tech Stack

| Tool | Role |
|------|------|
| **FastAPI** | HTTP API layer over the existing pipeline |
| **Pydantic** | Request/response validation (bundled with FastAPI) |
| **SQLAlchemy (async)** | DB access — `get_db()` yields one session per request |
| **pytest** | Unit + integration tests; 80% coverage target |
| **Docker** `python:3.13-slim` | API container (voice input excluded — see constraints) |
| **docker-compose** | Orchestrates api + postgres + langfuse stack |
| **GitHub Actions** | CI: lint + tests on every push and PR |
| **Langfuse** | LLM call tracing (latency, tokens, errors) — self-hosted via compose |
| **Prometheus + Grafana** | API-level metrics (request rate, error rate, step durations) |
| **ruff** | Linter — runs in CI |

---

## Running Locally

**Pipeline (CLI, no containers):**
```bash
python main.py                  # interactive mode
python main.py --text "..."     # direct text input
python main.py --no-llm         # heuristic mode, no API keys needed
```

**Tests:**
```bash
pytest tests/unit/ -q                              # fast — retriever is mocked
pytest tests/unit/ -m llm_path -q                 # production path only
pytest tests/unit/ -m heuristic -q                # fallback path only
pytest tests/unit/ --cov=. --cov-report=term-missing   # with coverage
```

**Full stack (API + DB + Langfuse):**
```bash
docker compose up               # starts api, postgres, langfuse-web + workers
```

---

## CI/CD (`.github/workflows/ci.yml`)

Runs on every push and every PR targeting `main`.

**Current steps:**
1. `ruff check .` — syntax and lint
2. `pytest tests/unit/ -q` — unit test suite (retriever mocked, no FAISS loaded)

**Required GitHub secret:** `OPENAI_API_KEY` (prevents KeyError if any test touches the LLM path)

**Planned additions (not yet live):**
- `docker build` smoke step (Week 2)
- `services: postgres` for API integration tests (Week 2)
- `LANGFUSE_ENABLED=false` env var — SDK runs in no-op mode, full Langfuse stack not started in CI (Week 3)

---

## Testing

**Folder layout:**

```
tests/
  unit/               ← CI runs this; all current tests live here
    conftest.py       ← autouse fixture: patches retriever with fake candidates
    test_smoke.py
    test_extractor.py
    test_database.py
    test_formatter.py
    test_reranker.py
    test_retriever_scoring.py
  integration/        ← not yet written; real API calls, not run in CI by default
```

**pytest marks** (registered in `pyproject.toml`):

| Mark | What | When to run |
|------|------|-------------|
| `heuristic` | Rule-based fallback path internals (`_clean_segment`, `_parse_quantity`, `extract_items_heuristic`) | Always — fast, no mocks needed |
| `llm_path` | Production LLM path — client mocked, wiring and parsing tested | Always — no real API calls |
| `integration` | _(future)_ Real API calls, prompt quality, schema compliance | Pre-release or nightly — costs money |

**Unit test files:**

| File | What it tests |
|------|--------------|
| `unit/conftest.py` | autouse fixture: patches `step2_retrieval.retriever.retrieve` with fake candidates — scoped to `unit/` so it won't bleed into future integration tests |
| `unit/test_smoke.py` | `extract_items()` returns expected shape (`items` + `queries` keys) |
| `unit/test_extractor.py` | `@pytest.mark.heuristic`: `_clean_segment`, `_parse_quantity`, `extract_items_heuristic`; `@pytest.mark.llm_path`: JSON parsing, list→dict normalization, metadata attachment, markdown-fenced responses, malformed JSON behavior, voice noise handling |
| `unit/test_retriever_scoring.py` | Pure scoring helpers: `_extract_core_name`, `_compute_name_penalty`, `_build_query_variants`, `_safe_float` — no FAISS, no mocks needed |
| `unit/test_reranker.py` | `rerank_single_item_heuristic()` |
| `unit/test_database.py` | DB layer |
| `unit/test_formatter.py` | Formatter step |

**Planned (not yet written):**

| File | What it will test |
|------|------------------|
| `unit/test_api.py` | FastAPI endpoints via `httpx.AsyncClient` |
| `integration/test_extractor_llm.py` | Real Groq API call — prompt quality, schema compliance, voice noise correction |
| `integration/test_pipeline.py` | Full `run_pipeline()` end-to-end |

> ⚠️ **REVIEW REQUIRED — test DB strategy not yet decided**
> API and integration tests require a database. Strategy (testcontainers-python vs GitHub Actions
> `services: postgres`) depends on whether the data engineer delivers a Docker image. Revisit
> before writing `test_api.py` or `integration/test_pipeline.py`.

**Rules that apply to all unit tests:**
1. The retriever autouse mock in `unit/conftest.py` is always active — `_load_resources()` (sentence-transformers + FAISS index) is never called during any unit test run.
2. LLM-path tests mock the client — no real API calls, no cost. Real prompt quality testing belongs in `tests/integration/`.

**Input fixtures:** `input_tests/SayFit-Test_ENG_01.json` … `_29.json` — 29 real test cases usable as payload fixtures.

---

## API Endpoints

> ⚠️ **REVIEW REQUIRED — depends on data engineer's DB schema**
> Endpoint paths and operations are stable. Request/response body shapes will change once the
> data engineer finalizes table structure. Revisit before implementing `api/schemas/`.

Base URL (local): `http://localhost:8000`

| Method | Path | Body / Params | Description |
|--------|------|---------------|-------------|
| `POST` | `/log` | `{ uid, text }` | Run pipeline on text input; returns meal result and `meal_id` immediately |
| `GET` | `/meals/{uid}/today` | — | All items logged today for this user |
| `GET` | `/meals/{uid}` | — | Full meal history for this user |
| `PATCH` | `/meals/{uid}/items/{item_id}` | `{ grams }` | Post-hoc portion correction; recalculates macros |
| `POST` | `/meals/{uid}/items` | `{ meal_id, item_name, grams }` | Manually add a missed item to an existing meal |
| `DELETE` | `/meals/{uid}/items/{item_id}` | — | Remove an item from a meal |

**Note:** `POST /log` does not run the interactive correction loop (`ask_user_corrections()`). That loop is CLI-only. The full add/edit/remove flow lives in the UI and maps to the three mutation endpoints above.

Schemas live in `api/schemas/`. Input is validated by Pydantic at the HTTP boundary; DB queries use SQLAlchemy parameterized queries.

---

## Docker Topology

> ⚠️ **REVIEW REQUIRED — depends on data engineer's DB decision (DuckDB vs Postgres)**
> If DuckDB is chosen, the postgres service below is app-only. If Postgres is chosen for both,
> the data pipeline may share this container or need its own. Revisit once that decision is locked.

```
api              python:3.13-slim
                 depends_on: postgres (healthy), langfuse-web (started)
                 mounts: data/faiss_index (read-only)
                 env: DATABASE_URL, OPENAI_API_KEY, LANGFUSE_HOST,
                      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

postgres         postgres:16-alpine
                 volume: postgres_data (survives compose down)
                 healthcheck: pg_isready -U sayfit

langfuse-web     official Langfuse self-hosted image
                 + langfuse-worker, clickhouse, minio, redis

prometheus       scrapes FastAPI /metrics                  (Week 3)
grafana          dashboard over prometheus data            (Week 3)
```

**Connection string pattern:** `postgresql+psycopg://sayfit:${DB_PASSWORD}@postgres:5432/sayfit`

The same `DATABASE_URL` env var is used locally, in compose, and in CI — no code path changes between environments.

---

## Observability

### Langfuse — LLM tracing

Two functions are wrapped with `@observe()`:
- `step1_extraction/extractor.py` → `extract_items()` → span name `step1_extraction`
- `step3_reranker/reranker.py` → `rerank_single_item()` → span name `step3_reranker`

`run_pipeline()` in `main.py` has an outer `@observe()` so the full pipeline appears as one trace with per-step child spans.

**Trace naming contract:**

> ⚠️ **REVIEW REQUIRED — not yet confirmed with ML engineer**
> Names below are proposed. Do not write integration code against these until the ML engineer
> has signed off. Both sides must agree before either side implements.

| Level | Name |
|-------|------|
| Top-level trace | `pipeline_run` |
| Extraction span | `step1_extraction` |
| Reranker span | `step3_reranker` |

**Common metadata keys:** `model`, `user_id`, `input_length`, `recipe_db_version`

**Scope boundary:**

| Concern | Owner |
|---------|-------|
| LLM latency, token cost, error rate, request volume | SW/MLOps (this layer) |
| Output quality scoring, prompt A/B evaluation | ML engineer |

The SW/MLOps layer sets up traces. The ML engineer attaches eval scores on top of the same traces. These do not overlap.

**In CI:** set `LANGFUSE_ENABLED=false` — SDK runs in no-op mode, nothing breaks, full stack never started.

### Prometheus + Grafana — API metrics

FastAPI exposes `/metrics` (via `prometheus-fastapi-instrumentator` or equivalent). Grafana dashboard covers: request rate, error rate, p95 latency, per-step durations.

---

## Constraints (affect the whole team)

**Step isolation is a hard rule.**
Every step module is callable standalone via `python -m stepN_*.run`. Steps do not import each other's internals. The API calls step functions directly — it does not shell out and does not introduce cross-step imports.

**Voice input is excluded from the API and Docker.**
`step0_voice_input` requires PortAudio and runs in an isolated subprocess from `main.py` to avoid a macOS PortAudio/PyTorch/FAISS semaphore conflict. `POST /log` accepts text only. The Docker image does not include PortAudio.

**Environment variables only — no hardcoded secrets.**
All credentials go through env vars validated at startup. `.env.example` documents every required variable. `.env` is gitignored.

---

## Environment Variables

| Variable | Required from | Purpose |
|----------|--------------|---------|
| `OPENAI_API_KEY` | Day 1 | LLM calls (Groq-compatible client) |
| `DB_PASSWORD` | Week 2 | Postgres container |
| `LANGFUSE_PUBLIC_KEY` | Week 3 | Langfuse SDK |
| `LANGFUSE_SECRET_KEY` | Week 3 | Langfuse SDK |
| `LANGFUSE_ENABLED` | Week 3 | Set `false` in CI to disable SDK |

---

## Known Technical Debt

| Item | Where | Action |
|------|-------|--------|
| `food_portion_lookup.py` duplicated | root + `step1_5_ontology_filter/` | Consolidate once test coverage exists as safety net |
| `main.py` is 1,226 lines | `main.py` | Split into smaller modules — do after tests exist |
| `data/sayfit_meals.db` tracked in git | `.gitignore` | Gitignore once Postgres migration lands |
