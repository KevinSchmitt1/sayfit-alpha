# SW Engineering & MLOps Layer

> **Covers:** API · Containerization · CI/CD · Testing · Observability
> **Merge target:** this file is one section of the shared team CLAUDE.md

---

## Tech Stack

| Tool | Role |
|------|------|
| **FastAPI** | HTTP API layer over the existing pipeline |
| **Pydantic** | Request/response validation (bundled with FastAPI) |
| **sqlite3** | Meals DB — OLTP workload (row-level inserts/updates/reads); file-based, no container needed |
| **DuckDB** | Data pipeline DB only — OLAP workload (large scans over nutrition dataset); owned by data engineer |
| **pytest** | Unit + integration tests; 80% coverage target |
| **Docker** `python:3.13-slim` | API container (voice input excluded — see constraints) |
| **docker-compose** | Two separate files: `docker-compose.yml` (API only) · `docker-compose.data.yml` (pipeline, TBD) |
| **GitHub Actions** | CI: lint + tests on every push and PR |
| **Langfuse** | LLM call tracing (latency, tokens, errors) — cloud (langfuse.com); no container needed |
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

**Full stack (API + DB):**
```bash
docker compose up               # starts api; Langfuse traces go to cloud
```

---

## CI/CD (`.github/workflows/ci.yml`)

Runs on every push and every PR targeting `main`.

**Current steps:**
1. `ruff check .` — syntax and lint
2. `pytest tests/unit/ -q` — unit test suite (retriever mocked, no FAISS loaded)
3. `pytest tests/integration/test_extractor_llm.py -q` — real OpenAI call, schema compliance only
4. `docker build .` — smoke test: confirms image builds without errors

**Required GitHub secret:** `OPENAI_API_KEY` — used by both unit and integration test steps

**Planned additions (not yet live):**
- `integration of step6` (recipe module — pending)
---

## Testing

**Folder layout:**

```
tests/
  unit/               ← CI always runs this (pytest tests/unit/ -q)
    conftest.py       ← autouse fixture: patches retriever with fake candidates
    test_smoke.py
    test_extractor.py
    test_database.py
    test_formatter.py
    test_reranker.py
    test_retriever_scoring.py
    test_api.py       ← all 7 FastAPI endpoints via TestClient
  integration/        ← planned; test_extractor_llm.py runs in CI, test_pipeline.py does not
    test_extractor_llm.py   ← real Groq API call; needs GROQ_API_KEY secret in CI
    test_pipeline.py        ← full pipeline; needs FAISS index — local only
```

**pytest marks** (registered in `pyproject.toml`):

| Mark | What | When to run |
|------|------|-------------|
| `heuristic` | Rule-based fallback path internals (`_clean_segment`, `_parse_quantity`, `extract_items_heuristic`) | Always — fast, no mocks needed |
| `llm_path` | Production LLM path — client mocked, wiring and parsing tested | Always — no real API calls |
| `integration` | Real API/LLM calls, schema compliance | CI (extractor only) + local (full pipeline) |

**Unit test files:**

| File | What it tests | Status |
|------|--------------|--------|
| `unit/conftest.py` | autouse fixture: patches `step2_retrieval.retriever.retrieve` with fake candidates — scoped to `unit/` so it won't bleed into integration tests | ✅ Done |
| `unit/test_smoke.py` | `extract_items()` returns expected shape (`items` + `queries` keys) | ✅ Done |
| `unit/test_extractor.py` | `@pytest.mark.heuristic`: `_clean_segment`, `_parse_quantity`, `extract_items_heuristic`; `@pytest.mark.llm_path`: JSON parsing, list→dict normalization, metadata attachment, markdown-fenced responses, malformed JSON behavior, voice noise handling | ✅ Done |
| `unit/test_retriever_scoring.py` | Pure scoring helpers: `_extract_core_name`, `_compute_name_penalty`, `_build_query_variants`, `_safe_float` — no FAISS, no mocks needed | ✅ Done |
| `unit/test_reranker.py` | `rerank_single_item_heuristic()` | ✅ Done |
| `unit/test_database.py` | DB layer | ✅ Done |
| `unit/test_formatter.py` | Formatter step | ✅ Done |
| `unit/test_api.py` | All 7 FastAPI endpoints via TestClient; retriever mock active; temp SQLite DB fixture | ✅ Done |

**Integration test files:**

| File | What it tests | Runs in CI | Status |
|------|--------------|-----------|--------|
| `integration/test_extractor_llm.py` | Real OpenAI API call — schema compliance, basic correctness (returns valid food items structure) | ✅ Yes — needs `OPENAI_API_KEY` secret | ✅ Done |
| `integration/test_pipeline.py` | Full `run_pipeline()` end-to-end | ❌ No — needs FAISS index on disk | ✅ Done (local only) |

> **Test DB strategy: SQLite temp file.** No container needed. API tests use pytest's `tmp_path`
> fixture to create a temporary SQLite file per test. DuckDB is the data engineer's OLAP
> pipeline DB — unrelated to API tests.

**Rules that apply to all unit tests:**
1. The retriever autouse mock in `unit/conftest.py` is always active — `_load_resources()` (sentence-transformers + FAISS index) is never called during any unit test run.
2. LLM calls are always mocked — no real API calls, no cost.

**Rules that apply to integration tests:**
1. No retriever mock — real FAISS or real Groq client used depending on the test.
2. Assertions check structure and schema only, not exact LLM output (non-deterministic). Prompt quality monitoring belongs in Langfuse, not tests.
3. `test_pipeline.py` requires FAISS index on disk — run locally only, never in CI.

**Input fixtures:** `input_tests/SayFit-Test_ENG_01.json` … `_29.json` — 29 real test cases usable as payload fixtures.

---

## API Endpoints

Base URL (local): `http://localhost:8000`

| Method | Path | Body / Params | Response | Description |
|--------|------|---------------|----------|-------------|
| `POST` | `/log` | `{ uid, text }` | `Meal` (201) | Run pipeline on text input; returns structured meal with item UUIDs |
| `GET` | `/meals/{uid}/today` | — | `list[Meal]` (200) | All meals with full item detail logged today |
| `GET` | `/meals/{uid}` | `?days=N` (default 30) | `MealHistory` (200) | Daily macro summary for the last N days |
| `PATCH` | `/meals/{uid}/items/{item_id}` | `{ meal_id, grams }` | 204 | Rescale item portion; macros recalculated proportionally |
| `POST` | `/meals/{uid}/items` | `{ meal_id, item_name, grams }` | `FoodItem` (201) | Add a missed item; nutrition resolved via FAISS top candidate |
| `DELETE` | `/meals/{uid}/items/{item_id}` | `?meal_id=<meal_id>` | 204 | Soft-delete item; recalculates meal totals |
| `DELETE` | `/meals/{uid}/{meal_id}` | — | 204 | Soft-delete a whole meal and all its items |

**Note:** `POST /log` does not run the interactive correction loop (`ask_user_corrections()`). That loop is CLI-only. The full add/edit/remove flow lives in the UI and maps to the four mutation endpoints above.

**Mutation pattern:** PATCH and DELETE endpoints return 204 No Content. Callers should follow up with `GET /meals/{uid}/today` to refresh state.

Schemas live in `api/schemas.py`. Input is validated by Pydantic at the HTTP boundary; DB queries use parameterized sqlite3 statements.

---

## Docker Topology

**Two separate compose files** — API stack and data pipeline are decoupled by design.

| File | Purpose | Cadence |
|------|---------|---------|
| `docker-compose.yml` | API only — always-on runtime service | Production / local dev |
| `docker-compose.data.yml` | Data pipeline — dbt + Prefect + DuckDB (TBD) | Occasional rebuild runs (weekly/monthly) |

**Rationale:** The data pipeline is a *build-time* concern — it runs occasionally to rebuild the FAISS index and nutrition database. The API is a *runtime* concern — it runs continuously to serve requests. Coupling them in one compose file would mean running a full orchestration stack 24/7 for a job that fires once a month.

**Handoff:** The pipeline writes the FAISS index and `combined_final.csv` to a shared volume. The API mounts that volume read-only.

### `docker-compose.yml` — API stack

```
api              python:3.13-slim
                 ports: 8000:8000
                 mounts: data/faiss_index (read-only), data/sayfit_meals.db (read-write)
                 env_file: .env
                 env: OPENAI_API_KEY, LANGFUSE_HOST (cloud),
                      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

prometheus       image: prom/prometheus
                 ports: 9090:9090
                 config: prometheus/prometheus.yml → scrapes api:8000/metrics every 15s
                 volume: prometheus_data:/prometheus (named, persistent)

grafana          image: grafana/grafana
                 ports: 3001:3000  (3000 reserved for future frontend)
                 volume: grafana_data:/var/lib/grafana (named, persistent)
                 datasource: http://prometheus:9090 (internal Docker DNS)
```

**Note:** No Postgres container. Meals DB is SQLite — file-based, mounted as a volume, no server process needed.
**Note:** No Langfuse container. Langfuse runs as a cloud service (langfuse.com). Set `LANGFUSE_HOST=https://cloud.langfuse.com` in `.env`.

**Startup check:** `api/main.py` lifespan function runs on container start:
- If `data/faiss_index` missing → raises `FileNotFoundError` with clear message (data pipeline must build it first)
- If `data/sayfit_meals.db` missing → creates empty file via `Path.touch()` (safe on first run)

### `docker-compose.data.yml` — Data pipeline

> ⚠️ **TBD — pending data engineer.** Architecture to be defined once data engineer begins work.
> Will use dbt + Prefect + DuckDB. Will write FAISS index and nutrition DB to the shared volume the API reads from.

---

## Observability

### Langfuse — LLM tracing

Three `@observe()` decorators are live — confirmed with ML engineer:

| File | Function | Decorator | Span name |
|------|----------|-----------|-----------|
| `main.py:153` | `run_pipeline()` | `@observe(name="pipeline_run")` | `pipeline_run` — top-level trace |
| `step1_extraction/extractor.py:173` | `extract_items()` | `@observe(name="step1_extraction")` | `step1_extraction` — child span |
| `step3_reranker/reranker.py:96` | `rerank_single_item()` | `@observe(name="step3_reranker")` | `step3_reranker` — child span |

The full pipeline appears as one trace in Langfuse with two child spans. Do not rename these without syncing with the ML engineer — they attach eval scores to the same trace names.

**Trace naming contract: ✅ confirmed**

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

Implemented via `prometheus-fastapi-instrumentator`.

**Implementation:**
- `api/main.py`: `Instrumentator().instrument(app)` at module level (attaches middleware to every request); `instrumentator.expose(app)` inside lifespan before `yield` (registers `GET /metrics` route)
- `prometheus/prometheus.yml`: scrape config — job `sayfit_api`, target `api:8000`, interval 15s
- `docker-compose.yml`: `prometheus` + `grafana` services with named volumes (`prometheus_data`, `grafana_data`) for data persistence across restarts
- `requirements-api.txt`: `prometheus-fastapi-instrumentator>=0.10.0` under `# --- Observability ---`

**Metrics available at `GET /metrics`:**
- `http_requests_total` — request count labelled by handler, method, status bucket (2xx/4xx/5xx)
- `http_request_duration_seconds` — latency histogram by handler (use for p95 queries)
- `http_request_size_bytes` / `http_response_size_bytes` — payload sizes
- `pipeline_duration_seconds` — custom histogram: end-to-end `run_pipeline()` duration; buckets tuned for LLM workloads `[1, 2, 5, 10, 15, 20, 30, 60]`
- `pipeline_errors_total` — custom counter: pipeline failures labelled by `error_type` (e.g. `ValueError`, `FileNotFoundError`)

**Access:**
- Prometheus UI: `http://localhost:9090` — check `Status → Target health`, target `sayfit_api` should show `UP`
- Grafana UI: `http://localhost:3001` — datasource URL: `http://prometheus:9090` (internal Docker DNS, not localhost)

**Grafana dashboard (TODO):** request rate (`sum by (handler) (rate(http_requests_total[1m]))`), p95 latency (`histogram_quantile(0.95, sum by (handler, le) (rate(http_request_duration_seconds_bucket[5m])))`), pipeline duration (`histogram_quantile(0.95, rate(pipeline_duration_seconds_bucket[5m]))`), pipeline errors (`rate(pipeline_errors_total[5m])`) — set up manually via UI at `localhost:3001`, persisted in `grafana_data` volume.

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
| `GROQ_API_KEY` | Day 1 | Groq LLM backend (free tier) |
| `LANGFUSE_PUBLIC_KEY` | Now | Langfuse cloud SDK |
| `LANGFUSE_SECRET_KEY` | Now | Langfuse cloud SDK |
| `LANGFUSE_HOST` | Now | `https://cloud.langfuse.com` |
| `LANGFUSE_ENABLED` | Now | Set `false` in CI to disable SDK |

---

## Known Technical Debt

| Item | Where | Action |
|------|-------|--------|
| `food_portion_lookup.py` duplicated | root + `step1_5_ontology_filter/` | Consolidate once test coverage exists as safety net |
| `main.py` is 1,226 lines | `main.py` | Split into smaller modules — do after tests exist |
| `data/sayfit_meals.db` tracked in git | `.gitignore` | Gitignore — SQLite file should not be in version control; remove once team agrees on a clean state |

---

## Open TODOs

### SW/MLOps (Kevin)

| Todo | Blocked by | Notes |
|------|-----------|-------|
| ~~Write `tests/unit/test_api.py`~~ | ✅ Done | All 7 endpoints via TestClient; temp SQLite fixture |
| ~~Write `tests/integration/test_extractor_llm.py`~~ | ✅ Done | Uses `OPENAI_API_KEY`; runs in CI |
| ~~Write `tests/integration/test_pipeline.py`~~ | ✅ Done | Local only, not CI |
| ~~Add `LANGFUSE_ENABLED=false` to CI env~~ | ✅ Decided against | SDK degrades gracefully when credentials missing — logs a warning, no crash, no functional need for the flag |
| ~~Prometheus + Grafana~~ | ✅ Done | `/metrics` live, Prometheus scraping, Grafana connected |
| ~~Grafana dashboard~~ | ✅ Done | 4 panels: Request Rate, p95 Latency, Pipeline p95 Duration, Pipeline Errors |
| ~~Rebuild Docker with ontology fix~~ | ✅ Done | `ontology_filter.py:336` guard shipped in Docker image |
| Gitignore `data/sayfit_meals.db` | Team agreement | File should not be in version control |

### Dependent on ML engineer

| Todo | Who | Notes |
|------|-----|-------|
| Add React frontend service to `docker-compose.yml` | ML engineer | New `frontend` service, port 3000, points at `api:8000` internally |
| Confirm frontend API base URL config | ML engineer + Kevin | Must agree on how frontend resolves the API URL in dev vs production |
| Attach Langfuse eval scores to traces | ML engineer | Builds on top of existing `pipeline_run` / `step1_extraction` / `step3_reranker` spans |

### Dependent on data engineer

| Todo | Who | Notes |
|------|-----|-------|
| Define `docker-compose.data.yml` | Data engineer | dbt + Prefect + DuckDB — architecture TBD |
| FAISS index handoff | Data engineer + Kevin | Pipeline writes index to shared volume; API mounts it read-only. Trigger and path need agreement. |
| FAISS index rebuild trigger | Data engineer | When data pipeline produces new CSV, who rebuilds the index and how? |

### Dependent on recipe module

| Todo | Who | Notes |
|------|-----|-------|
| Integrate step 6 into pipeline | Recipe module owner | New pipeline step after step 5; must respect step isolation contract |
| Recipe API endpoints | Recipe module owner + Kevin | Endpoints needed before frontend can surface recipe suggestions |
| Add step 6 to `docker-compose.yml` if needed | Kevin | Only if step 6 introduces a new service (e.g. vector DB for recipes) |

---

## Integration Checkpoints — Kevin's action items when teammates merge

Concrete changes Kevin needs to make to the API, Docker, CI, or tests when each team member ships their piece. Update this section each time a dependency lands.

### When ML engineer ships the Next.js frontend

| Action | File | Detail |
|--------|------|--------|
| Add CORS middleware | `api/main.py` | Browser on port 3000 calling API on port 8000 is a cross-origin request — the browser will block it without `CORSMiddleware`. Add `from fastapi.middleware.cors import CORSMiddleware`, configure allowed origins (`localhost:3000` for dev). |
| Add `frontend` service | `docker-compose.yml` | New service: image TBD by ML engineer, port `3000:3000`, env var for API URL. |
| Agree on API base URL pattern | `docker-compose.yml` + ML engineer | Browser-side JS must call `localhost:8000`; server-side Next.js (SSR) can call `api:8000` internally. Agree on which env var controls this (`NEXT_PUBLIC_API_URL` vs `API_URL`). |
| Add CORS header assertion to tests | `tests/unit/test_api.py` | After CORS middleware is added, assert `Access-Control-Allow-Origin` is present in responses. |

### When data engineer ships the data pipeline

| Action | File | Detail |
|--------|------|--------|
| Switch FAISS bind mount to named volume | `docker-compose.yml` | Currently `./data/faiss_index:/app/data/faiss_index:ro` (bind mount to local folder). Once the data pipeline writes to a named Docker volume, change to `faiss_index:/app/data/faiss_index:ro` and declare `faiss_index:` in the top-level `volumes:` block. Coordinate the exact volume name with the data engineer. |
| Confirm index rebuild trigger | `docker-compose.yml` + data engineer | Agree on how the API container picks up a new FAISS index after the pipeline runs — container restart signal, volume swap, or a file watcher. No API code change until the pattern is agreed. |
| Review `docker-compose.data.yml` | `docker-compose.data.yml` | Data engineer owns this file. Kevin reviews to confirm shared volume names match `docker-compose.yml` exactly and there are no port conflicts with the API stack. |

### When recipe module ships step 6

| Action | File | Detail |
|--------|------|--------|
| Add recipe endpoints | `api/recipes.py` (new file) | Recipe module owner defines the contract. Kevin implements FastAPI routes wired to the step 6 function. Put in a new `api/recipes.py` if more than 2 routes — `api/main.py` is already long. Include the router via `app.include_router()`. |
| Add Pydantic schemas | `api/schemas.py` | New request/response models for recipe endpoints. |
| Add unit tests | `tests/unit/test_api.py` | Same TestClient + mock pattern as existing endpoint tests. Mock step 6 the same way `run_pipeline` is currently mocked. |
| Add new packages if needed | `requirements-api.txt` | If step 6 introduces new dependencies, add under the relevant section. This invalidates the Docker pip layer — expect a slow rebuild once. |
| Add new service if needed | `docker-compose.yml` | Only if step 6 needs its own container (e.g. a vector DB for recipe embeddings). Confirm with recipe module owner before adding — don't add speculatively. |
| Add CI secret if needed | `.github/workflows/ci.yml` + GitHub settings | Only if step 6 integration test needs a new API key. Add the secret to GitHub Actions repo settings first, then reference it in the workflow. |
