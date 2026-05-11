# Pre-Planning: Dependencies & SQL/Docker Architecture

> **What this file is:** team alignment doc — what must be agreed with the data engineer and ML engineer before Kevin's technical work can proceed.
> **Kevin's technical plan lives in:** [sw-eng_mlops.md](sw-eng_mlops.md)

> CI scaffold is the one exception — it starts Week 1 Day 1 with no external dependencies.

---

## 1. Dependencies on the Data Engineer (Recipe SQL DB)

The data layer migration touches almost every box in the existing plan.

| My task | What changes when CSV → SQL | Action needed |
|---|---|---|
| Tests (Week 1) | Step 2 retrieval + Step 3 reranker currently read `data/combined_final.csv`. Once recipes live in SQL, fixtures must seed a test DB, not a CSV. | Decide test strategy *now* (see §4) so Week 1 tests aren't rewritten in Week 2. |
| FastAPI (Week 2) | `POST /log` and `GET /meals/...` need a DB session, not just file IO. Meal records likely reference `recipe_id` / `ingredient_id` foreign keys. | Get the **schema draft** from the data engineer before designing API response models. |
| Docker (Week 2) | No longer a single-container app — needs at minimum `api` + `postgres`. | Move from `Dockerfile` to `docker-compose.yml` (see §5). |
| CI postgres service (Week 2) | Existing plan mocks the retriever to avoid loading FAISS in CI. Now also need to decide: real ephemeral Postgres in CI, or mock the DB layer too. | Pick one strategy and stick with it. Recommendation: `services: postgres` in GitHub Actions. |
| Langfuse traces (Week 3) | Span metadata should include the **recipe DB version / commit hash** so traces are reproducible against the data state. | Coordinate with data eng on a versioning scheme (alembic revision, or DB-side `schema_version` table). |

**Hard blockers** — work that cannot finish until the data engineer delivers:
- Final API response schemas (need recipe/ingredient SQL shape).
- Docker compose for the API path (need DB image + connection string contract).
- Integration tests against real data flow.

**Non-blockers** — work that can start immediately:
- CI scaffold: `conftest.py` + `ci.yml` (Week 1 Day 1–2, zero external dependencies).
- Unit tests of pure functions in `step1_extraction` using `use_llm=False` — no LLM or FAISS needed.
- FastAPI scaffolding (routes, Pydantic request models, error handlers) against an interface, not the DB.
- Langfuse `@observe()` wiring on `extract_items()` and `rerank_single_item()` — pure LLM concern, independent of the data layer.

**Recommendation:** ask the data engineer for the **schema draft + a stub DB image** by end of Week 1.
If that slips, Week 2 (API + Docker) slips with it. Make this explicit in the team sync.

---

## 2. Dependencies on the ML Engineer (Shared Langfuse)

Before either side wires Langfuse in, lock these contracts:

**Lock these contracts:**
- **Hosting** — one self-hosted instance in `docker-compose.yml`, or Langfuse Cloud?
- **Project separation** — single project with tag-based filtering, or two projects (`sayfit-prod`, `sayfit-eval`)?
- **Trace naming** — `pipeline_run` + `step1_extraction` / `step3_reranker` / `step4_scoring`. Get ML engineer sign-off.
- **Prompt versioning ownership** — Kevin manages registered prompts in Langfuse (Week 3 task 5). Confirm no overlap.
- **Metadata schema** — agree on common keys: `model`, `user_id`, `input_length`, `recipe_db_version`.

**Scope boundary — agree on this explicitly:**

| Concern | Owner |
|---|---|
| LLM latency, token cost, error rate, request volume | Kevin (MLOps) |
| Output quality scoring, prompt A/B evaluation | ML engineer |

Kevin sets up the traces. ML engineer attaches eval scores on top. These do not overlap —
but both sides must agree on trace structure before either writes integration code.

**One-way dependency:** Kevin ships operational traces first. ML engineer's eval layer depends
on the trace structure being stable. SW Eng ships first, ML extends.

---

## 3. Cross-Cutting Considerations

- **Environment & secrets** — every new moving part goes through env vars validated at startup.
  Add a `.env.example`; never commit `.env`.
  Progressive rollout: `OPENAI_API_KEY` (Week 1) → `DB_PASSWORD` (Week 2) → `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (Week 3).
- **Test coverage target** — 80% floor with unit + integration. With a DB in the loop,
  "integration" means "API + DB", not just step-to-step. Plan that in Week 1, not Week 2.
- **Boundary validation** — FastAPI + Pydantic at the HTTP boundary;
  SQLAlchemy + parameterized queries at the DB boundary. Don't hand-roll either.
- **Scope discipline** — DVC stays out. MLflow stays out. The recipe SQL DB doesn't drag them back in.
- **Langfuse in CI** — set `LANGFUSE_ENABLED=false` in the CI workflow.
  The SDK runs in no-op mode; the full stack (clickhouse, redis, worker) is never started in CI.

---

## 4. Test DB Strategy — Depends on Data Engineer's Architecture

This decision cannot be made unilaterally — it depends on what database setup the data engineer delivers. The three options below map to different data engineer outputs:

| Strategy | When it fits | Pros | Cons |
|---|---|---|---|
| **SQLite in-memory** for unit tests, real Postgres only in integration | Data engineer delivers a Postgres schema late | Fast unit tests, no Docker needed for most runs | Postgres/SQLite dialect drift; some SQL won't translate |
| **testcontainers-python** (ephemeral Postgres per test session) | Data engineer delivers a Docker image early | Real Postgres, isolated, works in CI | Slower test startup (~5–10s); needs Docker in CI |
| **GitHub Actions `services: postgres`** | Data engineer delivers schema + seed data | Free, fast in CI; local devs run docker-compose | Local + CI parity requires discipline |

**Recommended:** testcontainers-python — but only confirmable once the data engineer confirms a Docker image will be available. Raise this in the next team sync.

---

## 5. SQL + Docker + API — The Technical Architecture

### `docker-compose.yml` topology

```yaml
services:
  api:
    depends_on:
      postgres: { condition: service_healthy }
      langfuse-web: { condition: service_started }
    environment:
      DATABASE_URL: postgresql+psycopg://sayfit:${DB_PASSWORD}@postgres:5432/sayfit
      LANGFUSE_HOST: http://langfuse-web:3000
      LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY}
      LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: sayfit
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: sayfit
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sayfit"]
      interval: 5s

  langfuse-web:  # + langfuse-worker, clickhouse, minio, redis per official compose
    ...

volumes:
  postgres_data:
```

### The five things that make this work cleanly

1. **Connection string via env var** — same code runs locally, in compose, and in CI.
2. **`depends_on: service_healthy`** — API waits for Postgres to accept connections.
3. **Named volume** — `postgres_data` survives `docker compose down`. Use `down -v` only for a clean slate.
4. **Alembic migrations as a one-shot `migrate` service** — data engineer owns migration files; Kevin owns the migration runner in the API image.
5. **SQLAlchemy async engine** + `get_db()` dependency — yields a session per request, never opens a new engine per request.

### Where the rest plugs in

- **FAISS index** — still a file. Mounted read-only into the API container.
- **Langfuse self-hosted** — adds 4 services. API only talks to `langfuse-web`.
- **Prometheus + Grafana** — scrape FastAPI `/metrics`. Independent of the DB.

### CI implications (Week 2 additions to ci.yml)

- Add `services: postgres` to the workflow.
- Keep mocked FAISS retriever (already in `conftest.py` from Week 1).
- Set `LANGFUSE_ENABLED=false` — SDK no-op, full Langfuse stack not started in CI.
- Add `docker build` smoke step.

---

## TL;DR — Before Planning the Three Weeks

1. **CI scaffold first** — `conftest.py` + `ci.yml` on Week 1 Day 1. No external dependencies. Push to main immediately so teammates have a safety net from day one.
2. **Get the recipe SQL schema draft from the data engineer by end of Week 1.** API design and test strategy both block on it.
3. **Lock the Langfuse contract with the ML engineer this week.** Hosting, project layout, trace names, prompt-versioning ownership, and the scope boundary (Kevin = operational, ML engineer = quality).
4. **Confirm data engineer's Docker image availability** — this determines which test DB strategy to use (§4).
5. **Move from `Dockerfile` to `docker-compose.yml` in Week 2.** Single-container is dead the moment Postgres lands.
6. **Keep the scope discipline from the existing plan.** DVC stays out. MLflow stays out.
