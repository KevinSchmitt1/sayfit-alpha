# Capstone 2 Plan: Kevin's Modules
## Software Engineering & APIs + MLOps & Monitoring

**Modules:** SW Engineering & APIs (curriculum weeks 1–2) · MLOps & Monitoring (curriculum weeks 4–5)
**Project timeline:** 3 project weeks (Week 1 = CI scaffold + tests; Week 2 = FastAPI + Docker; Week 3 = Langfuse + Prometheus/Grafana)
**Project:** sayfit-alpha

---

## My Working Points (Ownership)

| Area | What I'm building | Module |
|---|---|---|
| **CI scaffold** | GitHub Actions: lint + tests on every push | MLOps |
| **Test suite** | pytest coverage for all steps + integration | SW Eng |
| **API layer** | FastAPI wrapper around the existing pipeline | SW Eng |
| **Containerization** | Docker + docker-compose (API + Postgres + Langfuse) | SW Eng |
| **LLM observability** | Langfuse tracing for every LLM call — operational metrics only | MLOps |
| **API metrics** | Prometheus + Grafana for request-level monitoring | MLOps |

---

## Module 1: Software Engineering & APIs

### Tech stack
| Tool | Role |
|---|---|
| **pytest** | Unit + integration tests |
| **FastAPI** | API layer on top of the existing pipeline |
| **Docker** (`python:3.13-slim`) | Containerize the API path |
| **Podman** | Know it — worth mentioning in presentation as the secure production alternative |

### Why these choices
- **pytest** — fixtures already exist (29 test JSONs in `input_tests/`, each step's `example_input.json`/`example_output.json`). No setup cost, just write the tests.
- **FastAPI** — the pipeline is already modular: each step is standalone and outputs JSON. FastAPI just adds a transport layer without touching any existing code.
- **Docker** — Python 3.14 has no official slim image yet; `python:3.13-slim` works with all current deps. Voice input stays CLI-only (PortAudio doesn't work cleanly in containers — already isolated via subprocess).

### Step-by-step tasks

**Week 1 (Project Week 1) — CI first, then tests**

Day 1–2: CI scaffold (do this before any other code)
1. Create `tests/conftest.py` — patches `step2_retrieval.retriever.retrieve` with a hardcoded fake candidate list using `autouse=True`; runs automatically before every test without being called explicitly. FAISS and sentence-transformers are never invoked — their imports are fast, only `_load_resources()` is slow, and the mock prevents it from ever being called.
2. Create `.github/workflows/ci.yml` — two steps: `ruff check .` (syntax + lint) + `pytest tests/ -q --ignore=step0_voice_input`. Add `OPENAI_API_KEY` from GitHub secrets (Settings → Secrets → Actions). Push to main — CI is now live.

Day 3–5: Tests
3. Write unit tests for each step using `use_llm=False` where possible (extraction heuristic path requires no LLM or mock — zero setup cost)
4. Write integration test for the full `run_pipeline()` flow
5. Verify 80%+ coverage
6. Fix `food_portion_lookup.py` duplication (exists at root and under `step1_5/`) — under test cover now

**Week 2 (Project Week 2) — API + Docker**
7. Design the API correction model (see note below)
8. Create `api/` directory with FastAPI app
9. Add endpoints: `POST /log`, `GET /meals/{uid}/today`, `GET /meals/{uid}`, `PATCH /meals/{uid}/items/{item_id}` (post-hoc correction)
10. Wire endpoints to existing step modules (no changes to steps)
11. Write `Dockerfile` using `python:3.13-slim`
12. Test API path in container; confirm voice input falls back gracefully
13. Add to ci.yml: `docker build` smoke step + `services: postgres` block so API integration tests run against a real database. Add `DB_PASSWORD` to GitHub secrets.

> **API correction design decision:** `run_pipeline()` includes `ask_user_corrections()` — an interactive terminal loop that can't exist in a REST API. Decision: `POST /log` returns results immediately (no interactive loop); corrections are handled via `PATCH /meals/{uid}/items/{item_id}` after the fact. The CLI interactive loop remains intact for terminal use; the API path skips it.

---

## Module 2: MLOps & Monitoring

### My three MLOps tools
| Tool | Role | Scope |
|---|---|---|
| **GitHub Actions** | CI: structural health check on every push | Catches broken code |
| **Langfuse** | LLM operational tracing (latency, tokens, errors) | Catches degraded runtime behavior |
| **Prometheus + Grafana** | API infrastructure metrics (request rate, error rate, step durations) | Catches infrastructure problems |

> **Docker is a SW Eng prerequisite, not an MLOps tool.** Langfuse self-hosted runs inside docker-compose alongside the API — that's why Docker (Week 2) must be done before Langfuse (Week 3).

### On DVC vs Langfuse
The MLOps module curriculum includes **DVC** (Data Version Control) for data pipeline versioning and reproducibility. DVC is the right tool for tracking large datasets like `data/combined_final.csv` and the FAISS index across runs.

**For sayfit-alpha, my implementation focus is Langfuse instead of DVC.** Here's why:

SayFit is an LLM pipeline, not a classical ML training pipeline. The operational risks are:
- LLM call latency going up
- Token costs spiking
- LLM calls failing or timing out

DVC addresses none of these. Langfuse addresses all of them in production.

> DVC would be valuable if `sayfit-data-repo` needed reproducible rebuild pipelines. That's upstream infrastructure, not the focus of this capstone. Worth knowing and presenting as the correct tool for that problem.

### Why Langfuse
- **Built specifically for LLM pipelines** — not a classical ML tool being stretched to fit
- **Open-source and self-hostable** — runs locally via Docker Compose alongside the API
- **Low integration cost** — the `@observe()` decorator wraps existing functions with no structural changes
- **Bridges into the team's ML Engineering module** — the ML engineer can extend the same Langfuse instance with eval scores on top of the operational traces I set up

### What I monitor vs what the ML engineer monitors

| Concern | Owner | What it answers |
|---|---|---|
| LLM call latency per step | Kevin (MLOps) | Is step1 or step3 getting slower? |
| Token count per request | Kevin (MLOps) | Is cost spiking after a change? |
| LLM error rate / timeouts | Kevin (MLOps) | Are calls failing? |
| Pipeline run volume | Kevin (MLOps) | How many runs per hour/day? |
| Output quality scoring | ML engineer | Was the extracted food item correct? |
| Prompt evaluation / A/B | ML engineer | Is this prompt better than that one? |

I set up the traces. The ML engineer attaches eval scores on top. These do not overlap.

### Integration in sayfit-alpha
Entry points are in `extractor.py` (`extract_items()`) and `reranker.py` (`rerank_single_item()`) — these are the functions that call `llm_client.get_client().chat.completions.create()`. Wrap those with Langfuse traces:

```python
from langfuse.decorators import observe, langfuse_context

# in step1_extraction/extractor.py
@observe()
def extract_items(text: str, date_time: str = "", uid: str = "", use_llm: bool = True) -> dict:
    langfuse_context.update_current_observation(
        name="step1_extraction",
        metadata={"model": llm_client.extraction_model()}
    )
    # existing body unchanged

# in step3_reranker/reranker.py
@observe()
def rerank_single_item(...) -> dict:
    langfuse_context.update_current_observation(
        name="step3_reranker",
        metadata={"model": llm_client.reasoning_model()}
    )
    # existing body unchanged
```

> **Note:** `extraction_model()` and `reasoning_model()` in `llm_client.py` are model name getters, not LLM callers — do not wrap those.

Wrap `run_pipeline()` with an outer `@observe()` so the full pipeline appears as one trace with per-step child spans.

### Step-by-step tasks

**Week 3 (Project Week 3) — Langfuse + Prometheus/Grafana**
1. Add Langfuse to `docker-compose.yml` (official self-hosted setup). Add `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` to GitHub secrets.
2. Add `@observe()` to `extract_items()` in `extractor.py` and `rerank_single_item()` in `reranker.py`
3. Add outer trace on `run_pipeline()` in `main.py`
4. Verify traces appear in Langfuse UI with latency + token counts per step
5. Set up prompt versioning for the extraction and reasoning prompts (Kevin owns this — confirm ML engineer is not doing the same)
6. Add Prometheus metrics to the FastAPI layer (request latency, error rate, step durations)
7. Set up Grafana dashboard for API-level monitoring
8. Add to ci.yml: `LANGFUSE_ENABLED=false` env var (SDK runs in no-op mode in CI — full Langfuse stack never started in CI) + curl `/metrics` step to verify Prometheus endpoint is exposed

---

## Monitoring layers — how the three tiers fit together

```
Tier 1 — CI (GitHub Actions + pytest)
  Trigger: every push / PR
  Question: does the code still run?
  Catches: broken imports, refactoring mistakes, missing modules, broken function signatures

Tier 2 — Langfuse (Kevin, operational)
  Trigger: every real pipeline run
  Question: is it running well?
  Catches: latency spikes, token cost increases, LLM errors

Tier 3 — Langfuse (ML engineer extends, quality)
  Trigger: every real pipeline run
  Question: are the outputs still good?
  Catches: extraction accuracy drops, prompt regressions
```

CI and Langfuse do not overlap. Tier 2 and Tier 3 use the same traces — Kevin sets them up, ML engineer extends them.

---

## GitHub Secrets — progressive rollout

| Secret | Add when | Why |
|---|---|---|
| `OPENAI_API_KEY` | Week 1 (now) | Prevents KeyError if any test hits the real API |
| `DB_PASSWORD` | Week 2 | Postgres container in CI |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Week 3 | Langfuse SDK init |

---

## Full Integration Order

| # | Task | Module | Dependency |
|---|---|---|---|
| 1 | CI scaffold (`conftest.py` + `ci.yml`) | MLOps | — |
| 2 | pytest suite (unit + integration) | SW Eng | CI live |
| 3 | Consolidate `food_portion_lookup.py` | SW Eng | Tests passing (safety net) |
| 4 | FastAPI layer (incl. PATCH for corrections) | SW Eng | Tests passing |
| 5 | Docker + docker-compose | SW Eng | FastAPI done |
| 6 | Add docker build + postgres service to ci.yml | MLOps | Docker done |
| 7 | Langfuse setup in docker-compose | MLOps | Docker done |
| 8 | LLM tracing in `extract_items()` + `rerank_single_item()` | MLOps | Langfuse running |
| 9 | Prometheus + Grafana | MLOps | FastAPI stable |

---

## What the Team Needs to Know

**Langfuse vs MLflow — one tool, shared instance**
MLflow is designed for classical ML experiment tracking (training runs, model registries). SayFit has no training runs. **My recommendation: use Langfuse as the shared observability platform.** I set it up and own the operational monitoring layer; the ML Engineering module extends it for evals and experiment tracking.

If the team is already committed to MLflow, the fallback is: MLflow for their experiments, Prometheus + Grafana only for my API monitoring. But Langfuse does everything MLflow does for an LLM pipeline and more.

**Trace naming — agree before writing integration code**
Langfuse traces need consistent names so both modules can query them. Before either side writes integration code, agree on:
- Top-level trace name (e.g. `pipeline_run`)
- Span names per step (`step1_extraction`, `step3_reranker`, `step4_scoring`)
- Metadata fields to attach (model name, user ID, input length, recipe DB version)

---

## TL;DR

| Tool | Decision | Why |
|---|---|---|
| **pytest** | Use | Ready-made fixtures; CI depends on it |
| **FastAPI** | Use | Zero changes to existing step modules |
| **Docker** | Use | `python:3.13-slim`; voice excluded cleanly |
| **Podman** | Mention in presentation | Better security model than Docker |
| **GitHub Actions** | Use — start Week 1 Day 1 | CI scaffold before any other code |
| **Langfuse** | Use — primary MLOps tool | Operational LLM tracing (latency, tokens, errors); ML engineer extends for quality |
| **Prometheus + Grafana** | Use | API infrastructure metrics; no overlap with Langfuse |
| **DVC** | Know it, mention it | Right tool for data pipeline versioning — module curriculum; not the capstone focus |
| **MLflow** | Drop | Classical ML tool; Langfuse covers everything relevant here |
| **Evidently** | Drop | Feature drift for classical ML; no fit for an LLM pipeline |
| **Litestar** | Skip | FastAPI is the better choice for this project size |
