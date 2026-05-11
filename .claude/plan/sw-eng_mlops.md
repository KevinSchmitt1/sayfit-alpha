# Capstone 2 Plan: Kevin's Modules
## Software Engineering & APIs + MLOps & Monitoring

**Modules:** SW Engineering & APIs (curriculum weeks 1–2) · MLOps & Monitoring (curriculum weeks 4–5)
**Project timeline:** 3 project weeks (Week 1 = tests + API; Week 2 = Docker + CI/CD foundation; Week 3 = Langfuse + Prometheus/Grafana)
**Project:** sayfit-alpha

---

## My Working Points (Ownership)

| Area | What I'm building |
|---|---|
| **API layer** | FastAPI wrapper around the existing pipeline |
| **Test suite** | pytest coverage for all steps + integration |
| **Containerization** | Docker image for the API path |
| **LLM observability** | Langfuse tracing for every LLM call in the pipeline |
| **CI/CD** | GitHub Actions: lint + tests on every push |
| **API metrics** | Prometheus + Grafana for request-level monitoring |

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
- **Docker** — Python 3.14 has no official slim image yet; `python:3.13-slim` works with all current deps. Voice input stays CLI-only (PortAudio doesn't work cleanly in containers — this is already isolated via subprocess).

### Step-by-step tasks

**Week 1 (Project Week 1) — Tests first**
1. Write unit tests for each step using existing fixtures
2. Write integration test for the full `run_pipeline()` flow
3. Verify 80%+ coverage
4. Fix `food_portion_lookup.py` duplication (exists at root and under `step1_5/`) — under test cover now

**Week 2 (Project Week 2) — API + Docker**
5. Design the API correction model (see note below)
6. Create `api/` directory with FastAPI app
7. Add endpoints: `POST /log`, `GET /meals/{uid}/today`, `GET /meals/{uid}`, `PATCH /meals/{uid}/items/{item_id}` (post-hoc correction)
8. Wire endpoints to existing step modules (no changes to steps)
9. Write `Dockerfile` using `python:3.13-slim`
10. Test API path in container; confirm voice input falls back gracefully
11. Add GitHub Actions workflow: `ruff`, `black`, pytest on every push — use mocked retriever for CI (FAISS index and sentence-transformers model are too large to load in CI; mock `step2_retrieval/retriever.py` with fixture data so tests run fast and reliably)

> **API correction design decision:** `run_pipeline()` includes `ask_user_corrections()` — an interactive terminal loop that can't exist in a REST API. Decision: `POST /log` returns results immediately (no interactive loop); corrections are handled via `PATCH /meals/{uid}/items/{item_id}` after the fact. The CLI interactive loop remains intact for terminal use; the API path skips it.

---

## Module 2: MLOps & Monitoring

### Tech stack
| Tool | Role |
|---|---|
| **Langfuse** | Primary MLOps tool — LLM tracing, prompt versioning, output quality monitoring |
| **Prometheus + Grafana** | API-level infrastructure metrics |
| **GitHub Actions** | CI/CD: lint + test on every push |
| **DVC** | *(module curriculum tool — see note below)* |

### On DVC vs Langfuse
The MLOps module curriculum includes **DVC** (Data Version Control) for data pipeline versioning and reproducibility. DVC is the right tool for tracking large datasets like `data/combined_final.csv` and the FAISS index across runs.

**For sayfit-alpha, my implementation focus is Langfuse instead of DVC.** Here's why:

SayFit is an LLM pipeline, not a classical ML training pipeline. The operational risks are:
- LLM call latency going up
- Token costs spiking
- Extraction quality degrading after a prompt or model change

DVC addresses none of these — it tracks data artifacts between training runs. Langfuse addresses all of them in production.

> DVC would be valuable if `sayfit-data-repo` needed reproducible rebuild pipelines. That's upstream infrastructure, not the focus of this capstone. It's worth knowing and presenting as the correct tool for that problem.

### Why Langfuse
- **Built specifically for LLM pipelines** — not a classical ML tool being stretched to fit
- **Open-source and self-hostable** — can run locally via Docker Compose alongside the API
- **Covers three things at once:** tracing, prompt versioning, and output quality monitoring
- **Low integration cost** — the `@observe()` decorator wraps existing functions with no structural changes
- **Bridges into the team's ML Engineering module** — whoever owns eval/experiment tracking can extend the same Langfuse instance

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
1. Add Langfuse to `docker-compose.yml` (official self-hosted setup)
2. Add `@observe()` to `extract_items()` in `extractor.py` and `rerank_single_item()` in `reranker.py`
3. Add outer trace on `run_pipeline()` in `main.py`
4. Verify traces appear in Langfuse UI with latency + token counts per step
5. Set up prompt versioning for the extraction and reasoning prompts
6. Add Prometheus metrics to the FastAPI layer (request latency, error rate, step durations)
7. Set up Grafana dashboard for API-level monitoring
8. Document what Langfuse monitors vs what Prometheus monitors (no overlap)

---

## Full Integration Order

| # | Task | Module | Dependency |
|---|---|---|---|
| 1 | pytest suite (unit + integration) | SW Eng | — |
| 2 | Consolidate `food_portion_lookup.py` | SW Eng | Tests passing (safety net) |
| 3 | FastAPI layer (incl. PATCH for corrections) | SW Eng | Tests passing |
| 4 | Docker | SW Eng | FastAPI done |
| 5 | GitHub Actions CI (with mocked retriever) | MLOps | Tests + Docker |
| 6 | Langfuse setup | MLOps | FastAPI done |
| 7 | LLM tracing in `extract_items()` + `rerank_single_item()` | MLOps | Langfuse running |
| 8 | Prometheus + Grafana | MLOps | FastAPI stable |

---

## What the Team Needs to Know

**Langfuse vs MLflow — one tool, shared instance**
MLflow is designed for classical ML experiment tracking (training runs, model registries). SayFit has no training runs. **My recommendation: use Langfuse as the shared observability platform.** I set it up and own the monitoring layer; the ML Engineering module can extend it for evals and experiment tracking.

If the team is already committed to MLflow, the fallback is: MLflow for their experiments, Prometheus + Grafana only for my API monitoring. But Langfuse does everything MLflow does for an LLM pipeline and more.

**Trace naming — agree before writing integration code**
Langfuse traces need consistent names so both modules can query them. Before either side writes integration code, agree on:
- Top-level trace name (e.g. `pipeline_run`)
- Span names per step (`step1_extraction`, `step3_reranker`, `step4_scoring`)
- Metadata fields to attach (model name, user ID, input length)

---

## TL;DR

| Tool | Decision | Why |
|---|---|---|
| **pytest** | Use | Ready-made fixtures; CI depends on it |
| **FastAPI** | Use | Zero changes to existing step modules |
| **Docker** | Use | `python:3.13-slim`; voice excluded cleanly |
| **Podman** | Mention in presentation | Better security model than Docker |
| **Langfuse** | Use — primary MLOps tool | Open-source LLM tracing + prompt versioning + quality monitoring |
| **Prometheus + Grafana** | Use | API infrastructure metrics; complements Langfuse |
| **GitHub Actions** | Use | Standard for a GitHub repo |
| **DVC** | Know it, mention it | Right tool for data pipeline versioning — module curriculum; not the capstone focus |
| **MLflow** | Drop | Classical ML tool; Langfuse covers everything relevant here |
| **Evidently** | Drop | Feature drift for classical ML; no fit for an LLM pipeline |
| **Litestar** | Skip | FastAPI is the better choice for this project size |
