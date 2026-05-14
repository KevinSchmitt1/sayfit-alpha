# SayFit Alpha — Team CLAUDE.md

> This project is in early development. Most components are still being planned or built.
> This document tells you what exists, where we are heading, and what not to do.
> Do not assume anything is built unless it is listed under "What exists right now."

---

## What this project is

SayFit is a voice-to-nutrition food logger. A user speaks or types a meal description.
The pipeline transcribes it, extracts structured food items via LLM, retrieves matching
foods from a nutrition database using vector search, calculates macros, and saves the result.
The system will eventually be accessible via a REST API and a web frontend.

---

## What exists right now

These are the only things that are actually built and working:

- **Pipeline steps 0–5** in `step0_voice_input/` through `step5_database/` — the core pipeline runs end-to-end via `python main.py`
- **CI** — `.github/workflows/ci.yml` runs `ruff check` + `pytest` on every push
- **Basic tests** — `tests/conftest.py` (retriever mock) + `tests/test_smoke.py`
- **SQLite database** — `data/sayfit_meals.db` via `step5_database/database.py`
- **FAISS index** — built from `data/combined_final.csv`, rebuilt via `python main.py --build-index`

Everything else — the API, the frontend, Docker, Langfuse, Prometheus, the data pipeline, the recipe module — is planned but does not exist yet.

---

## Where we are heading

The team is building toward this target state, in rough order:

1. **FastAPI layer** wrapping the existing pipeline (Kevin)
2. **Docker + docker-compose** packaging the API + database (Kevin)
3. **Next.js frontend** consuming the API (ML engineer)
4. **LLM observability** via Langfuse (Kevin)
5. **Recipe suggestions** as a new pipeline step (recipe module)
6. **Data pipeline** making `combined_final.csv` reproducible (data engineer)

Do not build ahead of this sequence without checking with the team.

---

## Hard constraints — do not violate these

**Step isolation.**
Every pipeline step (`step1_extraction`, `step2_retrieval`, etc.) must remain callable
standalone via `python -m stepN_*.run`. Steps must not import each other's internals.
Any new code that calls pipeline steps must go through the public function interfaces only.

**Macros are never invented.**
The LLM may only select from candidates retrieved by the FAISS step.
It must never output nutrition values it did not receive from retrieval.
Unknown foods get flagged to the user — they are never silently estimated.

**Voice input stays out of the API and Docker.**
`step0_voice_input` requires PortAudio and runs as an isolated subprocess in `main.py`
due to a macOS PortAudio/PyTorch/FAISS conflict. The API accepts text only.
Do not add PortAudio to the Dockerfile.

**No secrets in code.**
All credentials go in `.env`. Never hardcode API keys, passwords, or tokens.
If a new external service is added, add its key to `.env.example` immediately.

**Do not change inter-step JSON contracts.**
Each step has `example_input.json` and `example_output.json` defining its interface.
These contracts are the handshake between steps. Do not change output shapes without
updating all downstream steps and their tests.

---

## What is not decided yet — do not assume or implement

These decisions are open. If your work depends on one of them, raise it with the team first.

| Decision | Why it matters |
|----------|---------------|
| SQLite vs Postgres for the meals database | Affects API design, Docker topology, test strategy |
| DuckDB vs Postgres for the data pipeline | Affects whether data eng shares Kevin's DB container |
| Langfuse self-hosted vs cloud | Affects docker-compose complexity |
| API endpoint paths and response shapes | Kevin and ML engineer must agree before frontend consumes them |
| SSE vs request-response for pipeline runs | Affects how the frontend and API are wired together |
| Recipe API endpoints | Needed before frontend can surface recipes |
| FAISS index rebuild trigger | When data pipeline produces a new CSV, who rebuilds the index? |

---

## Do not use these tools

These were evaluated and ruled out for this project. Do not introduce them:

| Tool | Reason |
|------|--------|
| MLflow | Classical ML experiment tracking — no training runs in SayFit |
| Evidently | Feature drift monitoring for classical ML — no fit here |
| Litestar | FastAPI is the agreed choice |
| DVC (in sayfit-alpha) | May be used in sayfit-data-repo; not in this repo |

---

## Where to find module detail

Each team member maintains their own section document:

| File | Covers |
|------|--------|
| `.claude/plan/CLAUDE_sw_mlops.md` | API, Docker, CI/CD, testing, observability (Kevin) |
| `.claude/CLAUDE_ml_eng.md` | ML components, frontend (ML engineer) |
| `.claude/CLAUDE_data_engineering.md` | Data pipeline, storage, versioning (data engineer) |
| `.claude/CLAUDE_Recipe_Suggestions.md` | Recipe suggestion layer (recipe module) |
