# SayFit Alpha — Shared Foundation
> **This section belongs to the whole team.**
> It describes what the system is, how modules connect, and what contracts exist between them.
> Every agent working on any part of this project should read this first.
> Individual module details live in the CLAUDE_*.md files alongside this one.
>
> **Reading guide:** This document describes the **target state** — the full system as it will
> exist when all modules are complete. Not everything listed here is built yet.
> Check the status table below before assuming any component exists.

---

## Build Status

| Component | Status | Owner |
|-----------|--------|-------|
| Pipeline steps 0–5 | ✅ Built | ML engineer |
| `step6_recipe` | ⏳ Planned | Recipe module |
| `api.py` (FastAPI backend) | ⏳ Planned | Software engineer |
| `frontend/` (Next.js) | ⏳ Planned | ML engineer |
| CI/CD (GitHub Actions) | ✅ Built | MLOps |
| Docker + docker-compose | ⏳ Planned | Software engineer |
| Langfuse observability | ⏳ Planned | MLOps |
| Data pipeline (Prefect/dbt/DVC) | ⏳ Planned | Data engineer |

---

## What SayFit Does

SayFit is a voice-to-nutrition food logger. A user speaks or types a meal description. The system
extracts structured food items, retrieves matching foods from a nutrition database via vector search,
calculates macros, and logs the result. After logging, the system suggests recipes that cover the
user's remaining daily macro targets.

The pipeline runs both as a terminal CLI and as a web application (Next.js frontend + FastAPI backend).
All pipeline steps are standalone modules — each can be run independently.

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Next.js 16, React 19, TypeScript)                │
│  localhost:3000                                             │
│  Pages: / (meal logger) · /history · /profile              │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP + SSE
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend (FastAPI) — api.py                                 │
│  localhost:8000                                             │
└──┬──────────────┬──────────────┬──────────────┬────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
Pipeline      step5_database  step6_recipe  /metrics
(steps 0–4)   (meals, users)  (suggestions) (Prometheus)

┌─────────────────────── Pipeline ───────────────────────────┐
│                                                             │
│  step0_voice_input   → Whisper ASR → {text, date_time, uid}│
│         │                                                   │
│  step1_extraction    → LLM → structured food items         │
│         │                                                   │
│  step1_5_ontology    → SentenceTransformer → categories     │
│         │                                                   │
│  step2_retrieval     → FAISS → top-20 candidates           │
│         │          ↑                                        │
│         │    combined_final.csv (data engineering output)  │
│         │                                                   │
│  step3_reranker      → LLM → best match + grams + macros   │
│         │                                                   │
│  step4_output        → table + user review (CLI only)      │
│         │                                                   │
│  step5_database      → persist meals, items, calibrations  │
│         │                                                   │
│  step6_recipe        → Spoonacular + Kaggle → top 3 recipes│
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────── Data Pipeline ─────────────────────────┐
│  sayfit-data-repo (separate repo)                           │
│  USDA + OpenFoodFacts → clean → deduplicate → merge        │
│         │                                                   │
│         └─► combined_final.csv → FAISS index rebuild       │
│                                  (python main.py --build-index)
└─────────────────────────────────────────────────────────────┘
```

---

## How to Start the Full System

**Backend only (API):**
```bash
python api.py                          # FastAPI → http://localhost:8000
```

**Frontend + Backend:**
```bash
python api.py                          # terminal 1
cd frontend && npm install && npm run dev  # terminal 2 → http://localhost:3000
```

**CLI only (no API, no frontend):**
```bash
python main.py                         # interactive mode
python main.py --text "..."            # direct text
python main.py --no-llm                # heuristic mode, no API key needed
```

**Rebuild FAISS index (after data pipeline produces new combined_final.csv):**
```bash
python main.py --build-index
```

**Recipe suggester (CLI):**
```bash
python -m step6_recipe.recipe_suggester --source combo
```

---

## Inter-Module Contracts

### API → Frontend (SSE + REST)

The frontend connects to the FastAPI backend. All endpoints are in `api.py`.

**Streaming pipeline run:**
```
GET /stream?text=...&uid=...&llm=...
Content-Type: text/event-stream

SSE event types:
  { event: "step",   step: N, total: N, msg: string }
  { event: "detail", key: string, data: unknown }
  { event: "done",   results: NutritionItem[], totals: Totals }
  { event: "error",  msg: string }
```

**REST endpoints (currently implemented in api.py):**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/save` | Save pipeline results as a meal → returns `meal_id` |
| `GET` | `/today` | Daily totals for a user |
| `GET` | `/search` | Search a single food item → NutritionItem |
| `GET` | `/profile/{uid}` | Read user profile |
| `POST` | `/profile/{uid}` | Save user profile |
| `DELETE` | `/meal/{meal_id}` | Delete a meal |
| `DELETE` | `/meal/{meal_id}/item/{item_id}` | Delete a meal item |
| `PATCH` | `/meal/{meal_id}/item/{item_id}` | Update item grams |

> ⚠️ **ALIGNMENT NEEDED — additional endpoints not yet in api.py**
> The SW/MLOps plan includes endpoints for manually adding items post-save and
> for recipe suggestions. These need to be added to api.py before the frontend
> can use them. Agree on path naming before implementing.

### Data Engineering → Pipeline

The data pipeline (in `sayfit-data-repo`) produces one file that the pipeline depends on:

| File | Where used | What breaks if it changes |
|------|-----------|--------------------------|
| `data/combined_final.csv` | `step2_retrieval/build_index.py` | FAISS index must be rebuilt |

**Column contract — these names must stay stable:**

| Column | Used by |
|--------|---------|
| `text_for_embedding` | FAISS index builder |
| `food_name` | Retriever display + name penalty |
| `kcal`, `protein`, `fat`, `carbs` | Reranker macro calculation |
| `source` | `"usda"` or `"off"` — ontology boost logic |

> ⚠️ **ALIGNMENT NEEDED — no versioning agreed yet**
> When `combined_final.csv` changes, someone must rebuild the FAISS index manually.
> No automatic trigger exists. The data engineering DVC plan may address this —
> but until it does, any data pipeline update must be communicated to the team
> so the index is rebuilt before the next deployment.

### step5_database → step6_recipe

The recipe layer reads directly from `step5_database` functions:

```python
from step5_database.database import get_db

db = get_db()
totals = db.get_daily_totals(uid)       # powers remaining-macro calculation
profile = db.get_user_profile(uid)      # powers macro goal targets
```

These function signatures must not change without updating `step6_recipe`.

---

## Shared Environment Variables

All variables go in `.env` at the repo root. Never commit `.env` — use `.env.example`.

| Variable | Required by | Purpose |
|----------|------------|---------|
| `GROQ_API_KEY` | steps 1, 3, 6 | Default LLM backend |
| `OPENAI_API_KEY` | steps 0, 1, 3, 6 | OpenAI backend + Whisper |
| `SPOONACULAR_API_KEY` | step6_recipe | Recipe search API (150 calls/day free tier) |
| `NEXT_PUBLIC_API_URL` | frontend | Backend URL (default: `http://localhost:8000`) |
| `WHISPER_MODEL` | step0 | Whisper model size (default: `base`) |
| `LANGFUSE_PUBLIC_KEY` | api.py (planned) | LLM observability |
| `LANGFUSE_SECRET_KEY` | api.py (planned) | LLM observability |
| `LANGFUSE_ENABLED` | api.py (planned) | Set `false` in CI |

> ⚠️ **ALIGNMENT NEEDED — DB password not yet shared**
> If the team migrates from SQLite to Postgres, `DB_PASSWORD` and `DATABASE_URL`
> become required. No decision made yet — see open items below.

---

## Key Constraints (apply to everyone)

**Step isolation is a hard rule.**
Every step module (`step1_extraction`, `step2_retrieval`, etc.) must be callable standalone
via `python -m stepN_*.run`. Steps must not import each other's internals.
The API calls step functions directly — it does not shell out.

**Voice input runs in a subprocess.**
`step0_voice_input` requires PortAudio and runs isolated from `main.py` to avoid a macOS
PortAudio/PyTorch/FAISS semaphore conflict. The API and Docker image exclude voice input.
`POST` endpoints accept text only.

**Macros are never invented.**
The LLM may only select from retrieved FAISS candidates — it cannot output nutrition values
it didn't receive from the retrieval step. Unknown foods get flagged to the user.

**Spoonacular free tier is 150 calls/day.**
Use `--source kaggle` during development to avoid exhausting the quota.

---

## Open Alignment Items

These are unresolved decisions that affect more than one module.
**No one should write code that depends on these until the team agrees.**

| # | Decision | Who it blocks | Options |
|---|----------|--------------|---------|
| 1 | **SQLite → Postgres migration** — current system uses SQLite; Kevin's plan migrates to Postgres | Kevin (Docker), data engineer (storage choice) | Keep SQLite for now · Migrate to Postgres · Use both (SQLite for meals, DuckDB for data pipeline) |
| 2 | **Data engineering DB: DuckDB vs Postgres** — data engineer evaluating both | Kevin (docker-compose), data engineer | DuckDB (simpler, local) · Postgres (shared with app DB) |
| 3 | **Langfuse: self-hosted vs cloud** | Kevin (docker-compose), ML engineer (evals) | Self-hosted in docker-compose · Langfuse Cloud |
| 4 | **Langfuse trace names** — proposed: `pipeline_run` / `step1_extraction` / `step3_reranker` | Kevin (observability), ML engineer (evals) | Confirm or rename before either side implements |
| 5 | **Recipe API endpoints** — step6 has no REST API yet | ML engineer (frontend), Kevin (api.py) | Define paths before frontend tries to consume them |
| 6 | **FAISS rebuild trigger** — no automated process when combined_final.csv changes | Data engineer, Kevin | Manual announcement · DVC pipeline hook · CI step |
