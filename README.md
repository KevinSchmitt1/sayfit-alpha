# SayFit Alpha

**Voice-to-nutrition food logger.** Tell SayFit what you ate — by text or voice — and get back matched foods, estimated portions, macros, and a saved meal history.

The app runs as a containerised REST API. The pipeline uses LLM extraction, ontology-guided retrieval, FAISS vector search, and a reranker to turn natural language into structured nutrition data stored in SQLite.

> **Status:** API + pipeline fully working. Frontend is not yet built — the API can be used directly via the Swagger UI or any HTTP client.

---

## How It Works

```
Text or audio input
        │
        ▼
┌───────────────────┐
│ Step 1: Extraction │  LLM parses food items, quantities, and categories
└────────┬──────────┘
         │
         ▼
┌────────────────────────┐
│ Step 1.5: Ontology     │  Category hints + portion size lookup
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Step 2: Retrieval      │  FAISS vector search over 80k+ foods (USDA + OFF)
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Step 3: Reranker       │  Picks best match, estimates grams, calculates macros
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Step 5: Database       │  Saves meal + items to SQLite
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Step 6: Recipe         │  Suggests recipes that fit remaining macro budget
│ Suggestions            │  Sources: Spoonacular API · local Kaggle DuckDB (179K recipes) · both
└────────────────────────┘
```

> Step 4 (terminal table formatter) is only active in CLI mode — the API returns JSON directly after Step 3.

---

## Before You Start

### 1. Get an API key

Groq is the default LLM backend — get a free key at [console.groq.com](https://console.groq.com).

Create a `.env` file in the repo root:

```bash
GROQ_API_KEY=your_key_here

# Optional — only needed if you want to use OpenAI instead of Groq
# OPENAI_API_KEY=your_key_here

# Optional — Langfuse cloud tracing (LLM call latency, token usage, errors)
# Free tier at langfuse.com is sufficient for development (~50k observations/month)
# LANGFUSE_PUBLIC_KEY=your_key
# LANGFUSE_SECRET_KEY=your_key

# Optional — Step 6 recipe suggestions via Spoonacular API
# Free tier: 150 calls/day. Without this, recipe step falls back to local Kaggle DB.
# SPOONACULAR_API_KEY=your_key
```

### 2. Get the FAISS index from sayfit-data-repo

The FAISS index is **not included in this repo** (too large for git) — it is built by the companion data pipeline:

**[KevinSchmitt1/sayfit-data-repo](https://github.com/KevinSchmitt1/sayfit-data-repo)**

That repo processes USDA FoodData Central + OpenFoodFacts, deduplicates entries, applies ontology labels, and outputs the index. Follow its README to run the pipeline, then copy the output here:

```
sayfit-alpha/
└── data/
    └── faiss_index/           ← from sayfit-data-repo  (REQUIRED)
        ├── food.index           ← FAISS vector index
        └── food_meta.pkl        ← metadata sidecar (item names, macros, categories)
```

> `food.index` and `food_meta.pkl` are always a matched pair from the same build — the FAISS index returns vector positions, and the pkl maps those positions back to the actual food rows.

All other data files (`food_ontology_300.json`, `portion_defaults.json`, `combined_final.csv`, `usda_final.csv`) ship with this repo and require no action.

---

## Running with Docker

This is the primary way to run SayFit.

```bash
docker compose up --build
```

> The first build takes several minutes — `openai-whisper` pulls PyTorch (~1.5 GB) and `ffmpeg` is installed as a system package. Subsequent builds are fast (layers are cached).

On startup the container:
1. Opens (or creates) `data/sayfit_meals.db` and runs `CREATE TABLE IF NOT EXISTS` for all tables
2. Checks that `data/faiss_index/food.index` exists — exits with a clear error if not
3. Starts the API server on port 8000

The FAISS index is mounted **read-only** inside the container (`./data/faiss_index:/app/data/faiss_index:ro`) — the pipeline cannot overwrite it.

### Available ports after `docker compose up`

| What | URL | What you can do there |
|------|-----|-----------------------|
| **FastAPI** | http://localhost:8000 | Send requests to the API |
| **Swagger UI** | http://localhost:8000/docs | Interactive API explorer — test all endpoints in the browser |
| **Prometheus metrics** | http://localhost:8000/metrics | Raw metrics scraped by Prometheus |
| **Prometheus UI** | http://localhost:9090 | Query metrics, inspect targets |
| **Grafana** | http://localhost:3001 | Dashboards — login: `admin` / `admin` |

---

## API Endpoints

Full interactive docs with request/response schemas: **http://localhost:8000/docs**

### Log a meal

```http
POST /log
Content-Type: application/json

{
  "uid": "user_001",
  "text": "i had 2 eggs and a banana for breakfast"
}
```

### Transcribe audio → text

```http
POST /transcribe
Content-Type: multipart/form-data

file: <audio file>   # WebM, Opus, WAV, MP3 — anything ffmpeg can decode
```

Returns `{ "text": "..." }`. Feed the result into `POST /log`.

### Meal history

```http
GET /meals/{uid}/today          # today's meals
GET /meals/{uid}?days=30        # last N days with macro totals
```

### Edit a logged meal

```http
PATCH  /meals/{uid}/items/{item_id}     # rescale a food item (recalculates macros)
POST   /meals/{uid}/items               # add a missed food item
DELETE /meals/{uid}/items/{item_id}     # remove a food item
DELETE /meals/{uid}/{meal_id}           # delete a whole meal
```

### Recipe suggestions (Step 6)

```http
POST /recipes/{uid}/suggest
Content-Type: application/json

{
  "source": "kaggle",        # "spoonacular", "kaggle", or "combo"
  "target_calories": 500,    # optional — defaults to full remaining budget
  "taste": "savory",         # optional: "savory", "sweet", "any"
  "max_time": 30             # optional: max prep time in minutes
}
```

```http
POST /recipes/{uid}/log-suggestion
```

Saves a chosen recipe suggestion to the daily meal log.

---

## Step 6 — Recipe Suggestions

After logging a meal, SayFit can suggest recipes that fit your **remaining macro budget** for the day.

Three data sources are supported:

| Flag | Source | Requires |
|------|--------|----------|
| `"spoonacular"` _(default)_ | Spoonacular REST API | `SPOONACULAR_API_KEY` in `.env` (free: 150 calls/day) |
| `"kaggle"` | Local DuckDB — 179K recipes, fully offline | One-time auto-download (~220 MB) |
| `"combo"` | Both merged, best 3 from combined pool | Both of the above |

### Kaggle recipe database

The Kaggle dataset is stored in a local DuckDB file (`data/kaggle_recipes.duckdb`). It is **not included in the repo** — it downloads itself automatically the first time it's needed:

```
Fetching recipe candidates from local Kaggle DB...
Downloading recipe database from GitHub Releases (~220 MB)...
This only happens once — subsequent runs use the local file.
100%  (220 MB)...
Done — saved to data/kaggle_recipes.duckdb
```

The import filters the raw 267K-row Kaggle dataset down to **179,654 standalone meals** by removing condiments, sauces, beverages, recipes under 100 kcal, and other non-meal categories.

### Data files overview

| File | Contents | How it gets there |
|------|----------|-------------------|
| `data/faiss_index/food.index` | FAISS vector index | Copy from sayfit-data-repo (required) |
| `data/faiss_index/food_meta.pkl` | Metadata sidecar — item names, macros, categories per vector | Copy from sayfit-data-repo (required, matched pair with food.index) |
| `data/food_ontology_300.json` | 300-category food ontology for portion hints | Included in this repo |
| `data/portion_defaults.json` | Fallback portion sizes | Included in this repo |
| `data/combined_final.csv` | Combined USDA + OFF food dataset | Included in this repo |
| `data/kaggle_recipes.duckdb` | 179K recipes with macros, ingredients, steps | Auto-downloaded on first use |
| `data/sayfit_meals.db` | User profiles + logged meals (SQLite) | Created automatically on startup |

---

## Observability

**Prometheus + Grafana** — request rate, error rate, and pipeline duration metrics are collected automatically via `prometheus-fastapi-instrumentator`. No configuration needed; dashboards are pre-loaded in Grafana at http://localhost:3001 (login: `admin` / `admin`).

**Langfuse** — LLM call tracing (latency, token usage, errors) for the extraction and reranker steps. Uses Langfuse cloud ([langfuse.com](https://langfuse.com)) — free tier is sufficient for development. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to `.env` to enable it. Tracing is silently skipped if the keys are absent.

---

## Local Development (without Docker)

```bash
# Python 3.14.x is the version used in this repo
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
PYTHONPATH=. uvicorn api.main:app --reload
```

### CLI mode

The original terminal pipeline still works for debugging and development:

```bash
python main.py                                        # interactive mode — type what you ate
python main.py --text "2 eggs and rice"               # single text input, no interaction
python main.py --record                               # record mic (default 15s) → transcribe
python main.py --record --duration 20                 # record with custom duration
python main.py --wav path/to/audio.wav                # transcribe an existing .wav file
python main.py --input inputs/my_meal.json            # file-based input (see format below)
python main.py --show-config                          # print active configuration and exit
```

**File input format** (`--input`):
```json
{
  "text": "i ate a pepperoni pizza and 3 eggs",
  "date_time": "2026-03-05T12:30:00",
  "UID": "user_001"
}
```

**LLM backend selection:**
```bash
python main.py --text "i ate a banana"          # default: Groq (GROQ_API_KEY)
python main.py --text "i ate a banana" --openai  # OpenAI (OPENAI_API_KEY)
python main.py --text "i ate a banana" --locllm  # Ollama-compatible local inference
python main.py --text "i ate a banana" --no-llm  # heuristic fallback, no API key needed
```

> Voice recording (`python main.py --record`) requires PortAudio and does **not** work inside Docker due to a macOS PortAudio/FAISS/OpenMP conflict. Use `POST /transcribe` for audio input when running the API.

---

## Running Individual Steps

Each pipeline step can run standalone for debugging or development:

```bash
# Step 0: record / transcribe audio
python -m step0_voice_input.run
python -m step0_voice_input.run --wav path/to/audio.wav

# Step 1: extract food items from text
python -m step1_extraction.run --input step1_extraction/example_input.json

# Step 1.5: apply ontology filtering
python -m step1_5_ontology_filter.ontology_filter --input step1_output.json

# Step 2: retrieve candidates
python -m step2_retrieval.run --input step2_retrieval/example_input.json

# Step 3: rerank retrieved candidates
python -m step3_reranker.run

# Step 4: format final output (CLI only)
python -m step4_output.run

# Step 5: database / persistence helpers
python -m step5_database.run
```

Each step has `example_input.json` and `example_output.json` in its directory defining its interface contract.

---

## User Calibration

The system learns from corrections made in CLI mode.

When a user corrects a vague item to a preferred amount (e.g. "a portion of noodles" → 500g), the system stores that preference in `data/calibrations/user_prefs.json`. Future runs reuse the stored amount automatically.

Calibrations are stored per user and per food item, and feed back into the reranker's portion estimation.

---

## Configuration

All settings live in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required for default mode) |
| `OPENAI_API_KEY` | — | OpenAI API key (optional) |
| `SPOONACULAR_API_KEY` | — | Spoonacular API key (optional, for recipe suggestions) |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key (optional, for LLM tracing) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key (optional, for LLM tracing) |
| `EXTRACTION_MODEL` | `llama-3.3-70b-versatile` | Groq extraction model |
| `REASONING_MODEL` | `llama-3.3-70b-versatile` | Groq reasoning model |
| `OPENAI_EXTRACTION_MODEL` | `gpt-4o-mini` | OpenAI extraction model |
| `OPENAI_REASONING_MODEL` | `gpt-4o-mini` | OpenAI reasoning model |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Local model for `--locllm` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `EMBEDDING_BATCH_SIZE` | `512` | Embedding batch size |
| `TOP_K_CANDIDATES` | `20` | Candidates per query |
| `MULTI_QUERY_POOLING` | `true` | Use multiple query variants in retrieval |
| `ONTOLOGY_BOOST_RANK1` | `1.30` | Rank-1 category boost |
| `ONTOLOGY_BOOST_RANK2` | `1.10` | Rank-2 category boost |
| `ONTOLOGY_BOOST_RANK3` | `1.03` | Rank-3 category boost |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `WHISPER_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `WHISPER_RECORD_SECONDS` | `15` | Default mic recording duration |
| `WHISPER_TARGET_DB` | `-20` | Target dB RMS for audio normalisation |
| `SAYFIT_DEFAULT_USER` | `default_user` | Default suggested user ID |
| `SAYFIT_DEVMODE` | `0` | Verbose debug output |

---

## Project Structure

```
sayfit-alpha/
├── api/                              # FastAPI layer
│   ├── main.py                       # App, lifespan checks, core endpoints, Prometheus metrics
│   ├── schemas.py                    # Pydantic request/response models
│   ├── transcribe.py                 # POST /transcribe — Whisper transcription endpoint
│   └── recipes.py                    # POST /recipes/* — Step 6 recipe endpoints
│
├── step0_voice_input/                # Audio recording + Whisper transcription (CLI only)
├── step1_extraction/                 # LLM food item extraction
├── step1_5_ontology_filter/          # Category prediction + portion hint resolution
├── step2_retrieval/                  # FAISS candidate retrieval
├── step3_reranker/                   # Best match selection + macro calculation
├── step4_output/                     # Terminal table formatter (CLI only)
├── step5_database/                   # SQLite persistence layer
├── step6_recipe/                     # Recipe suggestion module
│
├── data/                             # Data files — see "Before You Start"
│   ├── combined_final.csv            # Combined USDA + OFF food dataset
│   ├── food_ontology_300.json        # 300-category food ontology
│   ├── portion_defaults.json         # Fallback portion sizes
│   ├── faiss_index/                  # FAISS index — from sayfit-data-repo (required)
│   ├── kaggle_recipes.duckdb         # Recipe DB — auto-downloaded on first use
│   ├── calibrations/                 # Per-user correction history
│   └── sayfit_meals.db               # SQLite meal log — created on startup
│
├── tests/
│   ├── unit/                         # CI: mocked retriever, no FAISS needed
│   └── integration/                  # Local only: requires real FAISS index + API key
│
├── prometheus/
│   └── prometheus.yml                # Prometheus scrape config
│
├── main.py                           # CLI entry point
├── config.py                         # Paths, env vars, feature flags
├── llm_client.py                     # Groq / OpenAI / Ollama client factory
├── Dockerfile
├── docker-compose.yml                # API + Prometheus + Grafana
├── UI_README.md                      # Frontend integration guide (endpoints, TypeScript types)
└── .github/workflows/ci.yml          # CI: lint (ruff) + unit tests + docker build check
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Container exits on startup | Run `docker compose logs api` — almost always a missing FAISS index |
| `FAISS index not found` | Copy `data/faiss_index/` from sayfit-data-repo (see "Before You Start") |
| `food_meta.pkl` or `food.index` missing | Both files must be present in `data/faiss_index/` as a matched pair |
| `GROQ_API_KEY not set` | Add the key to `.env` |
| Whisper transcription slow | Set `WHISPER_MODEL=tiny` in `.env` |
| First `/transcribe` call is slow | Normal — Whisper loads into memory on first call (~5s). Subsequent calls are fast. |
| Wrong food matched | Use `PATCH /meals/{uid}/items/{item_id}` to correct the portion; CLI mode saves the correction for future runs |
| `source: "spoonacular"` returns 400 | Add `SPOONACULAR_API_KEY` to `.env`; use `"kaggle"` as a keyless fallback |
| Kaggle DB downloading every time | It should only download once to `data/kaggle_recipes.duckdb` — check write permissions on `data/` |
| Microphone not found (CLI) | Check that `sounddevice` can see your mic; voice recording does not work inside Docker |
