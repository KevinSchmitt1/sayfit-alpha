# SayFit Alpha 🍽️

**Voice-to-nutrition food logger** – tell the system what you ate and get matched foods, estimated portions, macros, and a saved meal log.

SayFit uses a modular pipeline with LLM extraction, ontology-guided retrieval, FAISS vector search, reranking, and local persistence to convert natural language food descriptions into structured nutrition logs.

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd sayfit-alpha

# 2. Create & activate a virtual environment
# Python 3.14.x is the version used in this repo
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cat > .env <<'EOF'
GROQ_API_KEY=your_groq_api_key_here
EOF

# Optional: also supported
# OPENAI_API_KEY=your_openai_api_key_here

# 5. Make sure the main data files exist in data/
#    Expected: usda_final.csv, combined_final.csv,
#    food_ontology_300.json, portion_defaults.json

# 6. Build the FAISS index (one-time)
python main.py --build-index

# 7. Run the pipeline
python main.py
```

If you want to use the notebooks too:

```bash
pip install ipykernel jupyter matplotlib seaborn
```

---

## How It Works

```
Voice / Text / JSON Input
          │
          ▼
┌──────────────────────────────┐
│ Step 0: Voice Input          │   Record mic / load .wav → Whisper
│ "i ate 2 eggs and pizza"    │   → {text, date_time, UID}
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Step 1: Extraction (LLM)     │   Parse foods, quantities, descriptions,
│ Structured food items        │   category ranks, and queries
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Step 1.5: Ontology Filter    │   Refine categories and resolve
│ Category + portion hints     │   portion hints before retrieval
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Step 2: Retrieval (FAISS)    │   Vector search over indexed food rows
│ Top-K candidates             │   with category boosts and penalties
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Step 3: Reranker             │   Pick best match, estimate grams,
│ Portion + macro calculation  │   and calculate nutrition totals
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Step 4: Output               │   Render terminal table + totals,
│ Review & correction          │   allow user corrections
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Step 5: Database             │   Save meals, meal items,
│ SQLite persistence           │   calibrations, and user profiles
└──────────────────────────────┘
```

---

## What Is In The Repo

This repo now contains more than the original project brief in `structure.md`.

It includes:

- a full terminal-first pipeline in `main.py`
- standalone step modules for debugging and iteration
- ontology-guided retrieval and portion hinting
- multiple LLM backends (Groq, OpenAI, Ollama, heuristic fallback)
- correction / learning flow with persistent calibrations
- SQLite meal logging and user profile storage

### What is not in the repo:

## EDA and Data

There is also notebook-based work in another github repo:

`https://github.com/KevinSchmitt1/sayfit-data-repo.git` 

This includes the EDA of the project and a full description of the cycle on how to build the database, used for this app, out of the cleaned data (usda and openfoodfacts).

It also includes the building of the ontology filter (categories).


---

## Project Structure

```
sayfit-alpha/
├── main.py                          # Main CLI and full pipeline orchestration
├── config.py                        # Paths, env loading, model settings, feature flags
├── llm_client.py                    # Shared Groq / OpenAI / Ollama client factory
├── food_portion_lookup.py           # Deterministic portion lookup helper
├── requirements.txt
├── README.md
├── structure.md                     # Original project brief
├── updates.md                       # Notes on ontology boost changes
│
├── step0_voice_input/               # Audio recording + Whisper transcription
│   ├── voice_recorder.py
│   └── run.py
│
├── step1_extraction/                # LLM / heuristic extraction from raw text
│   ├── extractor.py
│   └── run.py
│
├── step1_5_ontology_filter/         # Category prediction + portion hint resolution
│   ├── ontology_filter.py
│   ├── food_portion_lookup.py
│   └── __init__.py
│
├── step2_retrieval/                 # FAISS build + candidate retrieval
│   ├── build_index.py
│   ├── retriever.py
│   └── run.py
│
├── step3_reranker/                  # Reranker + calibrations
│   ├── reranker.py
│   ├── calibration.py
│   └── run.py
│
├── step4_output/                    # Terminal table / totals formatting
│   ├── formatter.py
│   └── run.py
│
├── step5_database/                  # SQLite persistence layer
│   ├── database.py
│   └── run.py
│
├── docs/
│   └── data_cleaning.md             # Cleaning overview for source food data
│
├── data/                            # Data files, FAISS index, DB, calibrations
│   ├── usda_final.csv
│   ├── combined_final.csv
│   ├── food_ontology_300.json
│   ├── portion_defaults.json
│   ├── faiss_index/
│   ├── calibrations/
│   └── sayfit_meals.db
│
├── inputs/                          # Optional JSON inputs for file mode
├── input_tests/                     # Evaluation / test input assets
├── outputs/                         # Per-run outputs written by the pipeline
│
├── eda.ipynb
├── analyze_tests.py
└── food_dbs.zip
```

---

## Usage Modes

### Interactive Mode (default)

```bash
python main.py
```

Type what you ate, review the results, correct them if needed, and save the meal.

### Direct Text Mode

```bash
python main.py --text "i had 2 eggs and a banana for breakfast"
```

### Voice Recording Mode

```bash
python main.py --record
python main.py --record --duration 20
python main.py --wav path/to/audio.wav
```

### File Input Mode

```bash
python main.py --input inputs/my_meal.json
```

Input JSON format:

```json
{
  "text": "i ate a pepperoni pizza and 3 eggs",
  "date_time": "2026-03-05T12:30:00",
  "UID": "user_001"
}
```

### Build The Index

```bash
python main.py --build-index
```

### Show Configuration

```bash
python main.py --show-config
```

### Backend Selection

Groq is the default backend.

```bash
python main.py --text "i ate a banana"
python main.py --text "i ate a banana" --openai
python main.py --text "i ate a banana" --locllm
python main.py --text "i ate a banana" --no-llm
```

- default: Groq via `GROQ_API_KEY`
- `--openai`: OpenAI via `OPENAI_API_KEY`
- `--locllm`: Ollama-compatible local inference
- `--no-llm`: heuristic fallback mode

---

## Running Individual Steps

Each step can run standalone for debugging or development:

```bash
# Step 0: record / transcribe audio
python -m step0_voice_input.run
python -m step0_voice_input.run --wav path/to/audio.wav

# Step 1: extract food items from text
python -m step1_extraction.run --input step1_extraction/example_input.json

# Step 1.5: apply ontology filtering
python -m step1_5_ontology_filter.ontology_filter --input step1_output.json

# Step 2: build the FAISS index
python -m step2_retrieval.build_index

# Step 2: retrieve candidates
python -m step2_retrieval.run --input step2_retrieval/example_input.json

# Step 3: rerank retrieved candidates
python -m step3_reranker.run

# Step 4: format final output
python -m step4_output.run

# Step 5: database / persistence helpers
python -m step5_database.run
```

---

## Data And Retrieval

The retriever in this repo works over **structured food rows**, not freeform documents.

The current FAISS builder reads `data/combined_final.csv` and stores the index in `data/faiss_index/`.

### Main data files used by the repo

- `data/combined_final.csv` – combined indexable food dataset used by the current builder
- `data/food_ontology_300.json` – ontology for category / portion hints
- `data/portion_defaults.json` – fallback portion defaults

### Retrieval behavior

The current retriever includes:

- multi-query pooling
- name penalties for weak lexical matches
- ontology-based category boosts
- ranked L1 boosting (`rank1`, `rank2`, `rank3`)
- L2 boost when the top-ranked category is confirmed

This makes retrieval more realistic than plain cosine similarity on food names.

---

## Portion And Matching Logic

The repo currently resolves portions and matching through several layers.

### Extraction adds

- `item_name`
- `quantity_raw`
- `quantity_parsed`
- `unit_hint`
- `description`
- `category_ranks`

### Ontology filter adds

- category hints (`cat_l1`, `cat_l2`, ranked L1 list)
- portion hints resolved from:
  - user calibrations
  - ontology defaults
  - category-level defaults
  - flat portion defaults

### Reranker then does

- pick the best candidate
- use the resolved portion hint unless explicit grams / ml were said
- calculate final nutrition totals from per-100g macros

If the match is uncertain, confidence is surfaced in the output instead of silently pretending certainty.

---

## User Calibration (Learning Engine)

The system learns from corrections.

If a user keeps correcting the same vague item to a preferred amount, the system stores that preference in:

- `data/calibrations/user_prefs.json`

Example:

- `a portion of noodles` gets corrected to `500g`
- future runs can reuse that amount automatically

Calibrations are stored per user and per food item.

---

## Database / Persistence

The project also stores meals persistently in SQLite.

Current database location:

- `data/sayfit_meals.db`

The database layer stores:

- users
- meals
- meal items
- calibrations
- user profiles
- daily target fields / macro goals

So the repo is already set up for meal history and not just one-shot terminal calculations.

---


## Configuration

All settings are in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (optional, required for Groq mode) | Groq API key |
| `OPENAI_API_KEY` | (optional) | OpenAI API key |
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

## Output And Review Flow

The main CLI does not just print a number and exit.

After the pipeline runs, the user can:

- review a table of items + matches + grams + macros
- see total calories / protein / fat / carbs
- inspect low-confidence matches
- edit amounts
- remove items
- add missed foods
- save corrections for future runs

That makes the system much less of a black box and much easier to iterate on.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FAISS index not found` | Run `python main.py --build-index` |
| `food_meta.pkl` or `food.index` missing | Rebuild the index in `data/faiss_index/` |
| `GROQ_API_KEY not set` | Add your key to `.env` as `GROQ_API_KEY=...` |
| You want local testing without API calls | Use `--no-llm` or `--locllm` |
| `combined_final.csv` not found | Place it under `data/combined_final.csv` for the current index builder |
| Whisper too slow | Use a smaller model, e.g. `WHISPER_MODEL=tiny` |
| Microphone not found | Check that `sounddevice` can see your microphone |
| Wrong food matched | Correct it interactively so the calibration layer can learn |

---

## Summary

This repo contains a working, modular nutrition logging system with:

- voice, text, and JSON input modes
- LLM extraction
- ontology-guided retrieval
- reranking and portion estimation
- terminal review and correction
- calibration memory
- SQLite meal persistence
- notebook / EDA support

If you just want to use it, run `python main.py`.
If you want to debug or improve it, each step can be run on its own.