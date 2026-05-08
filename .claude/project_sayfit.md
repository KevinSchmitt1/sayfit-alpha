---
name: SayFit Projects Overview
description: Architecture, tech stack, and purpose of sayfit-alpha and sayfit-data-repo — Kevin's capstone AI project
type: project
---
# SayFit — Voice-to-Nutrition Food Logger

**Two repos:** `sayfit-alpha` (main app) + `sayfit-data-repo` (data pipeline)
**Location:** `/Users/kevin/Documents/bootcamp_ai/`

---

## sayfit-alpha

**Path:** `/Users/kevin/Documents/bootcamp_ai/sayfit-alpha`
**Purpose:** End-to-end modular pipeline that converts spoken food descriptions into structured nutrition logs.

### Pipeline (6 steps)

| Step | Module | Function |
|------|--------|----------|
| 0 | `step0_voice_input/` | Record mic or load .wav → OpenAI Whisper → `{text, date_time, UID}` JSON |
| 1 | `step1_extraction/` | LLM (Groq/OpenAI/Ollama) extracts structured food items with quantities, descriptions, category ranks |
| 1.5 | `step1_5_ontology_filter/` | Refines food categories using `food_ontology_300.json`; resolves portion hints from calibrations/ontology/defaults |
| 2 | `step2_retrieval/` | FAISS vector search over `combined_final.csv`; multi-query pooling, ontology boosts, name penalties |
| 3 | `step3_reranker/` | Picks best match, estimates grams, calculates nutrition from per-100g macros, surfaces confidence |
| 4 | `step4_output/` | Terminal table with totals; interactive review/correction/add/remove flow |
| 5 | `step5_database/` | SQLite (`data/sayfit_meals.db`) — users, meals, meal items, calibrations, daily goals |

### Key design decisions
- Each step is a standalone module (can be run independently with `python -m step<N>_*.run`)
- LLM backend swappable at runtime: default Groq, `--openai`, `--locllm` (Ollama), `--no-llm` (heuristic)
- Calibration/learning engine: user corrections stored in `data/calibrations/user_prefs.json`; reused on future runs
- Not a black box: interactive terminal review after every run

### Tech stack
- **Python 3.14**, `python-dotenv`, `openai` (OpenAI-compatible client for Groq too)
- **LLMs:** Groq `llama-3.3-70b-versatile` (default), GPT-4o-mini, Ollama `qwen2.5:7b`
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector search:** `faiss-cpu`
- **Data:** `pandas`, `numpy`
- **Voice:** `openai-whisper`, `sounddevice`, `scipy`
- **Persistence:** SQLite (stdlib `sqlite3`)
- **Formatting/linting:** `ruff`, `black`

### Data files
- `data/combined_final.csv` — combined USDA + OFF food dataset (indexed for FAISS)
- `data/food_ontology_300.json` — 300-category food ontology
- `data/portion_defaults.json` — fallback portion sizes
- `data/faiss_index/` — built FAISS index (run `python main.py --build-index`)
- `data/sayfit_meals.db` — SQLite meal history

### Entry point
```bash
python main.py                  # interactive mode
python main.py --text "..."     # direct text
python main.py --record         # mic input
python main.py --build-index    # rebuild FAISS index
```

---

## sayfit-data-repo

**Path:** `/Users/kevin/Documents/bootcamp_ai/sayfit-data-repo`
**GitHub:** `https://github.com/KevinSchmitt1/sayfit-data-repo.git`
**Purpose:** Builds the food databases consumed by sayfit-alpha. Jupyter-notebook-first data pipeline.

### Pipeline stages (notebooks)

1. **`data_acq.ipynb`** — data acquisition and initial preprocessing
2. **`eda_new.ipynb`** / **`building_ont_filter/eda_ontology.ipynb`** — EDA
3. **`OFF Ontology Data Cleaning.ipynb`** — cleans Open Food Facts (40,996 → 28,020 rows); no deduplication
4. **`USDA Deduplication Pipeline.ipynb`** — deduplicates USDA names via `rapidfuzz` blocking + BFS clustering
5. **`building_ont_filter/`** — builds the 300-category ontology mapping used by sayfit-alpha's filter step

### Raw data sources
- **USDA FoodData Central** (Dec 2025 release) — `raw_data/FoodData_Central_csv_2025-12-18/`
- **Open Food Facts** — `raw_data/en.openfoodfacts.org.products.csv`
- **USDA Branded Foods JSON** — `raw_data/FoodData_Central_branded_food_json_2025-12-18.json`

### Key outputs (fed into sayfit-alpha)
- `combined_final.csv` — merged, cleaned, deduplicated USDA + OFF dataset
- `usda_final.csv` — final USDA with ontology labels
- `off_data_clean2.csv` / `off_nutrition_clean.csv` — cleaned OFF data

### USDA deduplication approach
- `item_norm`: unicode→ASCII→lowercase→remove special chars→tokenise→dedup tokens→sort alphabetically
- Blocking: `(first_4_chars, token_count, first_char)` to avoid O(n²)
- Fuzzy: `rapidfuzz.fuzz.token_set_ratio` threshold 95
- Clustering: BFS over similarity graph → canonical name by (frequency, shortest, alphabetical)

### Tech stack
- **Python**, `pandas`, `numpy`, `duckdb` (local DB for large data processing)
- **`rapidfuzz`** — fuzzy matching for USDA dedup
- **`openai`** + `.env` — OpenAI-powered ontology labeling (optional steps)
- `jupyterlab`, `ipykernel`, `ipywidgets`, `matplotlib`

---

**Why:** SayFit is Kevin's AI capstone — demonstrates RAG, LLM integration, vector search, data engineering, and user-facing ML product design in a real app.
**How to apply:** sayfit-data-repo feeds sayfit-alpha. Changes to the food ontology or cleaned CSVs require rebuilding the FAISS index in sayfit-alpha (`python main.py --build-index`).
