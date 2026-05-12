# sayfit-alpha — Project Knowledge

Mined and authored by Claude Code on 2026-05-08.
Companion to `CLAUDE.md` (conventions/workflow) and `README.md` (user docs).

---

## Complete File Map + Learnings

### Root level

| File | Role & key details |
|------|--------------------|
| `main.py` | 1,226-line orchestrator + CLI. `Spinner` class (threading, suppressed in devmode). `run_pipeline()` chains steps 1–4 and writes per-step JSON to dated `outputs/` subdirs. `ask_user_corrections()` handles edit / remove / add items inline. `run_onboarding_survey()` collects bodyweight, sex, age, PAL, training to compute TDEE via Müller RMR formula; macros at 2 g protein/kg, 0.9 g fat/kg. `run_main_menu()` is the full interactive loop. Voice step runs in an **isolated subprocess** to avoid PortAudio/FAISS/OpenMP semaphore conflict on macOS. **Known: over the 800-line limit — split after tests exist.** |
| `config.py` | Paths, `.env` loading, all model names, ontology boost constants, batch sizes, `DEV_MODE` flag. Single source of truth for all tunables. |
| `llm_client.py` | Backend factory. `configure(use_local, use_openai)` sets global backend. `extraction_model()` / `reasoning_model()` return the active model name. Backends: Groq (default), OpenAI, Ollama, heuristic fallback. |
| `food_portion_lookup.py` | Deterministic portion lookup. **Duplicate** of `step1_5_ontology_filter/food_portion_lookup.py` — known debt, consolidate once tests exist. |
| `analyze_tests.py` | Batch analysis utility for test-folder runs. No pytest suite yet — `run_analysis(run_dirs, batch_dir, report_path)`. |
| `CLAUDE.md` | Primary Claude Code instructions — architecture, conventions, backlog, agent cheat sheet. Read this first each session. |
| `structure.md` | Original capstone brief — inter-step JSON schemas, RAG explanation, component responsibilities. Treat as inter-step contract reference. |
| `requirements.txt` | `openai`, `python-dotenv`, `faiss-cpu`, `sentence-transformers`, `pandas`, `numpy`, `sounddevice`, `scipy`, `openai-whisper`, `ruff`, `black` |
| `bugs.md` | Open bugs — check before starting new work. |
| `eda.ipynb` | EDA notebook for the food datasets. |

---

### `step0_voice_input/`

| File | Details |
|------|---------|
| `voice_recorder.py` | Records from mic via `sounddevice`. Normalizes audio to `WHISPER_TARGET_DB` (RMS). Transcribes via OpenAI Whisper. Returns `{text, date_time, UID}`. |
| `run.py` | Standalone runner: `--wav`, `--uid`, `--output`, `--duration` |
| `example_output.json` | `{text, date_time, UID}` contract |

**Note:** Always run in a subprocess from `main.py` to avoid the macOS PortAudio/PyTorch/FAISS semaphore conflict.

---

### `step1_extraction/`

| File | Details |
|------|---------|
| `extractor.py` | LLM (strict JSON schema) or heuristic. Per-item output: `item_name`, `quantity_raw`, `quantity_parsed`, `unit_hint`, `description` (processing degree), `category_ranks`. Also emits `queries` list for the retriever. |
| `run.py` | Standalone: `--input <json>` |
| `example_input.json` / `example_output.json` | Step contract reference |

---

### `step1_5_ontology_filter/`

| File | Details |
|------|---------|
| `ontology_filter.py` | Embeds item names, compares against `food_ontology_300.json` via cosine similarity. Assigns `cat_l1`, `cat_l2`, ranked L1 list. Resolves portion hints in priority order: user calibrations → ontology defaults → category defaults → flat `portion_defaults.json`. Entrypoint: `apply_ontology_filter(extraction)`. Call `_build_l2_embed_index()` and `_load_food_index()` first. |
| `food_portion_lookup.py` | Deterministic lookup helper (duplicate of root-level — consolidate). |

---

### `step2_retrieval/`

| File | Details |
|------|---------|
| `build_index.py` | Reads `data/combined_final.csv`, embeds `text_for_embedding` in batches (model: `all-MiniLM-L6-v2`, `EMBEDDING_BATCH_SIZE=512`), saves FAISS index + `food_meta.pkl` to `data/faiss_index/`. One-time build. |
| `retriever.py` | Multi-query pooling (generates query variants per item). L2 cosine search. Name penalty for weak lexical matches. Ontology boosts: rank1 ×1.30, rank2 ×1.10, rank3 ×1.03. L2 confirm boost when top category confirmed. Returns top-K candidates (`TOP_K_CANDIDATES=20`). |
| `run.py` | Standalone runner |

---

### `step3_reranker/`

| File | Details |
|------|---------|
| `reranker.py` | Picks best candidate. Uses resolved portion hint unless explicit g/ml spoken. Calculates nutrition from per-100g macros × portion grams. Surfaces `confidence: "low"` when uncertain. Entrypoints: `rerank_all()` (batch), `rerank_single_item()` (LLM), `rerank_single_item_heuristic()`. `_parse_quantity()` converts raw quantity strings to floats. |
| `calibration.py` | `save_user_correction(uid, item_name, grams_per_unit)` — writes to `data/calibrations/user_prefs.json`. |
| `run.py` | Standalone runner |

---

### `step4_output/`

| File | Details |
|------|---------|
| `formatter.py` | Renders terminal table: item name, matched name, grams, kcal, protein, fat, carbs. Daily totals row. Returns a formatted string (printed by caller). |
| `run.py` | Standalone runner |

---

### `step5_database/`

| File | Details |
|------|---------|
| `database.py` | SQLite layer (`data/sayfit_meals.db`). Tables: `users`, `meals`, `meal_items`, `calibrations`, `user_profiles`. Profile fields: weight_kg, age_years, PAL, training_met, training_hours_per_week, kcal_daily, protein_daily, fat_daily, carbs_daily, goal (maintain/lose/gain). Key methods: `save_pipeline_result()`, `get_daily_totals()`, `get_meals_for_day()`, `delete_meal()`, `delete_meal_item()`, `update_meal_item_grams()`, `get_user_profile()`, `save_user_profile()`. Singleton via `get_db()`. |
| `run.py` | Standalone runner |

---

### `data/`

| File/Dir | What it is |
|----------|-----------|
| `combined_final.csv` | Merged USDA + OFF food dataset — primary FAISS index source |
| `usda_final.csv` | Cleaned USDA generic foods with ontology labels |
| `food_ontology_300.json` | 300-category food ontology with portion hints and hierarchy |
| `portion_defaults.json` | Fallback portion sizes (last resort in resolution chain) |
| `faiss_index/food.index` | Built FAISS binary index |
| `faiss_index/food_meta.pkl` | Food row metadata aligned with FAISS index |
| `calibrations/user_prefs.json` | Per-user, per-item learned portion preferences |
| `calibrations/last_user.txt` | Persists last used UID across sessions |
| `sayfit_meals.db` | SQLite meal history — **currently tracked in git; likely should be gitignored** |

---

### `input_tests/`

| File | What it is |
|------|-----------|
| `SF_Tests_ENG.json` | Bundled English test set |
| `SF_Tests_DE.json` | Bundled German test set |
| `SayFit-Test_ENG_01.json` … `_29.json` | 29 individual English test cases for `--test-folder` batch evaluation |

---

### `inputs/`

| File | What it is |
|------|-----------|
| `test_audio.wav` | Sample audio for manual Step 0 testing |

---

### `outputs/`

Dated run directories (`2026-03-11` … `2026-03-27`). Each run contains:
- `step1_extraction_output.json`
- `step1_5_ontology_output.json`
- `step2_retrieval_output.json`
- `step3_reranker_output.json`
- `meta.json` (test-folder runs only)

---

## Key Architectural Facts

- **Step isolation is a hard rule.** Every step is callable standalone via `python -m stepN_*.run`. Never break inter-step independence.
- **RAG grounds the LLM.** Macros are never invented — only retrieved from USDA/OFF. Unknown → flag to user.
- **Multi-backend LLM.** Groq default, OpenAI (`--openai`), Ollama (`--locllm`), heuristic (`--no-llm`). `llm_client.py` factory must stay backend-agnostic.
- **Voice → subprocess.** Step 0 always runs isolated to prevent macOS PortAudio/PyTorch/FAISS crash.
- **Calibration is per-user per-item.** `data/calibrations/user_prefs.json` stores gram corrections; reused automatically next run.
- **No pytest suite yet.** `analyze_tests.py` + 29 test JSONs exist, but no pytest. High priority before tuning.
- **`main.py` is 1,226 lines** — over 800-line rule. Split after tests exist.
- **`food_portion_lookup.py` is duplicated** at root and in `step1_5_ontology_filter/`. Consolidate.
