# Recipe Suggestion Layer

> **Covers:** Kaggle dataset ingestion · Spoonacular API · macro-fit scoring · recipe filtering & ranking · dual-source merge
> **Merge target:** this file is one section of the shared team CLAUDE.md

---

## Goal

The Recipe Suggestion Layer closes the loop between what a user has already eaten and what they should eat next. After a meal is logged, the system computes the user's remaining daily macros (calories, protein, fat, carbs) and surfaces recipe candidates that best fill that gap — ranked by how well their nutritional profile covers what is still needed for the day.

The layer draws from two complementary sources: a local SQLite database built from the Kaggle Food.com dataset (~130K recipes, offline, no quota) and the Spoonacular API (cloud, richer metadata, quota-limited). Both sources feed the same filtering and scoring pipeline, which scales single-portion recipes up to 2.5× if needed to meet coverage thresholds, then ranks survivors by a weighted macro-fit score (protein-heavy). The top 3 results are presented as an ASCII recipe card in the terminal, including preparation steps where available.

The system is designed to be practically useful — not just nutritionally correct. User preferences (taste, prep time, on-hand ingredients) shape both the API query and the local DB query, so suggestions feel relevant rather than algorithmic.

---

## Tech Stack

| Tool | Role |
|------|------|
| **Spoonacular API** | Cloud recipe search via `complexSearch` endpoint — returns recipes with full nutrition data |
| **Kaggle RecipeDB** | Local recipe dataset (~130K recipes) stored in `data/kaggle_recipes.db` after one-time CSV import |
| **SQLite** | Backing store for Kaggle recipes — queried directly via `sqlite3` (no ORM) |
| **`step5_database`** | Provides `get_daily_totals()` and `get_user_profile()` — powers the remaining-macros calculation |
| **LLM (Groq / OpenAI / Ollama)** | Translates remaining macros + user preferences → Spoonacular API query params; falls back to heuristics if unavailable |
| **stdlib `urllib`** | All Spoonacular HTTP calls — no `requests` or `httpx` dependency |

---

## Running Locally

**Launch the recipe suggester (interactive CLI):**
```bash
python -m step6_recipe.recipe_suggester                        # defaults to spoonacular source
python -m step6_recipe.recipe_suggester --user my_user_id     # specify user
python -m step6_recipe.recipe_suggester --source kaggle        # local DB only, no API quota consumed
python -m step6_recipe.recipe_suggester --source combo         # merges both sources before ranking
```

**One-time Kaggle DB import (must run before using `--source kaggle` or `--source combo`):**
```bash
python -m step6_recipe.recipe_kaggle_importer                  # imports KaggleData/RAW_recipes.csv → data/kaggle_recipes.db
```

> ⚠️ **SETUP REQUIRED — Kaggle DB is not built automatically**
> `data/kaggle_recipes.db` (262 MB) is not committed to git and is not created by CI. Any developer or
> environment using `kaggle` or `combo` mode must run the importer manually first. The importer is
> idempotent (`force=False` skips if DB already exists). `RAW_recipes.csv` (294 MB, also gitignored)
> must be present in `KaggleData/` before running.

**Required env vars (`.env`):**
```
SPOONACULAR_API_KEY=...   # free tier: 150 points/day
GROQ_API_KEY=...          # or OPENAI_API_KEY — used only by recipe_query_builder.py
```

---

## Data Sources

### Spoonacular API

- **Endpoint:** `https://api.spoonacular.com/recipes/complexSearch`
- **Always-on params:** `number=8`, `addRecipeNutrition=true`, `addRecipeInformation=true`, `instructionsRequired=true`
- **Macro params (LLM-generated or heuristic):**

| Param | Logic |
|-------|-------|
| `maxCalories` | `remaining_kcal × 1.1`, rounded to nearest 10 |
| `minProtein` | `remaining_protein × 0.3` — only if 10g < remaining protein < 80g |
| `maxFat` | Only set if remaining fat < 30g |
| `maxCarbs` | Only set if remaining carbs < 50g |
| `type` | `"main course"` (savory) or `"dessert"` (sweet) based on user preference |
| `maxReadyTime` | Set from user preference (15 / 30 / 60 min) |
| `includeIngredients` | Comma-joined on-hand ingredients if user provided any |
| `maxIngredients` | `5` if user selected "few ingredients" mode |

- **Quota:** Free tier gives 150 points/day (1 point per `complexSearch` call). Returns HTTP 402/403 when exhausted; `recipe_fetcher.py` catches this and returns an empty list — the pipeline does not crash.

> ⚠️ **OPERATIONAL CONSTRAINT — Spoonacular free-tier quota**
> With 8 results per call and 150 calls/day, heavy testing against `--source spoonacular` will exhaust
> the quota quickly. Use `--source kaggle` during development to avoid burning API points.
> Quota resets daily. Upgrade to a paid plan before any production deployment.

### Kaggle RecipeDB (Local)

- **Source file:** `KaggleData/RAW_recipes.csv` (Food.com dataset, ~230K rows before filtering)
- **After import:** ~130K recipes in `data/kaggle_recipes.db`, filtered by:
  - Tag blacklist (condiments, beverages, candy, marinades — 22 blocked tags)
  - Calorie floor: drops recipes < 100 kcal
  - Calorie ceiling: drops recipes > 20,000 kcal
  - Time validation: drops recipes with 0 or > 1,440 minutes

**SQLite schema (`kaggle_recipes` table):**

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Original Kaggle recipe ID |
| `name` | TEXT | Recipe title |
| `minutes` | INTEGER | Total cooking time |
| `tags` | TEXT (JSON) | Used for taste filtering via `LIKE` |
| `kcal` | REAL | Indexed |
| `protein_g` | REAL | Converted from FDA % Daily Value (50g reference) |
| `fat_g` | REAL | Converted from FDA % Daily Value (78g reference) |
| `carbs_g` | REAL | Converted from FDA % Daily Value (275g reference) |
| `ingredients` | TEXT (JSON) | |
| `n_ingredients` | INTEGER | |
| `steps_json` | TEXT (JSON) | Unique to Kaggle — shown in terminal output |

**Nutrition conversion from the Kaggle CSV:**
```
fat_g     = (fat_%DV   × 78)  / 100
protein_g = (protein_%DV × 50) / 100
carbs_g   = (carbs_%DV  × 275) / 100
```

---

## Recipe Pipeline (Step 6)

```
remaining macros (step5_database)
        │
        ▼
recipe_query_builder.py   ← LLM converts macros + preferences → Spoonacular params
        │                     (falls back to heuristic if LLM unavailable)
        ├──► recipe_fetcher.py        → Spoonacular API (cloud)
        └──► recipe_kaggle_fetcher.py → SQLite local DB
                    │
                    ▼ (merged if --source combo)
            recipe_filter.py   ← hard discard → scale-up logic → soft scoring
                    │
                    ▼ top 3
            recipe_formatter.py  ← ASCII terminal output with macro-fit bars
```

### Filtering & Ranking (`recipe_filter.py`)

Three-stage evaluation applied to every candidate recipe:

1. **Hard discard** — recipe protein < 5g AND remaining protein > 30g (protein-free, unrescuable)
2. **Scale-up check** — find minimum portions (max 2.5×, rounded to nearest 0.25) to hit both:
   - 80% calorie coverage of remaining
   - 60% protein coverage of remaining
   Discard if scaled recipe overshoots remaining kcal by > 20%
3. **Soft score** — weighted macro coverage (0–100):

| Macro | Weight |
|-------|--------|
| Protein | 45% |
| Calories | 30% |
| Carbs | 15% |
| Fat | 10% |

Top 3 by score are returned. Scaled recipes include `_portions` and `_fit_score` fields.

### User Preferences (Interactive CLI)

| Preference | Options |
|------------|---------|
| Taste | Savory / Sweet / Any |
| Time | Quick (≤15m) / Moderate (≤30m) / Relaxed (≤60m) / No limit |
| Ingredients | Free-text list of on-hand ingredients (optional) |
| Complexity | Few ingredients (≤5) toggle |

---

## Module Map

| File | Lines | What it does |
|------|-------|--------------|
| `recipe_suggester.py` | 223 | Orchestrates the full flow — reads remaining macros, collects preferences, calls fetchers, filter, formatter |
| `recipe_query_builder.py` | 119 | LLM → Spoonacular API params; heuristic fallback |
| `recipe_fetcher.py` | 87 | Calls Spoonacular `complexSearch`; normalizes response |
| `recipe_kaggle_fetcher.py` | 111 | Queries `kaggle_recipes.db`; auto-triggers importer if DB missing |
| `recipe_kaggle_importer.py` | 197 | One-time CSV → SQLite import with filtering and batched inserts |
| `recipe_filter.py` | 141 | Hard discard → scale-up logic → weighted scoring → top N |
| `recipe_formatter.py` | 82 | ASCII terminal output — macro bars, portions, prep steps (Kaggle only) |
| `__init__.py` | 3 | Re-exports `run_recipe_suggester` |

---

## Testing

> ⚠️ **No tests exist yet for step6_recipe — all entries below are planned**
> The Kaggle DB import and Spoonacular calls must be mocked or bypassed for offline CI runs.
> Decide on fixture strategy (pre-built sample DB vs in-memory SQLite) before writing test_kaggle_*.py.

| File | What it tests |
|------|--------------|
| `tests/test_recipe_filter.py` | _(planned)_ unit tests for `filter_and_rank()` — hard discard, scale-up, score weights |
| `tests/test_recipe_query_builder.py` | _(planned)_ heuristic fallback path with known remaining macros; no LLM call |
| `tests/test_recipe_kaggle_importer.py` | _(planned)_ `_parse_nutrition()` with known input arrays; import filtering logic |
| `tests/test_recipe_kaggle_fetcher.py` | _(planned)_ query builder with in-memory SQLite fixture |
| `tests/test_recipe_fetcher.py` | _(planned)_ response parsing with mocked `urllib` responses |
| `tests/test_recipe_integration.py` | _(planned)_ full `run_recipe_suggester()` with mocked DB totals + mocked API |

**Rules that apply to all step6 tests:**
1. Never call `https://api.spoonacular.com` in tests — mock `urllib.request.urlopen` or patch `recipe_fetcher.fetch_recipes`.
2. Never read from `data/kaggle_recipes.db` in tests — use an in-memory SQLite DB seeded with a small fixture set.
3. Use `use_llm=False` / patch the LLM call to test the heuristic path in `recipe_query_builder.py`.

---

## Future Goals

### Personalised Feedback — Recipe Likes & Preference Learning

Users should be able to rate or "like" a suggested recipe directly from the terminal output. Liked recipes feed a preference signal that shifts the ranking function over time — a recipe a user has liked before (or one similar to it) should surface more often than a nutritionally equivalent but unfamiliar one.

**Planned mechanics:**
- A `recipe_likes` table in `data/sayfit_meals.db` records `(user_id, recipe_id, source, liked_at)`.
- The filter/ranking step queries this table at score time and applies a small boost (e.g. +10 score points) to previously liked recipes or recipes sharing tags/ingredients with liked ones.
- A corresponding dislike signal can suppress recipes the user has actively rejected.

**Outstanding decisions before implementation:**
- Whether the boost is static (fixed +N) or learned (frequency-weighted).
- How to handle Spoonacular recipe IDs across quota resets (IDs are stable, but results vary by query).
- UI: single-key input after results are displayed, or a follow-up prompt after the meal is logged.

---

### User-Submitted Recipes — Custom Recipe Database

Users should be able to add their own recipes to the local SQLite database so they appear alongside Kaggle and Spoonacular results. This is especially useful for home staples or culturally specific dishes that neither dataset covers well.

**Planned mechanics:**
- A `user_recipes` table in `data/kaggle_recipes.db` (or a separate `data/user_recipes.db`) stores the same schema as `kaggle_recipes` with an additional `created_by` column.
- A `recipe_add` CLI command walks the user through entering: name, cooking time, ingredients list, macros per portion, and optional preparation steps.
- Macro input can be guided — the user either enters values directly or provides a rough ingredient list and the system estimates via the existing nutrition pipeline.
- User recipes are tagged with `_source="user"` and flow through the same `recipe_filter.py` and `recipe_formatter.py` as all other sources.
- `--source user` flag to query only user-submitted recipes; `combo` mode automatically includes them.

**Outstanding decisions before implementation:**
- Whether user recipes live in the same DB as Kaggle recipes or a separate file (separate is safer — avoids accidental overwrites on re-import).
- Macro entry UX: free-form text parsed by LLM vs structured field-by-field prompts.
- Deduplication strategy if a user adds a recipe that already exists in the Kaggle dataset.
