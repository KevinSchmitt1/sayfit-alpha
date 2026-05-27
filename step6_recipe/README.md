# Step 6 – Recipe Suggester

Suggests recipes that fit a user's remaining daily macro budget.
Pulls candidates from the Spoonacular API, a local Kaggle recipe database (DuckDB), or both combined.

---

## Quick Start

```bash
# Default: Spoonacular API
python -m step6_recipe.recipe_suggester --user salvo

# Local Kaggle DB (offline, no API key needed)
python -m step6_recipe.recipe_suggester --user salvo --source kaggle

# Both sources combined (best coverage)
python -m step6_recipe.recipe_suggester --user salvo --source combo
```

---

## First-Time Setup

### 1. API Keys

Add to your `.env` file:

```
SPOONACULAR_API_KEY=your_key_here   # free tier: 150 calls/day
```

Spoonacular free tier: [spoonacular.com/food-api](https://spoonacular.com/food-api)

### 2. Kaggle Recipe Database

The database is downloaded automatically on first use — no manual steps needed.

```bash
# Just run the suggester. If the DB is missing it downloads itself (~220 MB, once).
python -m step6_recipe.recipe_suggester --source kaggle
```

Output on first run:
```
  Fetching recipe candidates from local Kaggle DB...
  Downloading recipe database from GitHub Releases (~220 MB)...
  This only happens once — subsequent runs use the local file.
  100%  (220 MB)...
  Done — 220 MB saved to data/kaggle_recipes.duckdb
  Kaggle DB returned 20 candidate(s).
```

After that the file is cached locally and all subsequent runs are instant.

**Tested and verified** — auto-download from [github.com/KevinSchmitt1/sayfit-data-repo](https://github.com/KevinSchmitt1/sayfit-data-repo) works end-to-end.

---

**Rebuilding the DB locally** (only needed after filter/schema changes):

```bash
# 1. Download RAW_recipes.csv from Kaggle and place it at:
#    sayfit-alpha/KaggleData/RAW_recipes.csv
#    https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

# 2. Rebuild
python -m step6_recipe.recipe_kaggle_importer --force

# 3. Upload the new file as a GitHub Release asset (see Maintainer Notes below)
```

---

## Sources

| Flag | Source | Requires |
|------|--------|----------|
| `--source spoonacular` | Spoonacular REST API (default) | `SPOONACULAR_API_KEY` in `.env` |
| `--source kaggle` | Local DuckDB (179K recipes, offline) | One-time import (see above) |
| `--source combo` | Both merged, best 3 from combined pool | Both of the above |

---

## How It Works

```
Daily target − consumed today = remaining macros
            ↓
  Calorie threshold checks (exit early if not enough left)
    ├── ≤ 0 kcal    → "Already at your daily goal"
    ├── 1–99 kcal   → "Basically done — have a piece of fruit or call it a day"
    └── 100–200 kcal → Quick options (shake, yogurt, eggs, cottage cheese)
            ↓
  User preferences (kcal target, taste, time, ingredients)
            ↓
  Fetch candidates
    ├── Spoonacular: LLM builds API query → REST call
    └── Kaggle:      SQL query on local DuckDB
            ↓
  Filter & rank by macro fit
    ├── Hard discard: < 5g protein when > 30g protein still needed
    ├── Scale-up: min portions to hit 80% kcal / 60% protein thresholds
    ├── Over-budget cap: discard if scaled recipe > 120% remaining kcal
    └── Score: weighted macro coverage (P 45% · kcal 30% · C 15% · F 10%)
            ↓
  Display top 3 with fit score, macros, ingredients, steps / URL
            ↓
  Optional: save chosen recipe to today's meal log
```

---

## Preferences (asked at runtime)

| # | Question | Options |
|---|----------|---------|
| 1 | Target calories for this meal | Number or Enter to use all remaining |
| 2 | Taste | Savory / Sweet / Any |
| 3 | Max prep time | 15 / 30 / 60 min / No limit |
| 4 | Ingredients on hand | Comma-separated list or skip |
| 5 | Few ingredients | Max 5 ingredients (y/n) |

After results are shown, you can optionally save one of the top 3 recipes directly to today's meal log (macros are recorded as a logged meal).

---

## Kaggle Dataset Filtering

During import the following are removed:

- **Tag blacklist** — condiments, sauces, marinades, beverages, cocktails, dressings, spreads, candy, stocks, and 10 more non-meal categories
- **Calorie floor** — recipes below 100 kcal (thin extracts, garnishes)
- **Prep time cap** — times above 5 hours stored as NULL (won't appear when a time filter is active)

Result: 179,654 standalone meals out of 267,000 raw rows.

---

## File Overview

```
step6_recipe/
├── recipe_suggester.py        # Entry point — orchestrates the full pipeline
├── recipe_query_builder.py    # LLM translates remaining macros → Spoonacular params
├── recipe_fetcher.py          # Spoonacular API client
├── recipe_kaggle_importer.py  # One-time CSV → DuckDB import
├── recipe_kaggle_fetcher.py   # DuckDB query client
├── recipe_filter.py           # Scaling, discard rules, and scoring
└── recipe_formatter.py        # ASCII output with macro bars and steps
```

---

## Database

| File | Contents | Git |
|------|----------|-----|
| `data/kaggle_recipes.duckdb` | 179K recipes with macros, ingredients, steps | Ignored — auto-downloaded on first run |
| `data/sayfit_meals.db` | User profiles and logged meals (SQLite) | Ignored |
| `KaggleData/RAW_recipes.csv` | Raw source data for local rebuild | Ignored |

---

## Maintainer Notes — Publishing a New DB Version

When the DuckDB file is rebuilt and ready to ship:

1. Go to [github.com/KevinSchmitt1/sayfit-data-repo/releases/new](https://github.com/KevinSchmitt1/sayfit-data-repo/releases/new)
2. Create a new tag: `recipe-db-v1` (increment for future versions: `v2`, `v3`, …)
3. Title: `Recipe Database v1`
4. Attach `data/kaggle_recipes.duckdb` as a binary asset
5. Publish the release

Then update the tag in `recipe_kaggle_fetcher.py` if the tag changes:
```python
_RELEASE_URL = (
    "https://github.com/KevinSchmitt1/sayfit-data-repo"
    "/releases/download/recipe-db-v1/kaggle_recipes.duckdb"  # ← update tag here
)
```
