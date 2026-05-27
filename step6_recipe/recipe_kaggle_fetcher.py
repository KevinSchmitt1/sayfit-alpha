"""
SayFit Alpha – Kaggle Local Recipe Fetcher (Step 6)
====================================================
Queries data/kaggle_recipes.db (built by recipe_kaggle_importer.py).
Returns candidates in the same dict schema as recipe_fetcher.py so that
recipe_filter.filter_and_rank works identically on both sources.
"""

import json
import urllib.request
import duckdb
from typing import Any, Dict, List

from step6_recipe.recipe_kaggle_importer import _DB_PATH, _load_ingr_lookup

_MAX_KCAL_FACTOR = 1.25

# Pre-built DB hosted as a GitHub Release asset.
# Update the tag below whenever a new version is published.
_RELEASE_URL = (
    "https://github.com/KevinSchmitt1/sayfit-data-repo"
    "/releases/download/recipe-db-v1/kaggle_recipes.duckdb"
)


def _download_db() -> bool:
    """Download the pre-built DuckDB from GitHub Releases. Returns True on success."""
    print("  Downloading recipe database from GitHub Releases (~220 MB)...")
    print("  This only happens once — subsequent runs use the local file.")

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _DB_PATH.with_suffix(".tmp")

    def _progress(block_count, block_size, total):
        done = block_count * block_size
        if total > 0:
            pct = min(done / total * 100, 100)
            print(f"  {pct:.0f}%  ({done / 1_048_576:.0f} MB)...", end="\r", flush=True)

    try:
        urllib.request.urlretrieve(_RELEASE_URL, str(tmp), reporthook=_progress)
        tmp.rename(_DB_PATH)
        print(f"\n  Done — {_DB_PATH.stat().st_size / 1_048_576:.0f} MB saved to {_DB_PATH}")
        return True
    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}")
        if tmp.exists():
            tmp.unlink()
        return False


def fetch_kaggle_recipes(
    preferences: Dict[str, Any],
    remaining_macros: Dict[str, float],
    n: int = 20,
) -> List[Dict]:
    """
    Return up to n recipe candidates from the local Kaggle DuckDB.
    On first run the DB is downloaded automatically from GitHub Releases.
    Falls back to local import instructions if download fails.
    """
    if not _DB_PATH.exists():
        if not _download_db():
            print("  [INFO] To build the DB locally instead:")
            print("         1. Place KaggleData/RAW_recipes.csv in the project root")
            print("         2. Run: python -m step6_recipe.recipe_kaggle_importer")
            return []

    conn = duckdb.connect(str(_DB_PATH), read_only=True)
    try:
        return _query(conn, preferences, remaining_macros, n)
    finally:
        conn.close()


# ── Internal ──────────────────────────────────────────────────────────────────

def _has_canonical_column(conn) -> bool:
    try:
        cols = {row[0] for row in conn.execute("DESCRIBE kaggle_recipes").fetchall()}
        return "ingredients_canonical" in cols
    except Exception:
        return False


def _query(conn, preferences: Dict, remaining: Dict, n: int) -> List[Dict]:
    cal = remaining.get("calories", 0)
    conditions: List[str] = []
    params: List = []

    # Loose calorie ceiling
    if cal > 0:
        conditions.append("kcal <= ?")
        params.append(cal * _MAX_KCAL_FACTOR)

    # Prep-time ceiling — when a limit is set, require a known time within it;
    # NULL (unknown or originally > 5h) is excluded when a limit is active.
    max_time = preferences.get("max_time_minutes")
    if max_time:
        conditions.append("minutes <= ?")
        params.append(max_time)

    # Few-ingredient filter
    if preferences.get("few_ingredients"):
        conditions.append("n_ingredients <= 5")

    # Taste via tag substring search
    taste = preferences.get("taste", "any")
    if taste == "savory":
        conditions.append(
            "(tags LIKE '%main-dish%' OR tags LIKE '%lunch%' OR tags LIKE '%dinner%')"
        )
    elif taste == "sweet":
        conditions.append("(tags LIKE '%desserts%' OR tags LIKE '%sweet%')")

    # Ingredient inclusion — use canonical column when available for accurate matching
    use_canonical = _has_canonical_column(conn)
    lookup = _load_ingr_lookup() if use_canonical else {}
    for ing in preferences.get("ingredients", []):
        clean = ing.strip().lower()
        if not clean:
            continue
        if use_canonical:
            canonical = lookup.get(clean, clean)
            conditions.append("ingredients_canonical LIKE ?")
            params.append(f'%"{canonical}"%')
        else:
            conditions.append("LOWER(ingredients) LIKE ?")
            params.append(f"%{clean}%")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = (
        "SELECT id, name, minutes, tags, kcal, protein_g, fat_g, carbs_g, "
        f"ingredients, n_ingredients, steps_json FROM kaggle_recipes {where} ORDER BY RANDOM() LIMIT ?"
    )
    params.append(n)

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_recipe(row) for row in rows]


def _row_to_recipe(row: tuple) -> Dict:
    rec_id, name, minutes, tags_json, kcal, protein_g, fat_g, carbs_g, ingr_json, n_ingr, steps_json = row
    try:
        ingredients = json.loads(ingr_json)
    except Exception:
        ingredients = []
    try:
        steps = json.loads(steps_json) if steps_json else []
    except Exception:
        steps = []

    return {
        "id":               rec_id,
        "title":            name,
        "ready_in_minutes": minutes,
        "source_url":       None,
        "nutrition": {
            "calories": kcal,
            "protein":  protein_g,
            "fat":      fat_g,
            "carbs":    carbs_g,
        },
        "ingredient_count": n_ingr,
        "ingredients":      ingredients,
        "steps":            steps,
        "_source":          "kaggle",
    }
