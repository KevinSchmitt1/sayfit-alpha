"""
SayFit Alpha – Kaggle Recipe Importer (Step 6)
==============================================
One-time script: imports KaggleData/RAW_recipes.csv into a local DuckDB file
at data/kaggle_recipes.duckdb for fast querying.

Nutrition array in the CSV (indices):
  [kcal, fat%DV, sugar%DV, sodium%DV, protein%DV, sat_fat%DV, carbs%DV]

FDA % Daily Value reference amounts used for gram conversion:
  fat 78g  |  protein 50g  |  carbs 275g

Filtering during import:
  - Tag blacklist: excludes condiments, sauces, marinades, beverages, etc.
  - Calorie floor: kcal >= 100 (strips thin sauces/dressings not caught by tags)

Usage:
    python -m step6_recipe.recipe_kaggle_importer
    python -m step6_recipe.recipe_kaggle_importer --force
"""

import argparse
import ast
import csv
import json
import warnings

import duckdb

import config

_CSV_PATH      = config.ROOT_DIR / "KaggleData" / "RAW_recipes.csv"
_INGR_MAP_PATH = config.ROOT_DIR / "KaggleData" / "ingr_map.pkl"
_DB_PATH       = config.ROOT_DIR / "data" / "kaggle_recipes.duckdb"
_BATCH_SIZE    = 5_000

_ingr_lookup: dict | None = None


def _load_ingr_lookup() -> dict:
    """Load ingr_map.pkl into a {raw_ingr_lower: canonical} dict. Cached after first call."""
    global _ingr_lookup
    if _ingr_lookup is not None:
        return _ingr_lookup
    if not _INGR_MAP_PATH.exists():
        _ingr_lookup = {}
        return _ingr_lookup
    try:
        import pandas as pd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_pickle(_INGR_MAP_PATH)
        _ingr_lookup = {
            str(r).strip().lower(): str(c)
            for r, c in zip(df["raw_ingr"], df["replaced"])
        }
        print(f"  [INFO] Loaded ingredient map: {len(_ingr_lookup):,} entries")
    except Exception as e:
        print(f"  [WARN] Could not load ingr_map.pkl: {e} — skipping normalization")
        _ingr_lookup = {}
    return _ingr_lookup


def _normalize_ingredient(raw: str, lookup: dict) -> str:
    return lookup.get(raw.strip().lower(), raw)

_FAT_DV     = 78.0
_PROTEIN_DV = 50.0
_CARBS_DV   = 275.0

_MIN_KCAL = 100.0

_TAG_BLACKLIST = {
    "condiments-etc",
    "sauces",
    "savory-sauces",
    "sweet-sauces",
    "marinades-and-rubs",
    "salad-dressings",
    "spreads",
    "dips",
    "beverages",
    "cocktails",
    "candy",
    "cake-fillings-and-frostings",
    "cooking-mixes",
    "herb-and-spice-mixes",
    "canning",
    "jams-and-preserves",
    "jellies",
    "stocks",
    "garnishes",
}

_CREATE = """
CREATE TABLE IF NOT EXISTS kaggle_recipes (
    id                    INTEGER PRIMARY KEY,
    name                  TEXT    NOT NULL,
    minutes               INTEGER,
    tags                  TEXT,
    kcal                  REAL,
    protein_g             REAL,
    fat_g                 REAL,
    carbs_g               REAL,
    ingredients           TEXT,
    ingredients_canonical TEXT,
    n_ingredients         INTEGER,
    steps_json            TEXT
)
"""


def _parse_list(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def _parse_nutrition(raw: str):
    vals = _parse_list(raw)
    if len(vals) < 7:
        return None
    kcal      = float(vals[0])
    fat_g     = float(vals[1]) * _FAT_DV / 100.0
    protein_g = float(vals[4]) * _PROTEIN_DV / 100.0
    carbs_g   = float(vals[6]) * _CARBS_DV / 100.0
    return kcal, protein_g, fat_g, carbs_g


def import_csv(force: bool = False) -> int:
    """Import CSV into DuckDB. Returns number of rows inserted."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(_DB_PATH))
    try:
        conn.execute(_CREATE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_minutes ON kaggle_recipes(minutes)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kcal    ON kaggle_recipes(kcal)")

        if not force:
            count = conn.execute("SELECT COUNT(*) FROM kaggle_recipes").fetchone()[0]
            if count > 0:
                print(f"  [INFO] Kaggle DB already has {count:,} recipes. Pass --force to re-import.")
                return count

        # Drop and recreate so schema changes are always applied on force
        conn.execute("DROP TABLE IF EXISTS kaggle_recipes")
        conn.execute(_CREATE)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_minutes ON kaggle_recipes(minutes)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kcal    ON kaggle_recipes(kcal)")

        inserted = 0
        skipped  = 0
        batch    = []
        lookup   = _load_ingr_lookup()

        with open(_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nvals = _parse_nutrition(row["nutrition"])
                if nvals is None:
                    skipped += 1
                    continue
                kcal, protein_g, fat_g, carbs_g = nvals

                if kcal < _MIN_KCAL or kcal > 20_000:
                    skipped += 1
                    continue

                tags = _parse_list(row["tags"])
                if _TAG_BLACKLIST.intersection(tags):
                    skipped += 1
                    continue

                raw_name = row.get("name", "").strip()
                if not raw_name:
                    skipped += 1
                    continue

                raw_steps = _parse_list(row.get("steps", "[]"))
                if not raw_steps:
                    skipped += 1
                    continue

                raw_min = row.get("minutes", "")
                minutes = int(raw_min) if raw_min.isdigit() else None
                if minutes is not None and (minutes <= 0 or minutes > 300):
                    minutes = None

                ingredients           = _parse_list(row["ingredients"])
                ingredients_canonical = [_normalize_ingredient(i, lookup) for i in ingredients]
                steps                 = raw_steps
                n_ingr                = row.get("n_ingredients", "0")

                batch.append((
                    int(row["id"]),
                    raw_name.title(),
                    minutes,
                    json.dumps(tags),
                    round(kcal, 1),
                    round(protein_g, 1),
                    round(fat_g, 1),
                    round(carbs_g, 1),
                    json.dumps(ingredients),
                    json.dumps(ingredients_canonical),
                    int(n_ingr) if n_ingr.isdigit() else 0,
                    json.dumps(steps),
                ))

                if len(batch) >= _BATCH_SIZE:
                    conn.executemany(
                        "INSERT OR REPLACE INTO kaggle_recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    inserted += len(batch)
                    batch = []
                    print(f"  Imported {inserted:,} rows (skipped {skipped:,})...", end="\r", flush=True)

        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO kaggle_recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            inserted += len(batch)

        print(f"\n  Done — {inserted:,} recipes imported, {skipped:,} filtered out → {_DB_PATH}")
        return inserted
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Kaggle RAW_recipes.csv into DuckDB")
    parser.add_argument("--force", action="store_true", help="Re-import even if DB is already populated")
    args = parser.parse_args()
    import_csv(force=args.force)
