"""
Step 1.5 – Ontology Filter
===========================
Sits between Step 1 (Extraction) and Step 2 (Retrieval).

For each extracted food item it predicts the L1 and L2 food categories from
the ontology defined in data/usda_final.csv.  The predicted categories are
passed to Step 2 so the retriever can boost candidates that belong to the same
category – reducing noise and improving match quality.

Classification strategy (in priority order):
  1. Exact item_name lookup in usda_final.csv → real L1/L2/L3
  2. Keyword search: does a known item_name appear as substring of the query?
  3. L2 keyword scan: do any L2 category words appear in the query?
  4. L1 seed keywords (hard-coded): broad fallback
  5. "other" / "" for completely unknown items

Usage (as library):
    from step1_5_ontology_filter.ontology_filter import apply_ontology_filter
    filtered = apply_ontology_filter(extraction_output)

Standalone:
    python -m step1_5_ontology_filter.ontology_filter --input step1_out.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── lazy-loaded lookup structures ───────────────────────────────────────────
_loaded = False
# exact lookup: normalised item_name → (cat_l1, cat_l2, cat_l3)
_exact: dict[str, tuple[str, str, str]] = {}
# sorted list of (name_tokens, cat_l1, cat_l2, cat_l3) for substring search
_name_index: list[tuple[list[str], str, str, str]] = []
# l2 → l1 mapping built from data
_l2_to_l1: dict[str, str] = {}
# sorted list of (l2_tokens, l1) for l2 keyword scan
_l2_keywords: list[tuple[list[str], str, str]] = []


# ── fallback L1 seed keywords (for items completely outside the dataset) ─────
_L1_SEEDS: dict[str, list[str]] = {
    "dairy & eggs": [
        "milk", "cheese", "yogurt", "yoghurt", "butter", "cream", "egg", "eggs",
        "kefir", "quark", "whey", "casein", "curd", "ghee",
    ],
    "meat": [
        "beef", "pork", "lamb", "veal", "steak", "burger", "meatball", "sausage",
        "bacon", "ham", "salami", "pepperoni", "chorizo", "hot dog", "mince",
        "ground beef", "brisket", "ribs", "roast",
    ],
    "poultry": [
        "chicken", "turkey", "duck", "hen", "goose", "quail", "poultry",
        "chicken breast", "chicken thigh", "chicken wing",
    ],
    "fish & seafood": [
        "fish", "salmon", "tuna", "cod", "tilapia", "shrimp", "prawn",
        "lobster", "crab", "squid", "octopus", "herring", "mackerel",
        "sardine", "anchovy", "trout", "bass", "seafood", "mussels", "clam",
    ],
    "vegetables": [
        "carrot", "broccoli", "spinach", "lettuce", "tomato", "cucumber",
        "onion", "garlic", "pepper", "zucchini", "eggplant", "celery",
        "asparagus", "kale", "cabbage", "cauliflower", "peas", "corn",
        "beetroot", "radish", "leek", "artichoke", "arugula", "vegetable",
    ],
    "fruits": [
        "apple", "banana", "orange", "grape", "strawberry", "blueberry",
        "raspberry", "mango", "pineapple", "watermelon", "melon", "peach",
        "pear", "cherry", "plum", "kiwi", "avocado", "lemon", "lime",
        "grapefruit", "fig", "pomegranate", "papaya", "fruit",
    ],
    "grains & pasta": [
        "rice", "pasta", "noodle", "spaghetti", "penne", "macaroni",
        "oat", "oatmeal", "quinoa", "barley", "wheat", "flour", "cereal",
        "granola", "muesli", "couscous", "polenta", "bulgur",
    ],
    "baked goods": [
        "bread", "roll", "bun", "bagel", "croissant", "muffin", "cake",
        "cookie", "biscuit", "cracker", "wafer", "brownie", "donut",
        "pancake", "waffle", "scone", "pie", "tart", "pastry",
    ],
    "snacks": [
        "chip", "chips", "crisp", "pretzel", "popcorn", "nachos", "trail mix",
        "snack bar", "granola bar", "rice cake", "puff",
    ],
    "sweets & confectionery": [
        "chocolate", "candy", "sugar", "sweet", "gummy", "lollipop",
        "caramel", "toffee", "fudge", "marshmallow", "ice cream", "gelato",
        "sorbet", "pudding", "jelly",
    ],
    "beverages": [
        "water", "juice", "soda", "cola", "coffee", "tea", "latte",
        "cappuccino", "espresso", "smoothie", "shake", "beer", "wine",
        "spirits", "energy drink", "sports drink",
    ],
    "prepared & frozen meals": [
        "pizza", "lasagna", "lasagne", "casserole", "stew", "curry",
        "frozen meal", "ready meal", "instant noodle", "burrito", "wrap",
        "sandwich", "sub", "bowl",
    ],
    "condiments & sauces": [
        "ketchup", "mustard", "mayonnaise", "mayo", "sauce", "dressing",
        "vinegar", "soy sauce", "hot sauce", "salsa", "relish", "chutney",
        "hummus", "pesto", "guacamole", "jam", "honey", "syrup", "dip",
    ],
    "fats & oils": [
        "oil", "olive oil", "vegetable oil", "coconut oil", "lard",
        "margarine", "shortening",
    ],
    "legumes & beans": [
        "bean", "beans", "lentil", "lentils", "chickpea", "chickpeas",
        "tofu", "edamame", "soy", "peanut", "black bean", "kidney bean",
    ],
    "soups": [
        "soup", "broth", "stock", "bouillon", "bisque", "chowder",
        "minestrone", "ramen", "pho",
    ],
    "plant-based alternatives": [
        "plant-based", "vegan", "meat substitute", "soy milk", "almond milk",
        "oat milk", "tempeh", "seitan",
    ],
    "supplements": [
        "protein powder", "whey protein", "supplement", "vitamin",
        "mineral", "creatine", "bcaa", "omega", "probiotic", "collagen",
    ],
}


def _load() -> None:
    """Build all lookup structures from usda_final.csv (once)."""
    global _loaded, _exact, _name_index, _l2_to_l1, _l2_keywords

    if _loaded:
        return

    csv_path = config.USDA_FINAL_CSV
    if not csv_path.exists():
        print(f"⚠️  [Step 1.5] usda_final.csv not found at {csv_path} – using keyword fallback only")
        _loaded = True
        return

    df = pd.read_csv(csv_path, usecols=["item_name", "cat_l1", "cat_l2", "cat_l3"],
                     dtype=str, low_memory=False)
    df = df.dropna(subset=["item_name"])
    df[["cat_l1", "cat_l2", "cat_l3"]] = df[["cat_l1", "cat_l2", "cat_l3"]].fillna("")

    # 1. Exact name lookup (lowercase normalised)
    for _, row in df.iterrows():
        name = row["item_name"].strip().lower()
        _exact[name] = (row["cat_l1"], row["cat_l2"], row["cat_l3"])

    # 2. Name-substring index (sorted longest first for greedy matching)
    seen = set()
    entries = []
    for name, cats in _exact.items():
        if name in seen or not name:
            continue
        seen.add(name)
        entries.append((name.split(), cats[0], cats[1], cats[2]))
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    _name_index = entries

    # 3. L2 → L1 mapping + L2 keyword index
    l2_seen: set[str] = set()
    l2_entries = []
    for _, row in df[["cat_l1", "cat_l2"]].drop_duplicates().iterrows():
        l1, l2 = row["cat_l1"], row["cat_l2"]
        if not l2 or l2 in l2_seen:
            continue
        l2_seen.add(l2)
        _l2_to_l1[l2] = l1
        l2_entries.append((l2.split(), l1, l2))
    l2_entries.sort(key=lambda x: len(x[0]), reverse=True)
    _l2_keywords = l2_entries

    _loaded = True


def classify_item_name(item_name: str) -> tuple[str, str, str]:
    """
    Predict (cat_l1, cat_l2, cat_l3) for a food item name.

    Returns
    -------
    tuple[str, str, str]
        (cat_l1, cat_l2, cat_l3) — any level may be "" if unknown.
    """
    _load()
    name_lower = item_name.lower().strip()

    # 1 ── Exact match ──────────────────────────────────────────────────
    if name_lower in _exact:
        return _exact[name_lower]

    # 2 ── Name substring match (longest dataset name that fits in query) ─
    for tokens, l1, l2, l3 in _name_index:
        phrase = " ".join(tokens)
        if phrase in name_lower:
            return (l1, l2, l3)

    # 3 ── L2 keyword match ─────────────────────────────────────────────
    for tokens, l1, l2 in _l2_keywords:
        phrase = " ".join(tokens)
        if phrase in name_lower:
            return (l1, l2, "")

    # 4 ── L1 seed keyword match (longest phrase wins) ──────────────────
    best: tuple[int, str] | None = None
    for l1_cat, seeds in _L1_SEEDS.items():
        for seed in seeds:
            if seed in name_lower:
                if best is None or len(seed) > best[0]:
                    best = (len(seed), l1_cat)
    if best:
        return (best[1], "", "")

    return ("other", "", "")


def apply_ontology_filter(extraction: dict) -> dict:
    """
    Annotate each extracted item with predicted L1, L2, L3 food categories.

    Parameters
    ----------
    extraction : dict
        Output from Step 1 with keys "items" and "queries".

    Returns
    -------
    dict — original extraction extended with:
        "ontology": {
            "item1": {
                "item_name":        "egg",
                "predicted_cat_l1": "dairy & eggs",
                "predicted_cat_l2": "eggs",
                "predicted_cat_l3": "",
            }, ...
        },
        "category_hints": [          # aligned with queries list
            {"cat_l1": "dairy & eggs", "cat_l2": "eggs"}, ...
        ]
    """
    _load()
    print("\n🏷️  [Step 1.5] Applying ontology filter …")

    items: dict = extraction.get("items", {})
    queries: list = extraction.get("queries", [])

    ontology: dict[str, dict] = {}
    for key, item in items.items():
        name = item.get("item_name", "")
        l1, l2, l3 = classify_item_name(name)
        ontology[key] = {
            "item_name":        name,
            "predicted_cat_l1": l1,
            "predicted_cat_l2": l2,
            "predicted_cat_l3": l3,
        }
        print(f"   {name!r:30s} → L1={l1!r}  L2={l2!r}")

    # Build category_hints aligned with the queries list
    item_keys = list(items.keys())
    category_hints: list[dict] = []
    for q_idx in range(len(queries)):
        if q_idx < len(item_keys):
            ont = ontology[item_keys[q_idx]]
            category_hints.append({
                "cat_l1": ont["predicted_cat_l1"],
                "cat_l2": ont["predicted_cat_l2"],
            })
        else:
            category_hints.append({"cat_l1": "other", "cat_l2": ""})

    result = dict(extraction)
    result["ontology"] = ontology
    result["category_hints"] = category_hints

    print(f"   ✅ Ontology filter done – {len(ontology)} item(s) classified.")
    return result


# ── CLI usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1.5 – Ontology Filter")
    parser.add_argument("--input",  required=True, help="Path to step1_extraction_output.json")
    parser.add_argument("--output", help="Write annotated JSON here (default: stdout)")
    args = parser.parse_args()

    with open(args.input) as f:
        ext = json.load(f)

    result = apply_ontology_filter(ext)
    out_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out_json)
        print(f"   💾 Written to {args.output}")
    else:
        print(out_json)
