"""
Step 1.5 – Ontology Filter
===========================
Sits between Step 1 (Extraction) and Step 2 (Retrieval).

For each extracted food item it predicts the L1 and L2 food categories from
the ontology defined in data/usda_final.csv.  The predicted categories are
passed to Step 2 so the retriever can boost candidates that belong to the same
category – reducing noise and improving match quality.

Classification strategy:
  LLM mode (normal):
    - L1: ranked list from Step 1 extractor (LLM output)
    - L2: semantic cosine similarity of item_name against all L2 labels,
          constrained to the LLM's predicted L1 buckets (no LLM call needed)
  Heuristic mode (--no-llm):
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _log(*args, **kwargs):
    """Print only when developer mode is active."""
    if config.DEV_MODE:
        print(*args, **kwargs)

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

# ── semantic L2 index (built once on first use) ──────────────────────────────
_l2_embed_loaded = False
_l2_labels: list[str] = []          # L2 strings in index order
_l2_l1s: list[str] = []             # matching L1 for each L2
_l2_vecs: np.ndarray | None = None  # shape (n_l2, dim), normalised
_embed_model = None                 # SentenceTransformer singleton
_food_index: dict | None = None     # food_ontology_300 lookup index (lazy)
_portion_defaults_ont: dict | None = None  # portion_defaults.json cache

# ── food_ontology_300 vector index (built once alongside _food_index) ────────
_food_ont_labels: list[str] = []        # canonical label for each entry
_food_ont_entries: list[dict] = []      # corresponding food dicts
_food_ont_vecs: np.ndarray | None = None  # (n, dim), normalised
_ONTOLOGY_SIM_THRESHOLD = 0.72          # minimum cosine sim to accept a fuzzy hit

# Import unit alias map from food_portion_lookup (used for unit normalisation)
try:
    from .food_portion_lookup import UNIT_ALIASES as _UNIT_ALIASES
except ImportError:
    _UNIT_ALIASES: dict = {}

# ── Category-level portion defaults ─────────────────────────────────────────
# Median adult serving sizes (grams) per L1 category and unit type.
# Used as Tier-3 fallback when a food is not in food_ontology_300 or user_prefs.
# Values derived from standard dietitian portion guides (BDA, USDA MyPlate).
_CATEGORY_PORTION_DEFAULTS: dict[str, dict[str, float]] = {
    # Composed dishes, frozen meals — hearty plate-sized portions
    "prepared & frozen meals": {
        "serving": 350, "portion": 350, "plate": 400, "bowl": 400,
        "half": 175, "can": 400, "packet": 300, "default": 350,
    },
    # Cooked pasta/rice — pasta triples in weight when cooked
    "grains & pasta": {
        "plate": 120, "bowl": 120, "cup": 50, "cup cooked": 200,
        "serving": 120, "portion": 120, "default": 120,
    },
    # Protein cuts — moderate portions, fillet-sized
    "meat": {
        "serving": 150, "portion": 150, "slice": 80, "fillet": 170,
        "steak": 225, "patty": 120, "piece": 120, "default": 150,
    },
    "poultry": {
        "serving": 150, "portion": 150, "breast": 175, "thigh": 120,
        "wing": 90, "piece": 120, "default": 150,
    },
    "fish & seafood": {
        "serving": 140, "fillet": 150, "portion": 140, "piece": 120, "default": 140,
    },
    # Vegetables — cup or bowl-based
    "vegetables": {
        "serving": 120, "cup": 120, "bowl": 200, "portion": 150,
        "handful": 60, "plate": 200, "default": 120,
    },
    # Fruits — piece or cup-based
    "fruits": {
        "serving": 150, "piece": 150, "cup": 150, "handful": 100,
        "bowl": 200, "default": 150,
    },
    # Beverages — glass or can-based
    "beverages": {
        "glass": 250, "cup": 250, "mug": 300, "bottle": 500,
        "can": 330, "shot": 30, "serving": 250, "default": 250,
    },
    # Dairy items vary widely — use conservative serving
    "dairy & eggs": {
        "glass": 250, "cup": 250, "slice": 30, "serving": 150,
        "portion": 150, "bowl": 200, "default": 150,
    },
    # Baked goods — slice or piece-based
    "baked goods": {
        "slice": 40, "piece": 80, "serving": 80, "roll": 50,
        "muffin": 100, "default": 80,
    },
    # Snacks — small handfuls or packets
    "snacks": {
        "serving": 30, "handful": 30, "packet": 25, "bag": 50,
        "bowl": 50, "default": 30,
    },
    # Confectionery — very small amounts
    "sweets & confectionery": {
        "serving": 40, "piece": 15, "scoop": 60, "bar": 50,
        "square": 10, "default": 40,
    },
    # Condiments — tablespoon-level
    "condiments & sauces": {
        "tbsp": 15, "tablespoon": 15, "tsp": 5, "teaspoon": 5,
        "serving": 15, "cup": 240, "default": 15,
    },
    # Fats — very small amounts
    "fats & oils": {
        "tbsp": 14, "tablespoon": 14, "tsp": 5, "teaspoon": 5,
        "serving": 14, "default": 14,
    },
    # Soups — bowl or cup-based
    "soups": {
        "bowl": 350, "cup": 240, "serving": 300, "plate": 400,
        "mug": 300, "default": 300,
    },
    # Legumes — cooked cup-based
    "legumes & beans": {
        "cup": 180, "serving": 150, "bowl": 200, "portion": 150, "default": 150,
    },
    # Plant-based — similar to dairy / meat substitutes
    "plant-based alternatives": {
        "glass": 240, "cup": 240, "serving": 150, "piece": 100,
        "portion": 150, "default": 150,
    },
    # Supplements — scoop / serving
    "supplements": {
        "scoop": 30, "serving": 30, "tbsp": 15, "tsp": 5, "default": 30,
    },
}


def _load_embed_model():
    """Return the embedding model — reuses the retriever's singleton if already
    loaded, otherwise loads it here (e.g. when Step 1.5 runs before Step 2)."""
    global _embed_model
    if _embed_model is None:
        # Prefer reusing the retriever's already-loaded model to avoid loading
        # two SentenceTransformer instances in the same process (segfault risk).
        try:
            from step2_retrieval.retriever import _model as retriever_model
            if retriever_model is not None:
                _embed_model = retriever_model
                return _embed_model
        except ImportError:
            pass
        from sentence_transformers import SentenceTransformer
        import os as _os
        # Suppress HuggingFace/transformers log noise via the logging API.
        # We deliberately avoid OS-level fd redirection (dup2 to /dev/null)
        # because that approach races with any background thread writing to
        # sys.stdout (e.g. the Spinner), causing a deadlock on macOS.
        import logging as _logging
        _hf_loggers = [
            "sentence_transformers", "transformers", "huggingface_hub",
            "filelock", "torch",
        ]
        _prev_levels = {n: _logging.getLogger(n).level for n in _hf_loggers}
        for n in _hf_loggers:
            _logging.getLogger(n).setLevel(_logging.ERROR)
        try:
            _embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        finally:
            for n, lv in _prev_levels.items():
                _logging.getLogger(n).setLevel(lv)
    return _embed_model


def _build_l2_embed_index() -> None:
    """Embed all known L2 labels once and store normalised vectors."""
    global _l2_embed_loaded, _l2_labels, _l2_l1s, _l2_vecs
    if _l2_embed_loaded:
        return
    _load()  # ensure _l2_to_l1 is populated
    if not _l2_to_l1:
        _l2_embed_loaded = True
        return

    labels = list(_l2_to_l1.keys())
    l1s = [_l2_to_l1[l] for l in labels]
    model = _load_embed_model()
    vecs = model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    _l2_labels = labels
    _l2_l1s = l1s
    _l2_vecs = np.array(vecs, dtype="float32")
    _l2_embed_loaded = True
    _log(f"   📐 [Step 1.5] L2 semantic index built – {len(labels)} categories")


def _load_food_index() -> dict:
    """Lazy-load and build the food_ontology_300 lookup index + vector index."""
    global _food_index, _food_ont_labels, _food_ont_entries, _food_ont_vecs
    if _food_index is not None:
        return _food_index
    ont_path = config.FOOD_ONTOLOGY_FILE
    if not ont_path.exists():
        _food_index = {}
        return _food_index
    with open(ont_path) as f:
        ontology = json.load(f)
    index: dict = {}
    # For the vector index use the canonical label of each food entry once
    labels: list[str] = []
    entries: list[dict] = []
    for food in ontology.get("foods", []):
        fid = food.get("food_id") or food.get("item_id", "")
        canonical = food.get("label", "") or fid
        keys = {fid, canonical, *food.get("aliases", [])}
        for k in keys:
            if k:
                index[k.strip().lower().replace("-", " ")] = food
        if canonical:
            labels.append(canonical.strip().lower().replace("-", " "))
            entries.append(food)
    _food_index = index
    # Build normalised vector index over canonical labels
    if labels:
        model = _load_embed_model()
        vecs = model.encode(labels, normalize_embeddings=True, show_progress_bar=False)
        _food_ont_labels = labels
        _food_ont_entries = entries
        _food_ont_vecs = np.array(vecs, dtype="float32")
        _log(f"   📐 [Step 1.5] Food ontology vector index built – {len(labels)} entries")
    return _food_index


def _fuzzy_food_entry(key: str) -> dict | None:
    """Return the best-matching food_ontology_300 entry for *key*.

    First tries exact dict lookup, then falls back to cosine similarity
    over the canonical-label vector index.  Returns None when the best
    similarity is below _ONTOLOGY_SIM_THRESHOLD.
    """
    index = _load_food_index()  # also builds vector index as side-effect
    if key in index:
        return index[key]
    if _food_ont_vecs is None or len(_food_ont_labels) == 0:
        return None
    model = _load_embed_model()
    q_vec = model.encode([key], normalize_embeddings=True,
                         show_progress_bar=False)[0].astype("float32")
    sims = _food_ont_vecs @ q_vec
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    _log(f"      [portion fuzzy] '{key}' → '{_food_ont_labels[best_idx]}' (sim={best_sim:.3f})")
    if best_sim >= _ONTOLOGY_SIM_THRESHOLD:
        return _food_ont_entries[best_idx]
    return None


def _load_user_prefs_for_uid(uid: str) -> dict:
    """Return item_name → pref_dict for the given uid from user_prefs.json."""
    try:
        with open(config.CALIBRATION_FILE) as f:
            return json.load(f).get(uid, {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _load_portion_defaults_flat() -> dict:
    """Load portion_defaults.json (flat item → {default_grams, unit})."""
    global _portion_defaults_ont
    if _portion_defaults_ont is not None:
        return _portion_defaults_ont
    try:
        with open(config.PORTION_DEFAULTS_FILE) as f:
            _portion_defaults_ont = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        _portion_defaults_ont = {}
    return _portion_defaults_ont


def resolve_portion_hint(
    item_name: str,
    quantity_parsed: float | None,
    unit_hint: str | None,
    uid: str,
    cat_l1: str,
) -> dict:
    """
    Resolve portion size in grams through a 4-tier priority chain.

    Tier 1  User preferences (user_prefs.json) — personalized serving sizes
    Tier 2  food_ontology_300 — 300 known foods with per-unit gram weights
    Tier 3  Category-level defaults — per L1 category + unit type
    Tier 4  portion_defaults.json — simple flat lookup
    Tier 5  100 g fallback

    quantity_parsed acts as a multiplier (e.g. 3 slices × 125 g = 375 g).
    Explicit gram specifications (unit_hint = "g"/"ml") bypass all tiers.

    Returns {"grams": float, "unit": str, "source": str}
    How to improve: make the search engine similar to step 2 (retriever)
    """
    multiplier = float(quantity_parsed) if quantity_parsed is not None else 1.0
    key = item_name.lower().strip().replace("-", " ")

    # Vague quantity words override the multiplier to ~0.2 of a default serving.
    # They are NOT units, so norm_unit stays None and the tier chain provides
    # the base gram value (food default or category default).
    _VAGUE_QUANTITY_MULTIPLIER = 0.2
    _VAGUE_QUANTITIES = {
        "some", "some of",
        "a bit", "a bit of",
        "a little", "a little bit", "a little bit of",
        "a few",
        "a couple", "a couple of",
    }
    _vague = unit_hint and unit_hint.lower().strip() in _VAGUE_QUANTITIES
    if _vague:
        multiplier = _VAGUE_QUANTITY_MULTIPLIER
        unit_hint  = None   # don't pass it to UNIT_ALIASES — treat as no unit

    # Explicit gram/ml spec — quantity_parsed IS the gram amount
    if unit_hint and unit_hint.lower().strip() in ("g", "gram", "grams", "ml"):
        grams = float(quantity_parsed) if quantity_parsed is not None else 100.0
        return {"grams": grams, "unit": unit_hint.lower(), "source": "explicit_grams"}

    # Normalize unit using UNIT_ALIASES from food_portion_lookup
    norm_unit: str | None = None
    if unit_hint:
        u = unit_hint.lower().strip()
        norm_unit = _UNIT_ALIASES.get(u, u)

    # ── Tier 1: User preferences ──────────────────────────────────────────
    if uid:
        user_prefs = _load_user_prefs_for_uid(uid)
        # Exact lookup first; fuzzy fallback via cosine sim over pref keys
        pref_match = user_prefs.get(key)
        if pref_match is None and user_prefs:
            # embed the query and all stored pref keys, pick best match
            pref_keys = list(user_prefs.keys())
            model = _load_embed_model()
            vecs = model.encode([key] + pref_keys, normalize_embeddings=True,
                                show_progress_bar=False).astype("float32")
            q_vec, pref_vecs = vecs[0], vecs[1:]
            sims = pref_vecs @ q_vec
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= _ONTOLOGY_SIM_THRESHOLD:
                pref_match = user_prefs[pref_keys[best_idx]]
        if pref_match is not None:
            pref_grams = float(pref_match.get("preferred_grams", 0))
            if pref_grams > 0:
                return {
                    "grams": round(pref_grams * multiplier, 1),
                    "unit": pref_match.get("preferred_unit", "g"),
                    "source": "user_pref",
                }

    # ── Tier 2: food_ontology_300 (exact + fuzzy vector search) ───────────
    food_entry = _fuzzy_food_entry(key)
    if food_entry:
        common_units = food_entry.get("common_units", {})
        if norm_unit and norm_unit in common_units:
            return {
                "grams": round(float(common_units[norm_unit]) * multiplier, 1),
                "unit": norm_unit,
                "source": "ontology_unit",
            }
        default_grams = float(food_entry.get("default_grams", 0))
        if default_grams > 0:
            return {
                "grams": round(default_grams * multiplier, 1),
                "unit": food_entry.get("default_unit", "g"),
                "source": "ontology_default",
            }

    # ── Tier 3: Category-level defaults ────────────────────────────────────
    cat_units = _CATEGORY_PORTION_DEFAULTS.get(cat_l1.lower().strip(), {})
    if cat_units:
        if norm_unit and norm_unit in cat_units:
            return {
                "grams": round(float(cat_units[norm_unit]) * multiplier, 1),
                "unit": norm_unit,
                "source": "category_unit",
            }
        cat_default = cat_units.get("default", 0)
        if cat_default > 0:
            return {
                "grams": round(float(cat_default) * multiplier, 1),
                "unit": "serving",
                "source": "category_default",
            }

    # ── Tier 4: portion_defaults.json ──────────────────────────────────────
    flat = _load_portion_defaults_flat()
    if key in flat:
        flat_grams = float(flat[key].get("default_grams", 0))
        if flat_grams > 0:
            return {
                "grams": round(flat_grams * multiplier, 1),
                "unit": flat[key].get("unit", "g"),
                "source": "portion_defaults",
            }

    # ── Tier 5: fallback ───────────────────────────────────────────────────
    return {"grams": round(100.0 * multiplier, 1), "unit": "g", "source": "fallback"}


def classify_l2_semantic(item_name: str, allowed_l1s: list[str]) -> str:
    """
    Find the best matching L2 category for item_name using cosine similarity,
    constrained to L2 labels whose L1 is in allowed_l1s.

    Returns "" if no L2 index is available or no candidates exist.
    """
    _build_l2_embed_index()
    if _l2_vecs is None or not allowed_l1s:
        return ""

    # restrict to L2s that belong to one of the allowed L1s
    allowed_set = {a.lower().strip() for a in allowed_l1s}
    mask = [i for i, l1 in enumerate(_l2_l1s) if l1.lower().strip() in allowed_set]
    if not mask:
        return ""

    model = _load_embed_model()
    item_vec = model.encode([item_name], normalize_embeddings=True,
                             show_progress_bar=False)[0].astype("float32")

    # cosine similarity (vectors already normalised → dot product)
    candidate_vecs = _l2_vecs[mask]          # (n_candidates, dim)
    sims = candidate_vecs @ item_vec         # (n_candidates,)
    best_local = int(np.argmax(sims))
    return _l2_labels[mask[best_local]]


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
        _log(f"⚠️  [Step 1.5] usda_final.csv not found at {csv_path} – using keyword fallback only")
        _loaded = True
        return

    available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in ["item_name", "cat_l1", "cat_l2", "cat_l3"] if c in available_cols]
    df = pd.read_csv(csv_path, usecols=usecols, dtype=str, low_memory=False)
    df = df.dropna(subset=["item_name"])
    for col in ["cat_l1", "cat_l2", "cat_l3"]:
        if col not in df.columns:
            df[col] = ""
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
    _log("\n🏷️  [Step 1.5] Applying ontology filter …")

    items: dict = extraction.get("items", {})
    queries: list = extraction.get("queries", [])

    ontology: dict[str, dict] = {}
    for key, item in items.items():
        name = item.get("item_name", "")
        llm_ranks: list[str] = item.get("category_ranks", [])

        if llm_ranks:
            # LLM-provided ranked categories — use directly, normalise strings
            ranked_l1 = [c.strip().lower() for c in llm_ranks if c.strip()]
            l1 = ranked_l1[0] if ranked_l1 else "other"
            # Semantic L2 lookup constrained to the LLM's predicted L1 buckets
            l2 = classify_l2_semantic(name, ranked_l1) if ranked_l1 else ""
            l3 = ""
            source = "llm"
        else:
            # Fallback: rule-based classification (heuristic mode / no LLM)
            l1, l2, l3 = classify_item_name(name)
            ranked_l1 = [l1] if l1 and l1 != "other" else []
            source = "rules"

        ontology[key] = {
            "item_name":        name,
            "predicted_cat_l1": l1,
            "predicted_cat_l2": l2,
            "predicted_cat_l3": l3,
            "ranked_l1":        ranked_l1,
            "source":           source,
        }
        rank_str = " > ".join(f"{r!r}" for r in ranked_l1) if ranked_l1 else "'other'"

        # Resolve portion hint (Tier 1-5 chain) and attach to both dicts
        qty_parsed = item.get("quantity_parsed")
        unit_h = item.get("unit_hint")
        item_uid = item.get("uid", "")
        portion_hint_result = resolve_portion_hint(name, qty_parsed, unit_h, item_uid, l1)
        ontology[key]["portion_hint"] = portion_hint_result
        items[key]["portion_hint"] = portion_hint_result

        ph = portion_hint_result
        ph_str = f"{ph['grams']}g [{ph['source']}]"
        _log(f"   {name!r:30s} [{source}] → {rank_str}  |  {ph_str}")

    # Build category_hints aligned with the queries list
    item_keys = list(items.keys())
    category_hints: list[dict] = []
    for q_idx in range(len(queries)):
        if q_idx < len(item_keys):
            ont = ontology[item_keys[q_idx]]
            category_hints.append({
                "cat_l1":    ont["predicted_cat_l1"],
                "cat_l2":    ont["predicted_cat_l2"],
                "ranked_l1": ont["ranked_l1"],
            })
        else:
            category_hints.append({"cat_l1": "other", "cat_l2": "", "ranked_l1": []})

    result = dict(extraction)
    result["ontology"] = ontology
    result["category_hints"] = category_hints

    _log(f"   ✅ Ontology filter done – {len(ontology)} item(s) classified.")
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
