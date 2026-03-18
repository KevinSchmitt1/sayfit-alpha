"""
food_portion_lookup.py

Deterministic portion lookup for a voice nutrition app.

Assumption:
- item_name already comes from an upstream retriever / matcher.
- This function does NOT try to semantically match arbitrary food names.
- It resolves grams from:
    item_name + optional unit + optional quantity + optional modifier

Typical input example:
    {
        "item_name": "pepperoni pizza",
        "quantity": 1,
        "unit": "slice"
    }

or:
    {
        "item_name": "pepperoni pizza",
        "quantity": 1,
        "modifier": "half"
    }

or:
    {
        "item_name": "milk",
        "quantity": 2,
        "unit": "glass"
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


PORTION_MODIFIERS: Dict[str, float] = {
    "whole": 1.0,
    "full": 1.0,
    "half": 0.5,
    "halves": 0.5,
    "quarter": 0.25,
    "quarters": 0.25,
    "third": 1 / 3,
    "double": 2.0,
    "triple": 3.0,
}


UNIT_ALIASES: Dict[str, str] = {
    # generic
    "serving": "serving",
    "portion": "portion",
    "piece": "piece",
    "pieces": "piece",
    "pc": "piece",
    "pcs": "piece",
    "item": "piece",
    "items": "piece",
    "whole": "whole",
    "slice": "slice",
    "slices": "slice",
    "glass": "glass",
    "cup": "cup",
    "cups": "cup",
    "bowl": "bowl",
    "bowls": "bowl",
    "plate": "plate",
    "plates": "plate",
    "tbsp": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "can": "can",
    "cans": "can",
    "bottle": "bottle",
    "bottles": "bottle",
    "pot": "pot",
    "pots": "pot",
    "jar": "jar",
    "jars": "jar",
    "mug": "mug",
    "mugs": "mug",
    "scoop": "scoop",
    "scoops": "scoop",
    "handful": "handful",
    "handfuls": "handful",
    "clove": "clove",
    "cloves": "clove",
    "fillet": "fillet",
    "fillets": "fillet",
    "patty": "patty",
    "patties": "patty",
    "ball": "ball",
    "balls": "ball",
    "block": "block",
    "blocks": "block",
    "roll": "roll",
    "rolls": "roll",
    "ladle": "ladle",
    "ladles": "ladle",
    "packet": "packet",
    "packets": "packet",
    "packet_small": "packet",
    "packet_large": "packet",
    "wedge": "wedge",
    "wedges": "wedge",
    "ring": "ring",
    "rings": "ring",
    "stalk": "stalk",
    "stalks": "stalk",
    "head": "head",
    "heads": "head",
    "bulb": "bulb",
    "bulbs": "bulb",
    "ear": "ear",
    "ears": "ear",
    "shot": "shot",
    "shots": "shot",
    "shake": "shake",
    "tail": "tail",
    # frequent special keys from ontology
    "bottle small": "bottle_small",
    "small bottle": "bottle_small",
    "bottle_large": "bottle_large",
    "large bottle": "bottle_large",
    "bottle large": "bottle_large",
    "small bag": "small_bag",
    "large bag": "large_bag",
    "small portion": "small_portion",
    "medium portion": "medium_portion",
    "large portion": "large_portion",
    "side": "side",
    "large bowl": "large_bowl",
    "small box": "small_box",
    "double shot": "double_shot",
    "cup cooked": "cup_cooked",
    "bowl cooked": "bowl_cooked",
    "plate cooked": "plate_cooked",
    "cup dry": "cup",
    "cup chopped": "cup_chopped",
    "cup sliced": "cup_sliced",
    "cup diced": "cup_diced",
    "cup cubed": "cup_cubed",
    "cup shredded": "cup_shredded",
    "cup raw": "cup_raw",
    "cup cooked rice": "cup",
    "cup arils": "cup_arils",
    "serving 2": "serving_2",
    "serving 3": "serving_3",
    "serving 4": "serving_4",
    "serving 5": "serving_5",
    "serving 6": "serving_6",
    "stack 2": "stack_2",
    "stack 3": "stack_3",
    "half can": "half_can",
}


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def load_ontology(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_food_index(ontology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Builds a deterministic lookup index from:
    - food_id
    - label
    - aliases
    """
    index: Dict[str, Dict[str, Any]] = {}

    for food in ontology["foods"]:
        food_id = food.get("food_id") or food.get("item_id", "")
        candidates = {food_id, food.get("label", ""), *food.get("aliases", [])}
        for candidate in candidates:
            if not candidate:
                continue
            index[_normalize_text(candidate)] = food

    return index


def resolve_food(item_name: str, food_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    key = _normalize_text(item_name)
    if key not in food_index:
        raise KeyError(f"Unknown item_name: {item_name!r}")
    return food_index[key]


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None

    u = _normalize_text(unit)
    return UNIT_ALIASES.get(u, u)


def resolve_unit_grams(food: Dict[str, Any], unit: Optional[str]) -> float:
    """
    Priority:
    1. exact food-specific common_units match
    2. food default_unit
    3. error
    """
    common_units = food.get("common_units", {})
    default_unit = food["default_unit"]

    normalized_unit = normalize_unit(unit)

    if normalized_unit is None:
        return float(food["default_grams"])

    if normalized_unit in common_units:
        return float(common_units[normalized_unit])

    if normalized_unit == default_unit:
        return float(food["default_grams"])

    raise KeyError(
        f"Unit {unit!r} not defined for food {food['label']!r}. "
        f"Available units: {sorted(common_units.keys())}"
    )


def resolve_multiplier(
    quantity: Optional[float] = 1.0,
    modifier: Optional[str] = None,
) -> float:
    """
    quantity and modifier multiply together.

    Examples:
    - quantity=2, modifier=None      -> 2.0
    - quantity=1, modifier='half'    -> 0.5
    - quantity=2, modifier='half'    -> 1.0
    """
    q = 1.0 if quantity is None else float(quantity)

    if modifier is None:
        return q

    m_key = _normalize_text(modifier)
    if m_key not in PORTION_MODIFIERS:
        raise KeyError(f"Unknown modifier: {modifier!r}")

    return q * PORTION_MODIFIERS[m_key]


def lookup_portion_grams(
    entry: Dict[str, Any],
    food_index: Dict[str, Dict[str, Any]],
    *,
    round_result: bool = True,
) -> Dict[str, Any]:
    """
    Expected entry shape:
        {
            "item_name": "pepperoni pizza",
            "quantity": 1,
            "unit": "slice",
            "modifier": None
        }

    Minimal valid:
        {
            "item_name": "banana"
        }
    """
    if "item_name" not in entry or not entry["item_name"]:
        raise ValueError("entry must include a non-empty 'item_name'")

    item_name = entry["item_name"]
    quantity = entry.get("quantity", 1)
    unit = entry.get("unit")
    modifier = entry.get("modifier")

    food = resolve_food(item_name, food_index)
    unit_grams = resolve_unit_grams(food, unit)
    multiplier = resolve_multiplier(quantity=quantity, modifier=modifier)
    total_grams = unit_grams * multiplier

    if round_result:
        total_grams = round(total_grams, 2)

    return {
        "item_name_input": item_name,
        "matched_food_id": food.get("food_id") or food.get("item_id", ""),
        "matched_label": food["label"],
        "category": food["category"],
        "unit_requested": unit,
        "unit_resolved": normalize_unit(unit) or food["default_unit"],
        "quantity": quantity if quantity is not None else 1,
        "modifier": modifier,
        "unit_grams": unit_grams,
        "total_grams": total_grams,
    }


def lookup_portion_grams_safe(
    entry: Dict[str, Any],
    food_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Non-throwing wrapper for API usage.
    """
    try:
        result = lookup_portion_grams(entry, food_index)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e), "input": entry}


if __name__ == "__main__":
    ontology_path = Path("/mnt/data/food_ontology_300.json")
    ontology = load_ontology(ontology_path)
    food_index = build_food_index(ontology)

    demo_entries = [
        {"item_name": "pepperoni pizza", "quantity": 1, "unit": "slice"},
        {"item_name": "pepperoni pizza", "modifier": "half"},
        {"item_name": "milk", "quantity": 2, "unit": "glass"},
        {"item_name": "banana"},
        {"item_name": "pasta cooked", "unit": "plate"},
        {"item_name": "egg", "quantity": 3, "unit": "whole"},
    ]

    for entry in demo_entries:
        print(lookup_portion_grams_safe(entry, food_index))
