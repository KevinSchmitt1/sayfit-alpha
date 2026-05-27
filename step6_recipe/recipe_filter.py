"""
SayFit Alpha – Recipe Filter, Scaler & Scorer (Step 6)
=======================================================
Filters and ranks recipe candidates against the user's remaining macros.

Rules (in order):
  1. Hard discard — recipe protein is < 5g AND remaining protein > 30g
     (scaling can't rescue this; the food is structurally protein-free)

  2. Scale-up — if a recipe doesn't hit both coverage thresholds at 1 portion,
     calculate the minimum number of portions needed:
       min_scale = max(
           remaining_kcal * 0.80 / recipe_kcal,     # 80% kcal threshold
           remaining_protein * 0.60 / recipe_protein # 60% protein threshold
       )
     If min_scale <= MAX_SCALE (2.5), accept the recipe at that portion count.
     Otherwise discard.

  3. Soft score — weighted coverage across all four macros.
     Scores are computed on the *scaled* nutrition values.

Score weights:
  Protein  45%  |  Calories  30%  |  Carbs  15%  |  Fat  10%
"""

import math
from typing import List, Dict, Optional, Tuple


# ── Thresholds ────────────────────────────────────────────────────────────────

_MIN_KCAL_COVERAGE    = 0.80   # recipe (at final portion) must cover ≥ 80% of remaining kcal
_MIN_PROTEIN_COVERAGE = 0.60   # recipe (at final portion) must cover ≥ 60% of remaining protein
_PROTEIN_FLOOR_G      = 30.0   # only enforce protein threshold above this remaining amount
_MAX_SCALE            = 2.5    # discard if more than 2.5 portions needed to meet thresholds
_MAX_KCAL_OVER_BUDGET = 1.20   # discard if scaled recipe exceeds remaining kcal by more than 20%
_ABS_PROTEIN_MIN_G    = 5.0    # hard discard if recipe has < 5g protein (unrescuable by scaling)

# Score weights (must sum to 1.0)
_W_PROTEIN  = 0.45
_W_CALORIES = 0.30
_W_CARBS    = 0.15
_W_FAT      = 0.10


# ── Public API ────────────────────────────────────────────────────────────────

def filter_and_rank(recipes: List[Dict], remaining: Dict[str, float], top_n: int = 3) -> List[Dict]:
    """
    For each recipe: try to scale it to meet thresholds, score it, rank.
    Returns top_n recipes sorted by fit score (best first).
    """
    scored: List[Tuple[float, Dict]] = []

    for recipe in recipes:
        result = _evaluate(recipe, remaining)
        if result is not None:
            score, scaled_recipe = result
            scored.append((score, scaled_recipe))

    scored.sort(key=lambda t: t[0], reverse=True)

    output = []
    for score, recipe in scored[:top_n]:
        recipe["_fit_score"] = round(score * 100)
        output.append(recipe)

    return output


# ── Internal logic ────────────────────────────────────────────────────────────

def _evaluate(recipe: Dict, remaining: Dict[str, float]) -> Optional[Tuple[float, Dict]]:
    """
    Try to fit the recipe to the remaining macros.
    Returns (score, scaled_recipe) or None if the recipe should be discarded.
    """
    n = recipe.get("nutrition", {})
    r_cal     = remaining.get("calories", 0)
    r_protein = remaining.get("protein",  0)

    recipe_cal     = n.get("calories", 0)
    recipe_protein = n.get("protein",  0)

    # ── Hard discard: protein content too low to ever be useful ──────────────
    if r_protein > _PROTEIN_FLOOR_G and recipe_protein < _ABS_PROTEIN_MIN_G:
        return None

    # ── Calculate minimum scale needed to hit both thresholds ────────────────
    min_scale = 1.0

    if r_cal > 0 and recipe_cal > 0:
        needed = (r_cal * _MIN_KCAL_COVERAGE) / recipe_cal
        min_scale = max(min_scale, needed)

    if r_protein > _PROTEIN_FLOOR_G and recipe_protein > 0:
        needed = (r_protein * _MIN_PROTEIN_COVERAGE) / recipe_protein
        min_scale = max(min_scale, needed)

    # ── Discard if even at max scale thresholds can't be met ─────────────────
    if min_scale > _MAX_SCALE:
        return None

    # ── Discard if scaled recipe blows the calorie budget by more than 20% ───
    scale_check = math.ceil(min_scale * 4) / 4
    if r_cal > 0 and recipe_cal * scale_check > r_cal * _MAX_KCAL_OVER_BUDGET:
        return None

    # ── Round scale up to nearest 0.25 portion (e.g. 1.3 → 1.5) ─────────────
    scale = math.ceil(min_scale * 4) / 4  # guarantees thresholds are met

    # ── Build scaled recipe ───────────────────────────────────────────────────
    scaled = dict(recipe)
    if scale != 1.0:
        scaled["nutrition"] = {
            "calories": recipe_cal                  * scale,
            "protein":  recipe_protein              * scale,
            "fat":      n.get("fat",   0)           * scale,
            "carbs":    n.get("carbs", 0)           * scale,
        }
        scaled["_portions"] = scale
    else:
        scaled["_portions"] = 1.0

    # ── Soft score on scaled values ───────────────────────────────────────────
    score = _score(scaled["nutrition"], remaining)
    return score, scaled


def _score(nutrition: Dict, remaining: Dict[str, float]) -> float:
    def _cov(val: float, target: float) -> float:
        if target <= 0:
            return 1.0
        return min(val / target, 1.0)

    return (
        _cov(nutrition.get("protein",  0), remaining.get("protein",  0)) * _W_PROTEIN  +
        _cov(nutrition.get("calories", 0), remaining.get("calories", 0)) * _W_CALORIES +
        _cov(nutrition.get("carbs",    0), remaining.get("carbs",    0)) * _W_CARBS    +
        _cov(nutrition.get("fat",      0), remaining.get("fat",      0)) * _W_FAT
    )
