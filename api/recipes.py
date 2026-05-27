"""
SayFit Alpha – Recipe API Router
=================================
Two endpoints wired to step6_recipe functions:

  POST /recipes/{uid}/suggest      — fetch ranked recipe suggestions for remaining macros
  POST /recipes/{uid}/log-suggestion — save a chosen recipe to the daily meal log
"""

import datetime
import os
from typing import Optional

import fastapi

from api.schemas import (
    FoodItem,
    Meal,
    RecipeLogRequest,
    RecipeMacros,
    RecipeNutrition,
    RecipeSuggestion,
    RecipeSuggestRequest,
    RecipeSuggestResponse,
)
from step5_database.database import get_db
from step6_recipe.recipe_fetcher import fetch_recipes
from step6_recipe.recipe_filter import filter_and_rank
from step6_recipe.recipe_kaggle_fetcher import fetch_kaggle_recipes
from step6_recipe.recipe_query_builder import build_query_params
from step6_recipe.recipe_suggester import compute_remaining

router = fastapi.APIRouter()

_DONE_KCAL_THRESHOLD = 100   # below this: user is essentially at their goal
_LOW_KCAL_THRESHOLD  = 200   # below this: too little for a full recipe
_DEFAULT_MEAL_CAP    = 800   # when no target is set and remaining > this, cap at this
                             # prevents filter_and_rank from requiring an unrealistically
                             # large recipe (80% of 2000+ kcal) when the user hasn't eaten yet


def _scale_remaining(remaining: dict, target_calories: Optional[int]) -> dict:
    """Scale macro budget proportionally if the user wants a smaller meal.
    When no target is set, cap at _DEFAULT_MEAL_CAP so the filter thresholds
    stay reachable for users with a large remaining budget."""
    effective_target = target_calories or (
        _DEFAULT_MEAL_CAP if remaining["calories"] > _DEFAULT_MEAL_CAP else None
    )
    if not effective_target or effective_target >= remaining["calories"]:
        return remaining
    scale = target_calories / remaining["calories"]
    return {
        "calories": float(target_calories),
        "protein":  remaining["protein"] * scale,
        "fat":      remaining["fat"]     * scale,
        "carbs":    remaining["carbs"]   * scale,
    }


@router.post("/recipes/{uid}/suggest", response_model=RecipeSuggestResponse)
def suggest_recipes(uid: str, body: RecipeSuggestRequest):
    """
    Return up to 3 recipe suggestions ranked by macro fit against the user's
    remaining daily budget.  Preferences (taste, time, ingredients) are forwarded
    to the query builder and filter steps.
    """
    api_key = os.getenv("SPOONACULAR_API_KEY", "").strip()
    source = body.source

    if source in ("spoonacular", "combo") and not api_key:
        if source == "spoonacular":
            raise fastapi.HTTPException(
                status_code=400,
                detail="SPOONACULAR_API_KEY is not configured on this server.",
            )
        # combo without a key → fall back to kaggle-only silently
        source = "kaggle"

    remaining, _profile, _consumed = compute_remaining(uid)
    remaining_macros = RecipeMacros(**remaining)

    # Early exits when there is not enough budget left for a real recipe
    if remaining["calories"] <= 0:
        return RecipeSuggestResponse(
            remaining=remaining_macros,
            suggestions=[],
            message="You have already reached your daily calorie goal.",
        )
    if remaining["calories"] < _DONE_KCAL_THRESHOLD:
        return RecipeSuggestResponse(
            remaining=remaining_macros,
            suggestions=[],
            message="Only a few calories left — you're basically at your daily goal.",
        )
    if remaining["calories"] <= _LOW_KCAL_THRESHOLD:
        return RecipeSuggestResponse(
            remaining=remaining_macros,
            suggestions=[],
            message=(
                "Under 200 kcal remaining. Quick options: protein shake, "
                "Greek yogurt, cottage cheese, or hard-boiled eggs."
            ),
        )

    prefs = body.preferences.model_dump()
    effective = _scale_remaining(remaining, prefs.get("target_calories"))

    recipes: list = []
    if source in ("spoonacular", "combo"):
        query_params = build_query_params(effective, prefs)
        recipes.extend(fetch_recipes(query_params, api_key))
    if source in ("kaggle", "combo"):
        recipes.extend(fetch_kaggle_recipes(prefs, effective, n=20))

    ranked = filter_and_rank(recipes, effective, top_n=3)

    suggestions = [
        RecipeSuggestion(
            title=r["title"],
            fit_score=r.get("_fit_score", 0),
            scale_factor=r.get("_portions", 1.0),
            ready_in_minutes=r.get("ready_in_minutes", 0),
            source_url=r.get("source_url", ""),
            ingredients=r.get("ingredients", []),
            ingredient_count=r.get("ingredient_count", 0),
            source=r.get("_source", source),
            nutrition=RecipeNutrition(**r["nutrition"]),
        )
        for r in ranked
    ]

    return RecipeSuggestResponse(remaining=remaining_macros, suggestions=suggestions)


@router.post("/recipes/{uid}/log-suggestion", response_model=Meal, status_code=201)
def log_suggestion(uid: str, body: RecipeLogRequest):
    """
    Save a chosen recipe suggestion to the user's daily meal log.
    Macros are scaled by `portions` and recorded as a single serving-unit item.
    """
    n = body.nutrition
    title = body.title
    portion_note = f" x{body.portions:.2g}" if body.portions != 1.0 else ""

    calories = round(n.calories * body.portions, 1)
    protein  = round(n.protein  * body.portions, 1)
    fat      = round(n.fat      * body.portions, 1)
    carbs    = round(n.carbs    * body.portions, 1)

    saved = get_db().save_meal(
        user_id=uid,
        items=[{
            "item_name":       f"{title}{portion_note}",
            "matched_name":    title,
            "amount_grams":    0,
            "unit":            "serving",
            "calories":        calories,
            "protein":         protein,
            "fat":             fat,
            "carbs":           carbs,
            "confidence":      "high",
            "confidence_note": "saved from recipe suggestion",
        }],
        input_text=f"Recipe suggestion: {title}",
        meal_name=title,
    )

    return Meal(
        meal_id=saved["meal_id"],
        date_time=datetime.datetime.now().isoformat(),
        uid=uid,
        items=[
            FoodItem(
                item_id=saved["item_ids"][0],
                item_name=title,
                calories=calories,
                protein=protein,
                fat=fat,
                carbs=carbs,
                grams=0,
            )
        ],
    )
