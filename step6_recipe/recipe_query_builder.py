"""
SayFit Alpha – Recipe Query Builder (Step 6)
=============================================
Uses the LLM to convert remaining daily macros + user preferences
into a structured Spoonacular complexSearch API query.

Flow: remaining_macros + preferences → LLM → JSON params dict
"""

import json
import re
from typing import Dict, Any

import llm_client


_SYSTEM_PROMPT = """You are a nutrition assistant that converts meal preferences and remaining daily macros into a structured Spoonacular recipe search query.

Given the user's remaining macros and preferences, output a JSON object with valid Spoonacular complexSearch parameters.

Parameter rules:
- maxCalories: remaining calories × 1.1, rounded to nearest 10 (omit if calories <= 0)
- minProtein: remaining protein × 0.3 rounded down — only if 10g < remaining protein < 80g (a single recipe covers ~30% of daily protein at most)
- maxFat: remaining fat rounded up — only if remaining fat < 30g
- maxCarbs: remaining carbs rounded up + 5 — only if remaining carbs < 50g
- type: "main course" if taste=savory, "dessert" if taste=sweet, omit if taste=any
- maxReadyTime: max_time_minutes value — only if a time limit is set (not null)
- includeIngredients: comma-separated string of provided ingredients — omit if list is empty
- maxIngredients: 5 — only if few_ingredients is true
- Always include: number=8, addRecipeNutrition=true, addRecipeInformation=true, instructionsRequired=true

Output ONLY a valid JSON object. No explanation, no markdown fences, no extra text."""


def build_query_params(remaining_macros: Dict[str, float], preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to translate remaining macros + user preferences into
    Spoonacular complexSearch query parameters.

    Falls back to deterministic logic if the LLM call fails.
    """
    client = llm_client.get_client()
    model = llm_client.reasoning_model()

    payload = json.dumps(
        {"remaining_macros": remaining_macros, "preferences": preferences},
        indent=2,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip("` \n")
        params = json.loads(raw)

        params.setdefault("number", 8)
        params.setdefault("addRecipeNutrition", True)
        params.setdefault("addRecipeInformation", True)
        params.setdefault("instructionsRequired", True)

        return params

    except Exception as exc:
        print(f"  [WARN] LLM query builder failed ({exc}) — using fallback params")
        return _fallback_params(remaining_macros, preferences)


def _fallback_params(remaining_macros: Dict[str, float], preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback when the LLM call cannot be completed."""
    params: Dict[str, Any] = {
        "number": 8,
        "addRecipeNutrition": True,
        "addRecipeInformation": True,
        "instructionsRequired": True,
    }

    cal = remaining_macros.get("calories", 0)
    if cal > 0:
        params["maxCalories"] = round(cal * 1.1 / 10) * 10

    # Only apply protein floor if remaining is meaningful and use a conservative threshold
    # to avoid over-filtering (a single recipe rarely covers more than 30–40% of daily protein)
    protein = remaining_macros.get("protein", 0)
    if 10 < protein < 80:
        params["minProtein"] = int(protein * 0.3)

    fat = remaining_macros.get("fat", 999)
    if fat < 30:
        params["maxFat"] = int(fat) + 2

    carbs = remaining_macros.get("carbs", 999)
    if carbs < 50:
        params["maxCarbs"] = int(carbs) + 5

    taste = preferences.get("taste", "any")
    if taste == "savory":
        params["type"] = "main course"
    elif taste == "sweet":
        params["type"] = "dessert"

    if preferences.get("max_time_minutes"):
        params["maxReadyTime"] = preferences["max_time_minutes"]

    ingredients = preferences.get("ingredients", [])
    if ingredients:
        params["includeIngredients"] = ",".join(ingredients)

    if preferences.get("few_ingredients"):
        params["maxIngredients"] = 5

    return params
