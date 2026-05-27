"""
SayFit Alpha – Spoonacular Recipe Fetcher (Step 6)
===================================================
Calls the Spoonacular complexSearch endpoint and returns
normalised recipe dicts ready for formatting.

Uses only stdlib (urllib) — no additional dependencies required.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Any, List


_BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"


def _encode_params(params: Dict[str, Any]) -> str:
    """Encode params, serialising Python booleans as lowercase strings (Spoonacular is case-sensitive)."""
    safe = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}
    return urllib.parse.urlencode(safe)


def fetch_recipes(query_params: Dict[str, Any], api_key: str) -> List[Dict]:
    """
    Call Spoonacular complexSearch with the given params.

    Returns a list of normalised recipe dicts, each containing:
      id, title, ready_in_minutes, source_url,
      nutrition {calories, protein, fat, carbs},
      ingredient_count, ingredients (list of names)
    """
    params = {**query_params, "apiKey": api_key}
    url = _BASE_URL + "?" + _encode_params(params)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SayFit-Alpha/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())

    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        if exc.code in (402, 403):
            print("  [ERROR] Spoonacular daily quota exceeded. The free tier allows 150 points/day.")
            print("          Try again tomorrow or upgrade your plan at spoonacular.com.")
        else:
            print(f"  [ERROR] Spoonacular API error {exc.code}: {body[:200]}")
        return []

    except urllib.error.URLError as exc:
        print(f"  [ERROR] Network error: {exc.reason}")
        return []

    except Exception as exc:
        print(f"  [ERROR] Unexpected error fetching recipes: {exc}")
        return []

    results = data.get("results", [])
    return [_parse_recipe(r) for r in results]


def _parse_recipe(raw: Dict) -> Dict:
    """Flatten a Spoonacular result into a clean recipe dict."""
    nutrition = raw.get("nutrition", {})
    # nutrients is a list of {name, amount, unit} dicts
    nutrients: Dict[str, float] = {}
    for item in nutrition.get("nutrients", []):
        nutrients[item["name"].lower()] = item.get("amount", 0.0)

    extended = raw.get("extendedIngredients", [])

    return {
        "id": raw.get("id"),
        "title": raw.get("title", "Unknown Recipe"),
        "ready_in_minutes": raw.get("readyInMinutes") or 0,
        "source_url": raw.get("sourceUrl", ""),
        "nutrition": {
            "calories": nutrients.get("calories", 0.0),
            "protein":  nutrients.get("protein", 0.0),
            "fat":      nutrients.get("fat", 0.0),
            "carbs":    nutrients.get("carbohydrates", 0.0),
        },
        "ingredient_count": len(extended),
        "ingredients": [i.get("name", "") for i in extended],
    }
