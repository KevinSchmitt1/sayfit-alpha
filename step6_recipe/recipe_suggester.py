"""
SayFit Alpha – Step 6: Recipe Suggester
=========================================
Suggests recipes that fit the user's remaining daily macro budget.

Pipeline:
  1. Read today's consumed macros from SQLite
  2. Compute remaining macros  (daily target − consumed)
  3. Collect user preferences   (target kcal, taste, time, ingredients, portion size)
  4. LLM builds Spoonacular query params  (recipe_query_builder)
  5. Fetch matching recipes from Spoonacular API  (recipe_fetcher)
  6. Format and display results  (recipe_formatter)

Standalone usage:
    python -m step6_recipe.recipe_suggester
    python -m step6_recipe.recipe_suggester --user my_user_id
"""

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple, Dict

import config  # triggers load_dotenv → SPOONACULAR_API_KEY available via os.getenv
from step5_database.database import get_db
from step6_recipe.recipe_query_builder import build_query_params
from step6_recipe.recipe_fetcher import fetch_recipes
from step6_recipe.recipe_kaggle_fetcher import fetch_kaggle_recipes
from step6_recipe.recipe_filter import filter_and_rank
from step6_recipe.recipe_formatter import format_recipes

_LOW_KCAL_THRESHOLD  = 200  # kcal — below this, suggest quick options instead of recipes
_DONE_KCAL_THRESHOLD = 100  # kcal — below this, user is basically at their goal


# ── Preferences collector ─────────────────────────────────────────────────────

def collect_preferences(remaining_calories: float = 0) -> Dict:
    """Interactively collect user preferences via CLI prompts."""
    print("\n" + "─" * 64)
    print("  Step 6  –  Recipe Preferences")
    print("─" * 64)

    # 1. Target calories
    cal_hint = f"  (your remaining: {remaining_calories:.0f} kcal)" if remaining_calories > 0 else ""
    print(f"\n  [1] How many calories should this meal have?{cal_hint}")
    print("      Enter a number (e.g. 400) or press Enter to use all remaining.")
    raw = input("      Target kcal [Enter to skip]: ").strip()
    try:
        target_calories = int(raw) if raw else None
        if target_calories is not None and target_calories <= 0:
            target_calories = None
    except ValueError:
        target_calories = None

    # 2. Taste
    print("\n  [2] Taste preference:")
    print("      1 → Savory   (main course, soup, salad)")
    print("      2 → Sweet    (dessert, snack, breakfast treat)")
    print("      3 → Any")
    raw = input("      Choice [1/2/3, default=3]: ").strip() or "3"
    taste = {"1": "savory", "2": "sweet", "3": "any"}.get(raw, "any")

    # 3. Preparation time
    print("\n  [3] Max preparation time:")
    print("      1 → Quick      (≤ 15 min)")
    print("      2 → Moderate   (≤ 30 min)")
    print("      3 → Relaxed    (≤ 60 min)")
    print("      4 → No limit")
    raw = input("      Choice [1/2/3/4, default=4]: ").strip() or "4"
    max_time = {"1": 15, "2": 30, "3": 60, "4": None}.get(raw, None)

    # 4. Ingredients on hand
    print("\n  [4] Ingredients you already have (optional):")
    print("      Example:  zucchini, chicken, tomatoes")
    raw = input("      Ingredients (Enter to skip): ").strip()
    ingredients = [i.strip() for i in raw.split(",") if i.strip()] if raw else []

    # 5. Few-ingredients filter
    print("\n  [5] Prefer recipes with few ingredients?")
    raw = input("      Limit to max 5 ingredients? [y/n, default=n]: ").strip().lower()
    few_ingredients = raw in ("y", "yes")

    return {
        "target_calories":  target_calories,
        "taste":            taste,
        "max_time_minutes": max_time,
        "ingredients":      ingredients,
        "few_ingredients":  few_ingredients,
    }


# ── Macro helpers ─────────────────────────────────────────────────────────────

def compute_remaining(user_id: str) -> Tuple[Dict, Optional[Dict], Dict]:
    """
    Return (remaining_macros, profile, consumed) for today.

    If no user profile is set, remaining_macros falls back to sensible defaults
    so the pipeline still works for users who haven't configured daily targets.
    """
    db = get_db()
    today = datetime.now().strftime("%Y-%m-%d")
    consumed = db.get_daily_totals(user_id, today)
    profile = db.get_user_profile(user_id)

    if profile:
        remaining = {
            "calories": max(0.0, profile["kcal_daily"]     - consumed["calories"]),
            "protein":  max(0.0, profile["protein_daily"]  - consumed["protein"]),
            "fat":      max(0.0, profile["fat_daily"]      - consumed["fat"]),
            "carbs":    max(0.0, profile["carbs_daily"]    - consumed["carbs"]),
        }
    else:
        # No profile → use conservative defaults
        remaining = {"calories": 500.0, "protein": 30.0, "fat": 20.0, "carbs": 50.0}

    return remaining, profile, consumed


def _effective_remaining(remaining: Dict, target_calories: Optional[int]) -> Dict:
    """
    If the user wants a smaller meal than the full remaining budget,
    scale all macros proportionally so thresholds and scoring stay consistent.
    """
    if not target_calories or target_calories >= remaining["calories"]:
        return remaining
    scale = target_calories / remaining["calories"]
    return {
        "calories": float(target_calories),
        "protein":  remaining["protein"] * scale,
        "fat":      remaining["fat"]     * scale,
        "carbs":    remaining["carbs"]   * scale,
    }


def _print_macro_header(user_id: str, consumed: Dict, remaining: Dict, has_profile: bool) -> None:
    print("\n" + "=" * 64)
    print(f"  Recipe Suggester  —  user: {user_id}")
    print("=" * 64)
    print(
        f"  Consumed today  :  {consumed['calories']:.0f} kcal  |  "
        f"P {consumed['protein']:.0f}g  "
        f"F {consumed['fat']:.0f}g  "
        f"C {consumed['carbs']:.0f}g"
    )
    if has_profile:
        print(
            f"  Remaining       :  {remaining['calories']:.0f} kcal  |  "
            f"P {remaining['protein']:.0f}g  "
            f"F {remaining['fat']:.0f}g  "
            f"C {remaining['carbs']:.0f}g"
        )
    else:
        print("  [No daily profile set — using default remaining targets]")


def _print_low_kcal_tip(remaining_kcal: float) -> None:
    print(f"\n  Note: Only {remaining_kcal:.0f} kcal remaining — quick options that fit:")
    print("    • Protein shake        (~120–150 kcal, ~25g protein)")
    print("    • Greek yogurt (plain) (~150 kcal, ~15g protein)")
    print("    • Cottage cheese       (~100 kcal, ~14g protein)")
    print("    • 2 hard-boiled eggs   (~140 kcal, ~12g protein)\n")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_recipe_suggester(user_id: Optional[str] = None, source: str = "spoonacular") -> None:
    """
    Full recipe suggestion pipeline.

    Parameters
    ----------
    user_id : str, optional
        Defaults to the last-used user ID stored in data/calibrations/last_user.txt.
    source : str
        One of "spoonacular", "kaggle", or "combo" (both merged).
    """
    # Resolve API key (only required for spoonacular / combo)
    api_key = os.getenv("SPOONACULAR_API_KEY", "").strip()
    if source in ("spoonacular", "combo") and not api_key:
        print("[ERROR] SPOONACULAR_API_KEY not found. Add it to your .env file.")
        if source == "spoonacular":
            return
        print("  Falling back to Kaggle-only source.")
        source = "kaggle"

    # Resolve user ID
    if not user_id:
        last_user_file = config.ROOT_DIR / "data" / "calibrations" / "last_user.txt"
        try:
            user_id = last_user_file.read_text().strip() or config.DEFAULT_USER_ID
        except OSError:
            user_id = config.DEFAULT_USER_ID

    # Step 1 – remaining macros
    remaining, profile, consumed = compute_remaining(user_id)
    _print_macro_header(user_id, consumed, remaining, profile is not None)

    if remaining["calories"] <= 0 and profile:
        print("\n  You have already reached (or exceeded) your calorie target for today.")
        print("  No recipe suggestions needed — great job!\n")
        return

    # Below 100 kcal — user is essentially at their daily goal
    if 0 < remaining["calories"] < _DONE_KCAL_THRESHOLD:
        print(f"\n  Only {remaining['calories']:.0f} kcal left — you're basically at your daily goal!")
        print("  Have a small piece of fruit or just call it a day. Nice work!\n")
        return

    # 100–200 kcal — too little for a real recipe, suggest quick options
    if remaining["calories"] <= _LOW_KCAL_THRESHOLD:
        _print_low_kcal_tip(remaining["calories"])
        return

    # Step 2 – preferences
    preferences = collect_preferences(remaining_calories=remaining["calories"])

    # Adjust remaining budget if the user picked a specific calorie target
    effective = _effective_remaining(remaining, preferences.get("target_calories"))

    # Step 3 – fetch candidates (source-dependent)
    recipes: list = []

    if source in ("spoonacular", "combo"):
        print("\n  Building recipe query via LLM...")
        query_params = build_query_params(effective, preferences)
        print("  Fetching recipe candidates from Spoonacular API...")
        spoon_recipes = fetch_recipes(query_params, api_key)
        print(f"  Spoonacular returned {len(spoon_recipes)} candidate(s).")
        recipes.extend(spoon_recipes)

    if source in ("kaggle", "combo"):
        print("  Fetching recipe candidates from local Kaggle DB...")
        kaggle_recipes = fetch_kaggle_recipes(preferences, effective, n=20)
        print(f"  Kaggle DB returned {len(kaggle_recipes)} candidate(s).")
        recipes.extend(kaggle_recipes)

    if not recipes:
        print("\n  No recipes found matching your criteria.")
        print("  Suggestions: remove ingredient filters, increase time limit, or choose 'Any' taste.\n")
        return

    # Step 4 – filter and rank by macro fit
    ranked = filter_and_rank(recipes, effective, top_n=3)

    if not ranked:
        print("\n  All candidates were filtered out (macro fit too low).")
        print("  Try relaxing filters or run again — different results may come back.\n")
        return

    # Step 5 – format output
    _print_source_legend(source, ranked)
    format_recipes(ranked, effective)

    # Step 6 – optional save to daily log
    _prompt_save_recipe(ranked, user_id, remaining, profile)


def _prompt_save_recipe(
    ranked: list,
    user_id: str,
    remaining: Dict,
    profile,
) -> None:
    """Ask the user if they want to log one of the suggested recipes."""
    print("─" * 64)
    print("  Save a recipe to today's log?")
    print("  Enter 1, 2, or 3 to log the macros — or press Enter to skip.")
    try:
        raw = input("  Choice [1/2/3 or Enter]: ").strip()
    except EOFError:
        raw = ""

    if raw not in ("1", "2", "3"):
        print("  Skipped — nothing saved.\n")
        return

    idx = int(raw) - 1
    if idx >= len(ranked):
        print("  Invalid choice — nothing saved.\n")
        return

    recipe = ranked[idx]
    n = recipe["nutrition"]
    title = recipe["title"]
    portions = recipe.get("_portions", 1.0)
    portion_note = f" x{portions:.2g}" if portions != 1.0 else ""

    db = get_db()
    db.save_meal(
        user_id=user_id,
        items=[{
            "item_name":       f"{title}{portion_note}",
            "matched_name":    title,
            "amount_grams":    0,
            "unit":            "serving",
            "calories":        n["calories"],
            "protein":         n["protein"],
            "fat":             n["fat"],
            "carbs":           n["carbs"],
            "confidence":      "high",
            "confidence_note": "saved from recipe suggestion",
        }],
        input_text=f"Recipe suggestion: {title}",
        meal_name=title,
    )

    # Recompute and display updated remaining macros
    new_remaining, _, new_consumed = compute_remaining(user_id)
    print("\n" + "=" * 64)
    print(f"  Updated daily totals after saving \"{title}\":")
    print("─" * 64)
    print(
        f"  Consumed  :  {new_consumed['calories']:.0f} kcal  |  "
        f"P {new_consumed['protein']:.0f}g  "
        f"F {new_consumed['fat']:.0f}g  "
        f"C {new_consumed['carbs']:.0f}g"
    )
    if profile:
        print(
            f"  Remaining :  {new_remaining['calories']:.0f} kcal  |  "
            f"P {new_remaining['protein']:.0f}g  "
            f"F {new_remaining['fat']:.0f}g  "
            f"C {new_remaining['carbs']:.0f}g"
        )
    print("=" * 64 + "\n")


def _print_source_legend(source: str, ranked: list) -> None:
    if source not in ("combo",):
        return
    sources = [r.get("_source", "spoonacular") for r in ranked]
    counts  = {s: sources.count(s) for s in ("spoonacular", "kaggle") if sources.count(s)}
    parts   = [f"{v} from {k}" for k, v in counts.items()]
    print(f"\n  Sources in top results: {', '.join(parts)}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SayFit – Recipe Suggester")
    parser.add_argument("--user",   type=str, default=None,
                        help="User ID to look up in the database")
    parser.add_argument("--source", type=str, default="spoonacular",
                        choices=["spoonacular", "kaggle", "combo"],
                        help="Recipe source: spoonacular | kaggle | combo (default: spoonacular)")
    args = parser.parse_args()
    run_recipe_suggester(user_id=args.user, source=args.source)
