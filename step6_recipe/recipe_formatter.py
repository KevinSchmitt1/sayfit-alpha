"""
SayFit Alpha – Recipe Formatter (Step 6)
=========================================
Renders recipe suggestions as readable ASCII output.
"""

import textwrap
from typing import List, Dict


def format_recipes(recipes: List[Dict], remaining_macros: Dict[str, float]) -> None:
    """Print all recipe suggestions with macro fit indicators."""
    print("\n" + "=" * 64)
    print(f"  Recipe Suggestions  —  {len(recipes)} result(s)")
    print("=" * 64)

    for idx, recipe in enumerate(recipes, 1):
        _print_recipe(idx, recipe, remaining_macros)

    print("=" * 64)
    print("  Tip: run again with different filters to explore more options.")
    print("=" * 64 + "\n")


def _print_recipe(idx: int, recipe: Dict, remaining_macros: Dict[str, float]) -> None:
    n = recipe["nutrition"]
    time_str = f"{recipe['ready_in_minutes']} min" if recipe["ready_in_minutes"] else "N/A"
    ingr_count = recipe["ingredient_count"]
    portions = recipe.get("_portions", 1.0)
    fit_score = recipe.get("_fit_score", "—")

    # Title line with fit score
    fit_tag = f"  [fit: {fit_score}/100]" if fit_score != "—" else ""
    print(f"\n  [{idx}]  {recipe['title']}{fit_tag}")

    # Portion note if scaled
    if portions != 1.0:
        print(f"        Portions: {portions:.2g}x  (scaled up to meet macro thresholds)")

    print(f"        Time : {time_str:<8}  Ingredients: {ingr_count if ingr_count else '—'}")
    print(
        f"        Kcal : {n['calories']:>6.0f}  |  "
        f"Protein {n['protein']:.1f}g  "
        f"Fat {n['fat']:.1f}g  "
        f"Carbs {n['carbs']:.1f}g"
    )

    # Macro fit bar vs remaining calories
    cal_rem = remaining_macros.get("calories", 0)
    if cal_rem > 0 and n["calories"] > 0:
        fit_pct = (n["calories"] / cal_rem) * 100
        bar = _bar(fit_pct)
        fit_label = "over budget" if fit_pct > 110 else f"{fit_pct:.0f}% of remaining kcal"
        print(f"        Fit  : {bar}  {fit_label}")

    # Ingredient list (first 7, truncated if more)
    if recipe["ingredients"]:
        shown = recipe["ingredients"][:7]
        suffix = f"  +{len(recipe['ingredients']) - 7} more" if len(recipe["ingredients"]) > 7 else ""
        print(f"        Ingr : {', '.join(shown)}{suffix}")

    if recipe["source_url"]:
        print(f"        URL  : {recipe['source_url']}")

    # Preparation steps — shown for Kaggle recipes (no URL available)
    steps = recipe.get("steps", [])
    if steps and not recipe.get("source_url"):
        print(f"        Steps ({len(steps)} total):")
        for i, step in enumerate(steps, 1):
            _print_step(i, step.strip().capitalize())


def _print_step(num: int, text: str) -> None:
    """Print a numbered step with word-wrapped continuation lines."""
    prefix = f"          {num}. "
    cont = " " * len(prefix)
    lines = textwrap.wrap(text, width=72, initial_indent=prefix, subsequent_indent=cont)
    for line in lines:
        print(line)


def _bar(pct: float, width: int = 12) -> str:
    """ASCII progress bar capped at 100%."""
    filled = min(int(min(pct, 100) / (100 / width)), width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"
