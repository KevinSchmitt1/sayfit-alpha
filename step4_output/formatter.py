"""
Step 4 – Output Formatter
==========================
Takes the finalised food items from Step 3 and produces:
  - A pretty terminal table of items + matched food + grams + macros
  - Daily totals row
  - Optional save to JSON / CSV

Standalone:  python -m step4_output.run [--input step3_output.json]
"""

import csv
import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _log(*args, **kwargs):
    """Print only when developer mode is active."""
    if config.DEV_MODE:
        print(*args, **kwargs)


# ── Table rendering (pure Python, no extra deps) ────────────────────────────

def _pad(text: str, width: int, align: str = "<") -> str:
    """Pad text to a fixed width."""
    text = str(text)[:width]
    if align == ">":
        return text.rjust(width)
    elif align == "^":
        return text.center(width)
    return text.ljust(width)


def render_table(results: list[dict]) -> str:
    """
    Render the finalised items as an ASCII table.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys: item_name, matched_name, amount_grams, unit,
        processing_description, confidence, nutrition (dict with calories,
        protein, fat, carbs).

    Returns
    -------
    str – formatted table ready for print().
    """
    # columns: # | Item | Matched | Grams | Kcal | Protein | Fat | Carbs | Conf
    cols = [
        ("#",      3,  "^"),
        ("Item",   20, "<"),
        ("Matched Food", 25, "<"),
        ("Grams",  7,  ">"),
        ("Kcal",   8,  ">"),
        ("Protein", 8, ">"),
        ("Fat",    8,  ">"),
        ("Carbs",  8,  ">"),
        ("Conf",   6,  "^"),
    ]

    sep = "+" + "+".join("-" * (w + 2) for _, w, _ in cols) + "+"
    header = "|" + "|".join(f" {_pad(name, w, a)} " for name, w, a in cols) + "|"

    lines = [sep, header, sep]

    total_kcal = total_prot = total_fat = total_carbs = total_grams = 0.0

    for i, item in enumerate(results, 1):
        nutr = item.get("nutrition", {})
        kcal = nutr.get("calories", 0) or 0
        prot = nutr.get("protein", 0) or 0
        fat = nutr.get("fat", 0) or 0
        carbs = nutr.get("carbs", 0) or 0
        grams = item.get("amount_grams", 0) or 0
        conf = item.get("confidence", "?")

        total_kcal += kcal
        total_prot += prot
        total_fat += fat
        total_carbs += carbs
        total_grams += grams

        row_data = [
            str(i),
            item.get("item_name", ""),
            item.get("matched_name", ""),
            f"{grams:.0f}",
            f"{kcal:.1f}",
            f"{prot:.1f}g",
            f"{fat:.1f}g",
            f"{carbs:.1f}g",
            conf[:6],
        ]
        row = "|" + "|".join(
            f" {_pad(val, w, a)} " for val, (_, w, a) in zip(row_data, cols)
        ) + "|"
        lines.append(row)

    # totals row
    lines.append(sep)
    total_data = [
        "",
        "DAILY TOTAL",
        "",
        f"{total_grams:.0f}",
        f"{total_kcal:.1f}",
        f"{total_prot:.1f}g",
        f"{total_fat:.1f}g",
        f"{total_carbs:.1f}g",
        "",
    ]
    total_row = "|" + "|".join(
        f" {_pad(val, w, a)} " for val, (_, w, a) in zip(total_data, cols)
    ) + "|"
    lines.append(total_row)
    lines.append(sep)

    return "\n".join(lines)


def render_summary(results: list[dict]) -> str:
    """Render a short text summary with daily totals."""
    total_kcal = sum((r.get("nutrition", {}).get("calories", 0) or 0) for r in results)
    total_prot = sum((r.get("nutrition", {}).get("protein", 0) or 0) for r in results)
    total_fat = sum((r.get("nutrition", {}).get("fat", 0) or 0) for r in results)
    total_carbs = sum((r.get("nutrition", {}).get("carbs", 0) or 0) for r in results)

    lines = [
        "\n📊 Daily Totals:",
        f"   Calories : {total_kcal:.1f} kcal",
        f"   Protein  : {total_prot:.1f} g",
        f"   Fat      : {total_fat:.1f} g",
        f"   Carbs    : {total_carbs:.1f} g",
    ]

    # flag low-confidence items
    low_conf = [r for r in results if r.get("confidence") in ("low", "medium")]
    if low_conf:
        lines.append("\n⚠️  Items with uncertain matching:")
        for r in low_conf:
            note = r.get("confidence_note", "")
            lines.append(f"   • {r.get('item_name', '?')} → {r.get('matched_name', '?')} "
                         f"({r.get('confidence', '?')}) {note}")

    return "\n".join(lines)


def format_output(reranker_output: dict) -> str:
    """
    Full formatting of the Step 3 output.

    Parameters
    ----------
    reranker_output : dict
        Output from step3 with "results" list.

    Returns
    -------
    str – complete formatted output.
    """
    results = reranker_output.get("results", [])
    _log("\n📋 [Step 4] Formatting output …")

    table = render_table(results)
    summary = render_summary(results)
    output = f"\n{'=' * 60}\n  🍽️  SayFit – Nutrition Log\n{'=' * 60}\n\n{table}\n{summary}\n"
    return output


def save_log(reranker_output: dict, output_dir: Path | None = None) -> Path:
    """Save the results as a JSON log file with timestamp."""
    output_dir = output_dir or config.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"log_{ts}.json"
    with open(path, "w") as f:
        json.dump(reranker_output, f, indent=2)
    _log(f"   💾 Log saved: {path}")
    return path
