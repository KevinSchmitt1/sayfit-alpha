"""
SayFit Alpha – Main Pipeline
==============================
Orchestrates all steps:
  0.   Voice Input (optional)      → mic recording / .wav transcription
  1.   Item Extraction (LLM)       → structured food items from raw text
  1.5  Ontology Filter             → predict L1 food category per item
  2.   Retrieval (FAISS/RAG)       → top-K food candidates (category-boosted)
  3.   Reranker (LLM)              → best match + portion estimate + macros
  4.   Output                      → formatted table + daily totals
  5.   Database                    → persist meal to SQLite logbook

Supports two modes:
  • Interactive:  type / paste what you ate and get instant results.
  • File-based:   pass a voice-recorder JSON via --input.

Usage:
    python main.py                                   # interactive mode
    python main.py --input inputs/my_meal.json       # file mode
    python main.py --text "i ate 2 bananas and a coffee"
"""

import argparse
import itertools
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import config
import llm_client
from step5_database.database import get_db
from step1_extraction.extractor import extract_items
from step3_reranker.reranker import rerank_all
from step3_reranker.calibration import save_user_correction
from step4_output.formatter import format_output

# ── Last-used user persistence ────────────────────────────────────────────────
_LAST_USER_FILE = config.ROOT_DIR / "data" / "calibrations" / "last_user.txt"


def _get_last_user() -> str:
    """Return the uid that was used most recently, falling back to DEFAULT_USER_ID."""
    try:
        uid = _LAST_USER_FILE.read_text().strip()
        return uid if uid else config.DEFAULT_USER_ID
    except OSError:
        return config.DEFAULT_USER_ID


def _set_last_user(uid: str) -> None:
    """Persist uid so the next session picks it up as the default."""
    try:
        _LAST_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_USER_FILE.write_text(uid)
    except OSError:
        pass


# ── Progress spinner (used in normal / non-devmode) ──────────────────────────

class Spinner:
    """
    Displays an animated spinner with a step label while a pipeline
    step runs. Suppressed automatically in devmode (where verbose
    output already makes progress visible).
    """
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str):
        self._label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for frame in itertools.cycle(self._FRAMES):
            if self._stop_event.is_set():
                break
            sys.stdout.write(f"\r  {frame}  {self._label} …")
            sys.stdout.flush()
            time.sleep(0.08)

    def __enter__(self):
        if not config.DEV_MODE:
            self._thread.start()
        return self

    def __exit__(self, *_):
        if not config.DEV_MODE:
            self._stop_event.set()
            self._thread.join()
            # overwrite spinner line with a clean ✓ line
            sys.stdout.write(f"\r  \u2713  {self._label:<40}\n")
            sys.stdout.flush()


def _ensure_index():
    """Check that the FAISS index exists; if not, offer to build it."""
    index_path = config.INDEX_DIR / "food.index"
    if index_path.exists():
        return True

    print("\n⚠️  FAISS index not found. You need to build it first.")
    print("   This is a one-time step that indexes the food databases.")
    ans = input("   Build index now? [Y/n] ").strip().lower()
    if ans in ("", "y", "yes"):
        from step2_retrieval.build_index import build_index
        build_index()
        return True
    else:
        print("   Skipping index build. Retrieval will not work.")
        return False


def _make_run_dir(parent: Path | None = None, name: str | None = None) -> Path:
    """
    Create and return a run directory.
    - parent + name → parent/name/  (für test-folder Modus)
    - sonst         → outputs/<date>/run_NNN/
    """
    if parent is not None and name is not None:
        run_dir = parent / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = config.OUTPUTS_DIR / today
    date_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(date_dir.glob("run_*"))
    next_n = len(existing) + 1
    run_dir = date_dir / f"run_{next_n:03d}"
    run_dir.mkdir()
    return run_dir


def _make_batch_dir() -> Path:
    """Erstellt outputs/<date>/testrun_NNN/ für einen kompletten Test-Folder-Lauf."""
    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = config.OUTPUTS_DIR / today
    date_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(date_dir.glob("testrun_*"))
    next_n = len(existing) + 1
    batch_dir = date_dir / f"testrun_{next_n:03d}"
    batch_dir.mkdir()
    return batch_dir


def run_pipeline(
    text: str,
    date_time: str = "",
    uid: str = "default_user",
    meta: dict = None,
    use_llm: bool = True,
    run_parent: Path | None = None,
    run_name: str | None = None,
):
    """
    Run the full pipeline on a single text input.

    Returns the final reranker output dict.
    """
    run_dir = _make_run_dir(parent=run_parent, name=run_name)
    if config.DEV_MODE:
        print(f"   📁 Run output: {run_dir}")

    # Save test metadata if provided (used by analyze_tests.py)
    if meta:
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    if not config.DEV_MODE:
        print()  # blank line before spinner block

    # ── Step 1: Extraction ───────────────────────────────────────────────
    with Spinner("Extracting food items"):
        extraction = extract_items(text, date_time=date_time, uid=uid, use_llm=use_llm)

    ext_path = run_dir / "step1_extraction_output.json"
    with open(ext_path, "w") as f:
        json.dump(extraction, f, indent=2)
    if config.DEV_MODE:
        print(f"   💾 {ext_path.relative_to(config.ROOT_DIR)}")

    # ── Step 1.5: Ontology Filter ─────────────────────────────────────────
    from step1_5_ontology_filter.ontology_filter import (
        apply_ontology_filter, _build_l2_embed_index, _load_food_index,
    )
    with Spinner("Loading food model"):
        _build_l2_embed_index()
        _load_food_index()
    with Spinner("Classifying food categories"):
        extraction = apply_ontology_filter(extraction)

    ont_path = run_dir / "step1_5_ontology_output.json"
    with open(ont_path, "w") as f:
        json.dump(extraction, f, indent=2)
    if config.DEV_MODE:
        print(f"   💾 {ont_path.relative_to(config.ROOT_DIR)}")

    # ── Step 2: Retrieval ────────────────────────────────────────────────
    queries = extraction.get("queries", [])
    if not queries:
        print("   ⚠️  No queries extracted – nothing to retrieve.")
        return {"results": []}

    category_hints = extraction.get("category_hints", [])

    from step2_retrieval.retriever import retrieve
    with Spinner("Finding food matches"):
        retrieval = retrieve(queries, category_hints=category_hints)

    ret_path = run_dir / "step2_retrieval_output.json"
    with open(ret_path, "w") as f:
        json.dump(retrieval, f, indent=2)
    if config.DEV_MODE:
        print(f"   💾 {ret_path.relative_to(config.ROOT_DIR)}")

    # ── Step 3: Reranker ─────────────────────────────────────────────────
    with Spinner("Estimating portions & calculating macros"):
        reranked = rerank_all(extraction, retrieval, use_llm=use_llm)

    rer_path = run_dir / "step3_reranker_output.json"
    with open(rer_path, "w") as f:
        json.dump(reranked, f, indent=2)
    if config.DEV_MODE:
        print(f"   💾 {rer_path.relative_to(config.ROOT_DIR)}")

    # ── Step 4: Output ───────────────────────────────────────────────────
    output_text = format_output(reranked)
    print(output_text)

    return reranked


def ask_user_corrections(
    reranked: dict,
    uid: str = "default_user",
    use_llm: bool = True,
) -> dict:
    """
    Interactively review ALL items and let the user:
      - Edit the gram amount (nutrition is recalculated immediately)
      - Search for a completely different food (triggers re-retrieval + re-reranking)
      - Remove an item entirely (d<N>)
    """
    results = reranked.get("results", [])
    if not results:
        return reranked

    def _display(items: list) -> None:
        print("\n" + "=" * 60)
        print("  📋  Review your meal – make corrections before saving")
        print("=" * 60)
        for i, r in enumerate(items, 1):
            conf = r.get("confidence", "?")
            conf_tag = "  ⚠️ LOW" if conf == "low" else ""
            nutr = r.get("nutrition", {})
            print(
                f"  [{i}] {r.get('item_name')} → {r.get('matched_name')} "
                f"({r.get('amount_grams', '?')}g | "
                f"{nutr.get('calories', '?')} kcal){conf_tag}"
            )
        print()
        print("  Enter a number (N) to edit | d<N> to remove | Enter to accept all")

    _display(results)

    # Track items that need a new database search (by object reference)
    items_to_research: list[dict] = []
    _any_changes = False

    while True:
        choice = input("\n  > ").strip()
        if not choice or choice.lower() in ("q", "quit", "done"):
            break

        # ── Remove: d1, d2, … ──────────────────────────────────────────
        if choice.lower().startswith("d"):
            try:
                idx = int(choice[1:]) - 1
                if 0 <= idx < len(results):
                    removed = results.pop(idx)
                    # also remove from research queue if it was queued
                    items_to_research = [x for x in items_to_research if x is not removed]
                    _any_changes = True
                    print(f"  🗑️  Removed: \"{removed.get('item_name')}\"")
                    if results:
                        _display(results)
                    else:
                        print("  (no items left)")
                else:
                    print(f"  ❌ Invalid. Choose 1–{len(results)}.")
            except ValueError:
                print("  ❌ Invalid command. Use d1, d2, etc.")
            continue

        # ── Edit ────────────────────────────────────────────────────────
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(results):
                print(f"  ❌ Invalid. Choose 1–{len(results)}.")
                continue
        except ValueError:
            print("  ❌ Invalid command.")
            continue

        item = results[idx]
        nutr = item.get("nutrition", {})
        print(
            f"\n  [{idx+1}] {item['item_name']} → {item['matched_name']}\n"
            f"       {item.get('amount_grams', '?')}g | "
            f"{nutr.get('calories', '?')} kcal | "
            f"P:{nutr.get('protein', '?')}g | "
            f"F:{nutr.get('fat', '?')}g | "
            f"C:{nutr.get('carbs', '?')}g"
        )
        print()

        # --- Search for a different food (triggers re-retrieval) -------
        new_name = input(
            f"  Search for different food (Enter to keep '{item['item_name']}'): "
        ).strip()
        if new_name:
            item["item_name"] = new_name
            item["_needs_research"] = True
            # Pre-seed the gram override with the current amount so re-search
            # preserves it unless the user explicitly changes grams below
            item["_gram_override"] = item.get("amount_grams")
            if item not in items_to_research:
                items_to_research.append(item)
            print(f"  🔍 Will re-search for: '{new_name}'")

        # --- Correct gram amount ---------------------------------------
        new_grams_str = input(
            f"  New amount in grams (Enter to keep {item.get('amount_grams', '?')}g): "
        ).strip()
        if new_grams_str:
            try:
                new_grams = float(new_grams_str)
                old_grams = item.get("amount_grams") or 100
                per100 = item.get("nutrition_per_100g", {})

                if per100:
                    # Accurate: scale from stored per-100g values
                    factor = new_grams / 100.0
                    item["nutrition"] = {
                        "calories": round((per100.get("calories") or 0) * factor, 1),
                        "protein":  round((per100.get("protein")  or 0) * factor, 1),
                        "fat":      round((per100.get("fat")       or 0) * factor, 1),
                        "carbs":    round((per100.get("carbs")     or 0) * factor, 1),
                    }
                else:
                    # Fallback: proportional scaling from existing totals
                    scale = new_grams / old_grams
                    item["nutrition"] = {
                        "calories": round((nutr.get("calories") or 0) * scale, 1),
                        "protein":  round((nutr.get("protein")  or 0) * scale, 1),
                        "fat":      round((nutr.get("fat")       or 0) * scale, 1),
                        "carbs":    round((nutr.get("carbs")     or 0) * scale, 1),
                    }

                item["amount_grams"] = new_grams
                item["unit"] = "g"
                # Store so re-search can respect the user's override
                item["_gram_override"] = new_grams
                _any_changes = True

                save_user_correction(uid, item["item_name"], new_grams)
                print(
                    f"  ✅ Updated: {new_grams}g | "
                    f"{item['nutrition'].get('calories', '?')} kcal"
                )
            except ValueError:
                print("  ❌ Invalid number.")

    # ── Re-search for corrected item names ────────────────────────────────
    if items_to_research:
        from step2_retrieval.retriever import retrieve
        from step3_reranker.reranker import rerank_single_item, rerank_single_item_heuristic

        print(f"\n🔍 Re-searching {len(items_to_research)} corrected item(s) …")
        for item in items_to_research:
            new_name = item["item_name"]
            print(f"   Searching for: \"{new_name}\" …")

            new_retrieval = retrieve([new_name], category_hints=[])
            new_candidates = []
            for ri in new_retrieval.get("items", []):
                new_candidates.extend(ri.get("candidates", []))

            if not new_candidates:
                print(f"   ⚠️  No candidates found for \"{new_name}\". Keeping current match.")
                continue

            if use_llm:
                new_result = rerank_single_item(item, new_candidates, uid=uid)
            else:
                new_result = rerank_single_item_heuristic(item, new_candidates)

            # If the user also changed grams, apply that override on top
            gram_override = item.get("_gram_override")
            if gram_override:
                per100 = new_result.get("nutrition_per_100g", {})
                if per100:
                    factor = gram_override / 100.0
                    new_result["nutrition"] = {
                        "calories": round((per100.get("calories") or 0) * factor, 1),
                        "protein":  round((per100.get("protein")  or 0) * factor, 1),
                        "fat":      round((per100.get("fat")       or 0) * factor, 1),
                        "carbs":    round((per100.get("carbs")     or 0) * factor, 1),
                    }
                new_result["amount_grams"] = gram_override

            # Replace in results list (match by identity)
            for j, r in enumerate(results):
                if r is item:
                    results[j] = new_result
                    break

            nutr_new = new_result.get("nutrition", {})
            print(
                f"   ✅ Re-matched: \"{new_name}\" → \"{new_result.get('matched_name')}\" "
                f"({new_result.get('amount_grams')}g | {nutr_new.get('calories', '?')} kcal)"
            )

    # Clean up internal tracking flags
    for item in results:
        item.pop("_needs_research", None)
        item.pop("_gram_override", None)

    reranked["results"] = results
    if _any_changes or items_to_research:
        print(format_output(reranked))

    return reranked


def run_onboarding_survey(uid: str) -> None:
    """
    One-time personal profile setup.

    Asks the user about bodyweight, age, daily activity level (PAL) and
    structured training, then calculates their Total Daily Energy Expenditure
    and macro targets and persists the result to the database.

    Formulas
    --------
    RMR (kcal)       = (0.047 × weight_kg + 1.009 + 0.001452 × age + 3.2) × 239
    Activity (kcal)  = (PAL − 1) × RMR
    Training (kcal)  = (MET − 1) × weight_kg × daily_training_hours
    TDEE             = RMR + Activity + Training

    Macros
    ------
    Protein : 2.0 g × weight_kg  →  × 4 kcal/g
    Fat     : 0.9 g × weight_kg  →  × 9 kcal/g
    Carbs   : (TDEE − protein_kcal − fat_kcal) ÷ 4 kcal/g
    """
    print("\n" + "=" * 60)
    print("  👤  Personal Profile Setup")
    print("=" * 60)
    print("  Answer a few questions to set your daily calorie and")
    print("  macro targets.  This runs once per user ID.\n")

    # ── Weight ────────────────────────────────────────────────────────────
    while True:
        try:
            weight = float(input("  Your weight (kg): ").strip())
            if 20.0 < weight < 350.0:
                break
            print("  ❌ Please enter a realistic weight (20–350 kg).")
        except ValueError:
            print("  ❌ Please enter a number (e.g. 75 or 82.5).")

    # ── Sex ───────────────────────────────────────────────────────────────
    while True:
        sex = input("  Your sex (m/f): ").strip().lower()
        if sex in ("m", "male", "f", "female"):
            is_male = sex in ("m", "male")
            break
        print("  ❌ Please enter m or f.")

    # ── Age ───────────────────────────────────────────────────────────────
    while True:
        try:
            age = int(input("  Your age (years): ").strip())
            if 10 <= age <= 120:
                break
            print("  ❌ Please enter a realistic age.")
        except ValueError:
            print("  ❌ Please enter a whole number.")

    # ── Daily activity level (PAL) ────────────────────────────────────────
    print("""
  How active is your typical day?
  [1] Mostly sedentary – bed-bound or very limited mobility          (PAL 1.25)
  [2] Office work, mostly sitting, little or no exercise             (PAL 1.45)
  [3] Student / office worker, occasional walking or standing        (PAL 1.65)
  [4] Service work – sales, cooking, on your feet most of the day    (PAL 1.85)
  [5] Physical labour – construction, farming, or competitive sport   (PAL 2.20)""")

    _pal_map = {1: 1.25, 2: 1.45, 3: 1.65, 4: 1.85, 5: 2.20}
    while True:
        try:
            choice = int(input("\n  Activity level [1–5]: ").strip())
            if choice in _pal_map:
                pal = _pal_map[choice]
                break
            print("  ❌ Please enter a number from 1 to 5.")
        except ValueError:
            print("  ❌ Please enter a number.")

    # ── Structured training ───────────────────────────────────────────────
    trains = input("\n  Do you exercise or train regularly? [y/n]: ").strip().lower() in ("y", "yes")
    training_met = 0.0
    training_hours_per_week = 0.0

    if trains:
        print("""
  Training intensity:
  [1] Light    – gentle walking, yoga, stretching           (~2 METs)
  [2] Moderate – brisk walking, cycling, light gym work     (~4.5 METs)
  [3] Vigorous – running, HIIT, heavy weightlifting         (~8 METs)""")

        _met_map = {1: 2.0, 2: 4.5, 3: 8.0}
        while True:
            try:
                met_choice = int(input("\n  Training intensity [1–3]: ").strip())
                if met_choice in _met_map:
                    training_met = _met_map[met_choice]
                    break
                print("  ❌ Please enter 1, 2 or 3.")
            except ValueError:
                print("  ❌ Please enter a number.")

        while True:
            try:
                sessions = int(input("  Sessions per week: ").strip())
                minutes  = int(input("  Minutes per session: ").strip())
                if sessions >= 0 and minutes >= 0:
                    training_hours_per_week = sessions * minutes / 60.0
                    break
                print("  ❌ Please enter non-negative numbers.")
            except ValueError:
                print("  ❌ Please enter whole numbers.")

    # ── Energy calculations ───────────────────────────────────────────────
    # Resting Metabolic Rate (Müller formula; MJ/day × 239 → kcal/day)
    # Male:   0.047 × kg + 1.009 + 0.001452 × age + 3.2
    # Female: 0.047 × kg         + 0.001452 × age + 3.2
    if is_male:
        rmr_mj = 0.047 * weight + 1.009 + 0.001452 * age + 3.2
    else:
        rmr_mj = 0.047 * weight + 0.001452 * age + 3.2
    rmr_kcal = rmr_mj * 239.0

    # Lifestyle activity above resting
    activity_kcal = (pal - 1.0) * rmr_kcal

    # Net training energy (MET − 1 avoids double-counting the RMR baseline)
    daily_training_hours = training_hours_per_week / 7.0
    net_training_kcal = max(0.0, (training_met - 1.0) * weight * daily_training_hours)

    kcal_daily = round(rmr_kcal + activity_kcal + net_training_kcal, 0)

    # ── Macro targets ─────────────────────────────────────────────────────
    protein_g    = round(2.0 * weight, 1)
    fat_g        = round(0.9 * weight, 1)
    protein_kcal = protein_g * 4.0
    fat_kcal     = fat_g * 9.0
    carbs_g      = round(max(0.0, kcal_daily - protein_kcal - fat_kcal) / 4.0, 1)

    # ── Persist ───────────────────────────────────────────────────────────
    get_db().save_user_profile(
        uid=uid,
        weight_kg=weight,
        age_years=age,
        pal=pal,
        training_met=training_met,
        training_hours_per_week=training_hours_per_week,
        kcal_daily=kcal_daily,
        protein_daily=protein_g,
        fat_daily=fat_g,
        carbs_daily=carbs_g,
    )

    breakdown = f"RMR {rmr_kcal:.0f}  +  activity {activity_kcal:.0f}"
    if net_training_kcal > 0:
        breakdown += f"  +  training {net_training_kcal:.0f}"

    print(f"\n  ✅ Profile saved for '{uid}'")
    print("\n  📊 Your daily targets:")
    print(f"     Calories : {kcal_daily:.0f} kcal  ({breakdown})")
    print(f"     Protein  : {protein_g:.1f} g   (2 g × {weight:.0f} kg)")
    print(f"     Fat      : {fat_g:.1f} g   (0.9 g × {weight:.0f} kg)")
    print(f"     Carbs    : {carbs_g:.1f} g   (remaining calories ÷ 4)")
    print()


def _check_and_run_onboarding(uid: str, force: bool = False) -> None:
    """Run the profile survey for new users, or when force=True (update case)."""
    if force or get_db().get_user_profile(uid) is None:
        run_onboarding_survey(uid)


# ── Menu width ──────────────────────────────────────────────────────────────
_W = 60


def _progress_bar(consumed: float, target: float, width: int = 22) -> str:
    """ASCII progress bar. █ = normal fill, ▓ = over target, ░ = empty."""
    if target <= 0:
        return "░" * width
    frac = min(consumed / target, 1.0)
    filled = int(frac * width)
    char = "▓" if consumed > target else "█"
    return char * filled + "░" * (width - filled)


def _print_day_overview(uid: str, meal_date: str) -> None:
    """Macro progress bars + meal list for the day."""
    totals  = get_db().get_daily_totals(uid, meal_date)
    profile = get_db().get_user_profile(uid)
    meals   = get_db().get_meals_for_day(uid, meal_date)

    print("\n" + "=" * _W)
    print(f"  📊  Today's Overview – {meal_date}  (@{uid})")
    print("=" * _W)

    if profile:
        rows = [
            ("Calories", totals["calories"], profile["kcal_daily"],    "kcal"),
            ("Protein",  totals["protein"],  profile["protein_daily"], "g"),
            ("Fat",      totals["fat"],       profile["fat_daily"],     "g"),
            ("Carbs",    totals["carbs"],     profile["carbs_daily"],   "g"),
        ]
        for label, consumed, target, unit in rows:
            bar = _progress_bar(consumed, target)
            rem = target - consumed
            tag = "over" if rem < 0 else "left"
            pct = int(consumed / target * 100) if target > 0 else 0
            print(f"  {label:<9}  {bar}  {consumed:6.0f} / {target:.0f} {unit}  ({abs(rem):.0f} {tag}, {pct}%)")
    else:
        print(f"  Calories : {totals['calories']:.1f} kcal")
        print(f"  Protein  : {totals['protein']:.1f} g")
        print(f"  Fat      : {totals['fat']:.1f} g")
        print(f"  Carbs    : {totals['carbs']:.1f} g")
        print("  (complete your profile to see targets)")

    print()
    if meals:
        print(f"  Meals today ({len(meals)}):")
        for i, meal in enumerate(meals, 1):
            names = ", ".join(it["item_name"] for it in meal.get("items", [])[:3])
            if len(meal.get("items", [])) > 3:
                names += " …"
            print(f"  [{i}] {meal['logged_at'][11:16]}  {names:<33}  {meal['total_calories']:.0f} kcal")
    else:
        print("  No meals logged today yet.")
    print("=" * _W)


def _edit_meal_of_day(uid: str, meal_date: str) -> None:
    """Remove items or whole meals from today's log."""
    meals = get_db().get_meals_for_day(uid, meal_date)
    if not meals:
        print("\n  No meals logged today.")
        return

    print("\n" + "=" * _W)
    print(f"  ✏️   Edit a meal – {meal_date}")
    print("=" * _W)
    for i, meal in enumerate(meals, 1):
        names = ", ".join(it["item_name"] for it in meal.get("items", [])[:3])
        print(f"  [{i}] {meal['logged_at'][11:16]}  {names}  ({meal['total_calories']:.0f} kcal)")
    print()

    while True:
        raw = input("  Select meal [1–N] or Enter to cancel: ").strip()
        if not raw:
            return
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(meals):
                break
            print(f"  ❌ Choose 1–{len(meals)}.")
        except ValueError:
            print("  ❌ Enter a number.")

    meal  = meals[idx]
    items = meal.get("items", [])

    def _show():
        print(f"\n  Meal  {meal['logged_at'][11:16]}  —  {meal['total_calories']:.0f} kcal total")
        for i, it in enumerate(items, 1):
            print(f"   [{i}] {it['item_name']:<25}  {it['amount_grams']:.0f}g  {it['calories']:.0f} kcal")
        print()
        print("  d<N> remove item  |  dm delete whole meal  |  <N> edit grams  |  Enter → back")

    _show()

    while True:
        cmd = input("\n  > ").strip().lower()
        if not cmd:
            break

        if cmd == "dm":
            if input("  ❗ Delete this entire meal? [y/N]: ").strip().lower() in ("y", "yes"):
                get_db().delete_meal(meal["meal_id"])
                print("  🗑️  Meal deleted.")
            break

        if cmd.startswith("d"):
            try:
                i = int(cmd[1:]) - 1
                if 0 <= i < len(items):
                    it = items.pop(i)
                    get_db().delete_meal_item(it["item_id"], meal["meal_id"])
                    print(f"  🗑️  Removed: {it['item_name']}")
                    if items:
                        _show()
                    else:
                        print("  🗑️  All items removed — meal deleted.")
                        break
                print(f"  ❌ Choose 1–{len(items)}.")
            except ValueError:
                print("  ❌ Use d1, d2 … to remove an item.")
            continue

        try:
            i = int(cmd) - 1
            if i < 0 or i >= len(items):
                print(f"  ❌ Choose 1–{len(items)}.")
                continue
            it = items[i]
            raw_g = input(f"  New grams for '{it['item_name']}' (now {it['amount_grams']:.0f}g): ").strip()
            new_grams = float(raw_g)
            get_db().update_meal_item_grams(it["item_id"], meal["meal_id"], new_grams)
            scale = new_grams / (it["amount_grams"] or 100.0)
            it["calories"]     = round(it["calories"]     * scale, 1)
            it["amount_grams"] = new_grams
            print(f"  ✅ Updated to {new_grams:.0f}g  ({it['calories']:.0f} kcal)")
            _show()
        except ValueError:
            print("  ❌ Invalid input.")


def _run_voice_record(uid: str, duration: int | None = None) -> dict | None:
    """
    Run Step 0 (recording + Whisper) in an isolated subprocess to avoid the
    PortAudio/FAISS/OpenMP semaphore conflict on macOS.  Returns the voice_data
    dict on success, or None on failure.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        tmp_path = Path(tf.name)

    cmd = [
        sys.executable, "-m", "step0_voice_input.run",
        "--uid", uid,
        "--output", str(tmp_path),
        "--duration", str(duration or config.WHISPER_RECORD_SECONDS),
    ]
    env = {**__import__("os").environ, "SAYFIT_DEVMODE": "1"} if config.DEV_MODE else None

    ret = subprocess.run(cmd, cwd=str(config.ROOT_DIR), env=env)
    if ret.returncode != 0:
        print("  ❌ Recording failed.")
        tmp_path.unlink(missing_ok=True)
        return None

    with open(tmp_path) as f:
        voice_data = json.load(f)
    tmp_path.unlink(missing_ok=True)

    # persist intermediate output
    voice_path = config.OUTPUTS_DIR / "step0_voice_output.json"
    with open(voice_path, "w") as f:
        json.dump(voice_data, f, indent=2)
    if config.DEV_MODE:
        print(f"   💾 Voice output saved: {voice_path}")

    return voice_data


def run_main_menu(uid: str, use_llm: bool = True, duration: int | None = None) -> None:
    """Full interactive main-menu loop. Returns when the user chooses Exit."""
    _check_and_run_onboarding(uid)
    _set_last_user(uid)

    while True:
        meal_date = datetime.now().strftime("%Y-%m-%d")
        today_str = datetime.now().strftime("%a %d %b %Y")
        totals    = get_db().get_daily_totals(uid, meal_date)

        print("\n" + "=" * _W)
        print(f"  🍽️   SayFit  ·  {today_str}  ·  @{uid}")
        print("=" * _W)
        print(f"  Today so far:  {totals['calories']:.0f} kcal  │"
              f"  P {totals['protein']:.0f}g  F {totals['fat']:.0f}g  C {totals['carbs']:.0f}g")
        print("─" * _W)
        print("  [Enter]         Record voice 🎙️")
        print("  [1]             Type meal manually ⌨️")
        print("  [2]             Today's overview")
        print("  [3]             Edit a meal today")
        print("  [4]             Update my profile")
        print("  [5]             Change user")
        print("  [6]             Exit")
        print("─" * _W)

        choice = input("  > ").strip()

        # ── Voice record ───────────────────────────────────────────────────
        if choice == "":
            secs = duration or config.WHISPER_RECORD_SECONDS
            print(f"\n  🎙️  Recording for {secs}s … (speak now)")
            voice_data = _run_voice_record(uid, duration)
            if voice_data and voice_data.get("text"):
                text      = voice_data["text"]
                date_time = voice_data.get("date_time", datetime.now().isoformat())
                print(f"  📝 Transcribed: \"{text}\"")
                reranked = run_pipeline(text, date_time=date_time, uid=uid, use_llm=use_llm)
                if reranked.get("results"):
                    reranked = ask_user_corrections(reranked, uid=uid, use_llm=use_llm)
                    if config.DEV_MODE:
                        print("\n💾 [Step 5] Saving to database …")
                    get_db().save_pipeline_result(reranked, uid=uid, input_text=text, meal_date=meal_date)
                    get_db().print_daily_summary(uid, meal_date)

        # ── Type meal manually ─────────────────────────────────────────────
        elif choice == "1":
            text = input("\n  ⌨️  What did you eat? > ").strip()
            if not text:
                continue
            date_time = datetime.now().isoformat()
            reranked  = run_pipeline(text, date_time=date_time, uid=uid, use_llm=use_llm)
            if reranked.get("results"):
                reranked = ask_user_corrections(reranked, uid=uid, use_llm=use_llm)
                if config.DEV_MODE:
                    print("\n💾 [Step 5] Saving to database …")
                get_db().save_pipeline_result(reranked, uid=uid, input_text=text, meal_date=meal_date)
                get_db().print_daily_summary(uid, meal_date)

        # ── Overview ──────────────────────────────────────────────────────
        elif choice == "2":
            _print_day_overview(uid, meal_date)

        # ── Edit meal ─────────────────────────────────────────────────────
        elif choice == "3":
            _edit_meal_of_day(uid, meal_date)

        # ── Update profile ────────────────────────────────────────────────
        elif choice == "4":
            run_onboarding_survey(uid)

        # ── Change user ───────────────────────────────────────────────────
        elif choice == "5":
            new_uid = input("  New user ID: ").strip()
            if new_uid:
                uid = new_uid
                _check_and_run_onboarding(uid)
                _set_last_user(uid)
                print(f"  ✅ Switched to @{uid}")

        # ── Exit ──────────────────────────────────────────────────────────
        elif choice == "6":
            print("\n  👋 Goodbye! Stay fit!\n")
            break

        else:
            print("  ❌ Invalid choice.")


def main():
    parser = argparse.ArgumentParser(
        description="SayFit Alpha – Food Nutrition Logger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # interactive mode
  python main.py --record                           # record from mic → full pipeline
  python main.py --record --duration 15             # record 15 seconds
  python main.py --wav my_audio.wav                 # transcribe .wav → full pipeline
  python main.py --text "2 eggs and a banana"       # single query
  python main.py --input inputs/meal.json           # file input
  python main.py --build-index                      # build FAISS index only
        """,
    )
    parser.add_argument("--record", action="store_true", help="Record from microphone (Step 0)")
    parser.add_argument("--wav", type=str, help="Path to a .wav file to transcribe (Step 0)")
    parser.add_argument("--duration", type=int, default=None, help="Recording duration in seconds (with --record)")
    parser.add_argument("--input", type=str, help="Path to voice-recorder JSON")
    parser.add_argument("--text", type=str, help="Direct text input (what you ate)")
    parser.add_argument("--uid", type=str, default=None, help="User ID (prompted if not provided)")
    parser.add_argument("--default-uid", type=str, default=None,
                        help="Set a persistent default user ID for this session (overrides SAYFIT_DEFAULT_USER)")
    parser.add_argument("--update-profile", action="store_true",
                        help="Re-run the personal profile survey to update daily targets")
    parser.add_argument("--test-folder", type=str, help="Run all test JSONs in a folder")
    parser.add_argument("--no-llm", action="store_true",
                        help="Step 3 ohne LLM – regelbasierter Fallback (kein API-Key nötig)")
    parser.add_argument("--openai", action="store_true",
                        help="Use OpenAI instead of Groq (requires OPENAI_API_KEY in .env)")
    parser.add_argument("--locllm", action="store_true",
                        help="Use local Ollama instead of Groq (qwen2.5:7b)")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index and exit")
    parser.add_argument("--show-config", action="store_true", help="Print configuration and exit")
    parser.add_argument("--devmode", action="store_true",
                        help="Show verbose step-by-step debug output (developer mode)")
    args = parser.parse_args()

    # activate developer mode before any module import that might print
    if args.devmode:
        config.DEV_MODE = True

    # --default-uid overrides the env-backed DEFAULT_USER_ID for this session
    if args.default_uid:
        config.DEFAULT_USER_ID = args.default_uid

    # --record/--wav will prompt interactively; all other modes fall back silently.
    # Interactive menu (else branch) uses _get_last_user() itself, so leave uid=None there.
    if args.uid is None and not args.record and not args.wav:
        if args.text or args.input or args.test_folder:
            # non-interactive single-shot modes: use last-used uid as silent default
            args.uid = _get_last_user()
        # else: interactive menu — uid assigned in the else branch below

    # configure LLM backend (must happen before any pipeline step)
    if not args.no_llm:
        llm_client.configure(use_local=args.locllm, use_openai=args.openai)

    # show config
    if args.show_config:
        config.print_config()
        return

    # build index only
    if args.build_index:
        from step2_retrieval.build_index import build_index
        build_index()
        return

    # update profile only (no meal logging)
    if args.update_profile and not (args.record or args.wav or args.text or args.input):
        uid = args.uid or input("  Your User ID [default_user]: ").strip() or "default_user"
        _check_and_run_onboarding(uid, force=True)
        return

    # ensure index exists
    has_index = _ensure_index()

    print()
    if config.DEV_MODE:
        if not args.no_llm:
            config.print_config(
                active_extraction_model=llm_client.extraction_model(),
                active_reasoning_model=llm_client.reasoning_model(),
            )
        else:
            config.print_config()

    if args.record or args.wav:
        # ── Step 0: Voice input ──────────────────────────────────────
        if args.uid is None:
            args.uid = input(f"  Your User ID [{config.DEFAULT_USER_ID}]: ").strip() or config.DEFAULT_USER_ID
            print()
        _check_and_run_onboarding(args.uid, force=args.update_profile)
        if args.record:
            # Run recording+transcription in an isolated subprocess so that
            # PortAudio (sounddevice) and Whisper/PyTorch are fully torn down
            # before this process loads FAISS/OpenMP.  On macOS, both create
            # OS-level semaphore pools that conflict when loaded in the same
            # process, causing a segfault.
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w"
            ) as tf:
                tmp_path = Path(tf.name)

            cmd = [
                sys.executable, "-m", "step0_voice_input.run",
                "--uid", args.uid,
                "--output", str(tmp_path),
                "--duration", str(args.duration or config.WHISPER_RECORD_SECONDS),
            ]
            if config.DEV_MODE:
                # pass devmode env flag so _log() prints inside the subprocess
                env = {**__import__("os").environ, "SAYFIT_DEVMODE": "1"}
            else:
                env = None

            ret = subprocess.run(cmd, cwd=str(config.ROOT_DIR), env=env)
            if ret.returncode != 0:
                print("❌ Recording failed.")
                sys.exit(1)

            with open(tmp_path) as f:
                voice_data = json.load(f)
            tmp_path.unlink(missing_ok=True)
        else:
            from step0_voice_input.voice_recorder import transcribe_wav
            voice_data = transcribe_wav(args.wav, uid=args.uid)

        # save intermediate
        voice_path = config.OUTPUTS_DIR / "step0_voice_output.json"
        with open(voice_path, "w") as f:
            json.dump(voice_data, f, indent=2)
        if config.DEV_MODE:
            print(f"   💾 Voice output saved: {voice_path}")

        # feed into pipeline
        uid = voice_data.get("UID", args.uid)
        date_time = voice_data.get("date_time", datetime.now().isoformat())
        reranked = run_pipeline(
            text=voice_data["text"],
            date_time=date_time,
            uid=uid,
            use_llm=not args.no_llm,
        )
        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=uid, use_llm=not args.no_llm)

            # ── Step 5: Database ─────────────────────────────────────────
            if config.DEV_MODE: print("\n💾 [Step 5] Saving to database …")
            meal_date = date_time[:10]
            get_db().save_pipeline_result(reranked, uid=uid, input_text=voice_data["text"], meal_date=meal_date)
            get_db().print_daily_summary(uid, meal_date)

    elif args.input:
        # file-based mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            sys.exit(1)
        with open(input_path) as f:
            data = json.load(f)
        uid = data.get("UID", args.uid)
        date_time = data.get("date_time", datetime.now().isoformat())
        reranked = run_pipeline(
            text=data["text"],
            date_time=date_time,
            uid=uid,
            use_llm=not args.no_llm,
        )
        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=uid, use_llm=not args.no_llm)

            # ── Step 5: Database ─────────────────────────────────────────
            if config.DEV_MODE: print("\n💾 [Step 5] Saving to database …")
            meal_date = date_time[:10]
            get_db().save_pipeline_result(reranked, uid=uid, input_text=data["text"], meal_date=meal_date)
            get_db().print_daily_summary(uid, meal_date)


    elif args.test_folder:
        # ── Test mode: run all JSON inputs in folder ─────────────────────
        test_path = Path(args.test_folder)

        if not test_path.exists():
            print(f"❌ Test folder not found: {test_path}")
            sys.exit(1)

        files = sorted(test_path.glob("*.json"))

        if not files:
            print("❌ No JSON test files found.")
            sys.exit(1)

        batch_dir = _make_batch_dir()
        print(f"\n🧪 Running {len(files)} tests  →  {batch_dir}\n")

        created_run_dirs: list[Path] = []

        for f in files:
            run_name = f.stem          # z.B. "SayFit-Test_ENG_01"
            print("=" * 60)
            print(f"TEST: {f.name}")

            with open(f) as jf:
                data = json.load(jf)

            reranked = run_pipeline(
                text=data["text"],
                date_time=data.get("date_time", datetime.now().isoformat()),
                uid=data.get("UID", args.uid),
                meta={
                    "test_file": f.name,
                    "input_text": data["text"],
                    "test_folder": str(test_path.resolve()),
                },
                use_llm=not args.no_llm,
                run_parent=batch_dir,
                run_name=run_name,
            )

            created_run_dirs.append(batch_dir / run_name)
            print("=" * 60 + "\n")

        # ── Automatische Analyse nach dem Test-Lauf ────────────────────────
        if created_run_dirs:
            # Datei liegt neben dem Batch-Ordner, heißt genauso (z.B. testrun_001.txt)
            report_path = batch_dir.parent / f"{batch_dir.name}.txt"

            print("\n" + "=" * 60)
            print("📊 AUTOMATISCHE ANALYSE")
            print("=" * 60)

            from analyze_tests import run_analysis
            run_analysis(created_run_dirs, batch_dir, report_path)



    elif args.text:
        # direct text mode
        date_time = datetime.now().isoformat()
        reranked = run_pipeline(
            text=args.text,
            date_time=date_time,
            uid=args.uid,
            use_llm=not args.no_llm,
        )
        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=args.uid, use_llm=not args.no_llm)

            # ── Step 5: Database ─────────────────────────────────────────
            if config.DEV_MODE: print("\n💾 [Step 5] Saving to database …")
            meal_date = date_time[:10]
            get_db().save_pipeline_result(reranked, uid=args.uid, input_text=args.text, meal_date=meal_date)
            get_db().print_daily_summary(args.uid, meal_date)

    else:
        # interactive main-menu mode
        if not has_index:
            print("\n❌ Cannot run interactive mode without FAISS index. "
                  "Run: python main.py --build-index")
            sys.exit(1)
        uid = args.uid or _get_last_user()
        run_main_menu(uid, use_llm=not args.no_llm, duration=args.duration)


if __name__ == "__main__":
    main()
