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
import json
import sys
from datetime import datetime
from pathlib import Path

import config
import llm_client
from step5_database.database import get_db
from step1_extraction.extractor import extract_items
from step3_reranker.reranker import rerank_all
from step3_reranker.calibration import save_user_correction
from step4_output.formatter import format_output


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
    print(f"   📁 Run output: {run_dir}")

    # Save test metadata if provided (used by analyze_tests.py)
    if meta:
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ── Step 1: Extraction ───────────────────────────────────────────────
    extraction = extract_items(text, date_time=date_time, uid=uid, use_llm=use_llm)

    # save intermediate
    ext_path = run_dir / "step1_extraction_output.json"
    with open(ext_path, "w") as f:
        json.dump(extraction, f, indent=2)
    print(f"   💾 Extraction saved: {ext_path}")

    # ── Step 1.5: Ontology Filter ─────────────────────────────────────────
    from step1_5_ontology_filter.ontology_filter import apply_ontology_filter
    extraction = apply_ontology_filter(extraction)

    ont_path = run_dir / "step1_5_ontology_output.json"
    with open(ont_path, "w") as f:
        json.dump(extraction, f, indent=2)
    print(f"   💾 Ontology saved: {ont_path}")

    # ── Step 2: Retrieval ────────────────────────────────────────────────
    queries = extraction.get("queries", [])
    if not queries:
        print("   ⚠️  No queries extracted – nothing to retrieve.")
        return {"results": []}

    category_hints = extraction.get("category_hints", [])

    from step2_retrieval.retriever import retrieve
    retrieval = retrieve(queries, category_hints=category_hints)

    ret_path = run_dir / "step2_retrieval_output.json"
    with open(ret_path, "w") as f:
        json.dump(retrieval, f, indent=2)
    print(f"   💾 Retrieval saved: {ret_path}")

    # ── Step 3: Reranker ─────────────────────────────────────────────────
    reranked = rerank_all(extraction, retrieval, use_llm=use_llm)

    rer_path = run_dir / "step3_reranker_output.json"
    with open(rer_path, "w") as f:
        json.dump(reranked, f, indent=2)
    print(f"   💾 Reranker saved: {rer_path}")

    # ── Step 4: Output ───────────────────────────────────────────────────
    output_text = format_output(reranked)
    print(output_text)

    return reranked


def ask_user_corrections(reranked: dict, uid: str = "default_user") -> dict:
    """
    Interactively ask the user to correct results — but ONLY if at least one
    item has low confidence (really bad match). High / medium → auto-accept.
    """
    results = reranked.get("results", [])
    if not results:
        return reranked

    low_conf_items = [r for r in results if r.get("confidence") == "low"]
    if not low_conf_items:
        return reranked

    print("\n" + "=" * 60)
    print("  ⚠️  Poor match detected – please review")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        if r.get("confidence") == "low":
            print(f"  [{i}] {r.get('item_name')} → {r.get('matched_name')} "
                  f"({r.get('amount_grams', '?')}g)  ← LOW confidence")
    print()
    print("  Enter item number to correct, or press Enter to accept as-is.")
    print("  Type 'q' to finish.\n")

    while True:
        choice = input("  Correct item # (or Enter to accept / q to quit): ").strip()
        if choice in ("", "q", "quit"):
            break

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(results):
                print(f"  ❌ Invalid item number. Choose 1–{len(results)}.")
                continue
        except ValueError:
            print("  ❌ Please enter a number.")
            continue

        item = results[idx]
        print(f"\n  Currently: {item['item_name']} → {item['matched_name']} "
              f"({item.get('amount_grams', '?')}g)")

        new_grams = input("  New amount in grams (or Enter to keep): ").strip()
        if new_grams:
            try:
                grams = float(new_grams)
                item["amount_grams"] = grams
                item["unit"] = "g"

                # recalculate nutrition from the matched candidate's per-100g values
                # We need to look up the original per-100g data
                # For now, scale proportionally from existing values
                old_grams = item.get("_old_grams", item.get("amount_grams", 100))
                if old_grams and old_grams > 0:
                    # store original grams for re-scaling
                    nutr = item.get("nutrition", {})
                    # scale: we recalculate from the original data
                    # The nutrition in the item is for old_grams. We need per-100g first.
                    pass

                # save user calibration
                save_user_correction(uid, item["item_name"], grams)
                print(f"  ✅ Updated: {item['item_name']} → {grams}g")
            except ValueError:
                print("  ❌ Invalid number. Skipping.")
                continue

        new_name = input("  New matched food name (or Enter to keep): ").strip()
        if new_name:
            item["matched_name"] = new_name
            print(f"  ✅ Updated match: {new_name}")

    # re-render output
    output_text = format_output(reranked)
    print(output_text)

    return reranked


def interactive_mode():
    """Interactive terminal loop for logging meals."""
    print("\n" + "=" * 60)
    print("  🍽️  SayFit Alpha – Interactive Mode")
    print("=" * 60)
    print("  Type what you ate (or 'quit' to exit).")
    print("  Example: \"i ate a pepperoni pizza and 3 eggs\"\n")

    uid = input("  Your User ID [default_user]: ").strip() or "default_user"
    print()

    while True:
        text = input("  🎤 What did you eat? > ").strip()
        if not text or text.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye! Stay fit!")
            break

        date_time = datetime.now().isoformat()
        reranked = run_pipeline(text, date_time=date_time, uid=uid)

        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=uid)

            # ── Step 5: Database ─────────────────────────────────────────
            print("\n💾 [Step 5] Saving to database …")
            meal_date = date_time[:10]
            get_db().save_pipeline_result(reranked, uid=uid, input_text=text, meal_date=meal_date)
            get_db().print_daily_summary(uid, meal_date)

        print("\n" + "-" * 60 + "\n")


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
    parser.add_argument("--uid", type=str, default="default_user", help="User ID")
    parser.add_argument("--test-folder", type=str, help="Run all test JSONs in a folder")
    parser.add_argument("--no-llm", action="store_true",
                        help="Step 3 ohne LLM – regelbasierter Fallback (kein API-Key nötig)")
    parser.add_argument("--openai", action="store_true",
                        help="Use OpenAI instead of Groq (requires OPENAI_API_KEY in .env)")
    parser.add_argument("--locllm", action="store_true",
                        help="Use local Ollama instead of Groq (qwen2.5:7b)")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index and exit")
    parser.add_argument("--show-config", action="store_true", help="Print configuration and exit")
    args = parser.parse_args()

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

    # ensure index exists
    has_index = _ensure_index()

    print()
    if not args.no_llm:
        config.print_config(
            active_extraction_model=llm_client.extraction_model(),
            active_reasoning_model=llm_client.reasoning_model(),
        )
    else:
        config.print_config()

    if args.record or args.wav:
        # ── Step 0: Voice input ──────────────────────────────────────
        from step0_voice_input.voice_recorder import record_and_transcribe, transcribe_wav

        if args.wav:
            voice_data = transcribe_wav(args.wav, uid=args.uid)
        else:
            duration = args.duration or config.WHISPER_RECORD_SECONDS
            voice_data = record_and_transcribe(duration=duration, uid=args.uid)

        # save intermediate
        voice_path = config.OUTPUTS_DIR / "step0_voice_output.json"
        with open(voice_path, "w") as f:
            json.dump(voice_data, f, indent=2)
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
            reranked = ask_user_corrections(reranked, uid=uid)

            # ── Step 5: Database ─────────────────────────────────────────
            print("\n💾 [Step 5] Saving to database …")
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
            reranked = ask_user_corrections(reranked, uid=uid)

            # ── Step 5: Database ─────────────────────────────────────────
            print("\n💾 [Step 5] Saving to database …")
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
            reranked = ask_user_corrections(reranked, uid=args.uid)

            # ── Step 5: Database ─────────────────────────────────────────
            print("\n💾 [Step 5] Saving to database …")
            meal_date = date_time[:10]
            get_db().save_pipeline_result(reranked, uid=args.uid, input_text=args.text, meal_date=meal_date)
            get_db().print_daily_summary(args.uid, meal_date)

    else:
        # interactive mode
        if not has_index:
            print("\n❌ Cannot run interactive mode without FAISS index. "
                  "Run: python main.py --build-index")
            sys.exit(1)
        interactive_mode()


if __name__ == "__main__":
    main()
