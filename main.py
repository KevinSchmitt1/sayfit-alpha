"""
SayFit Alpha – Main Pipeline
==============================
Orchestrates all steps:
  1. Item Extraction (LLM)       → structured food items from raw text
  2. Retrieval (FAISS/RAG)       → top-K food candidates from databases
  3. Reranker (LLM)              → best match + portion estimate + macros
  4. Output                      → formatted table + daily totals

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
from step1_extraction.extractor import extract_items
from step3_reranker.reranker import rerank_all
from step3_reranker.calibration import save_user_correction
from step4_output.formatter import format_output, save_log


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


def run_pipeline(text: str, date_time: str = "", uid: str = "default_user"):
    """
    Run the full pipeline on a single text input.

    Returns the final reranker output dict.
    """
    # ── Step 1: Extraction ───────────────────────────────────────────────
    extraction = extract_items(text, date_time=date_time, uid=uid)

    # save intermediate
    ext_path = config.OUTPUTS_DIR / "step1_extraction_output.json"
    with open(ext_path, "w") as f:
        json.dump(extraction, f, indent=2)
    print(f"   💾 Extraction saved: {ext_path}")

    # ── Step 2: Retrieval ────────────────────────────────────────────────
    queries = extraction.get("queries", [])
    if not queries:
        print("   ⚠️  No queries extracted – nothing to retrieve.")
        return {"results": []}

    from step2_retrieval.retriever import retrieve
    retrieval = retrieve(queries)

    ret_path = config.OUTPUTS_DIR / "step2_retrieval_output.json"
    with open(ret_path, "w") as f:
        json.dump(retrieval, f, indent=2)
    print(f"   💾 Retrieval saved: {ret_path}")

    # ── Step 3: Reranker ─────────────────────────────────────────────────
    reranked = rerank_all(extraction, retrieval)

    rer_path = config.OUTPUTS_DIR / "step3_reranker_output.json"
    with open(rer_path, "w") as f:
        json.dump(reranked, f, indent=2)
    print(f"   💾 Reranker saved: {rer_path}")

    # ── Step 4: Output ───────────────────────────────────────────────────
    output_text = format_output(reranked)
    print(output_text)

    return reranked


def ask_user_corrections(reranked: dict, uid: str = "default_user") -> dict:
    """
    Interactively ask the user if the results are correct.
    If not, let them correct amounts and save calibrations.
    """
    results = reranked.get("results", [])
    if not results:
        return reranked

    print("\n" + "=" * 60)
    print("  ✏️  Review & Correct")
    print("=" * 60)
    print("  Is this correct? Enter the item number to correct it,")
    print("  or press Enter to accept all results.")
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
            # save final log
            save_log(reranked)

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
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index and exit")
    parser.add_argument("--show-config", action="store_true", help="Print configuration and exit")
    args = parser.parse_args()

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
        reranked = run_pipeline(
            text=voice_data["text"],
            date_time=voice_data.get("date_time", datetime.now().isoformat()),
            uid=voice_data.get("UID", args.uid),
        )
        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=voice_data.get("UID", args.uid))
            save_log(reranked)

    elif args.input:
        # file-based mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            sys.exit(1)
        with open(input_path) as f:
            data = json.load(f)
        reranked = run_pipeline(
            text=data["text"],
            date_time=data.get("date_time", datetime.now().isoformat()),
            uid=data.get("UID", args.uid),
        )
        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=data.get("UID", args.uid))
            save_log(reranked)

    elif args.text:
        # direct text mode
        reranked = run_pipeline(
            text=args.text,
            date_time=datetime.now().isoformat(),
            uid=args.uid,
        )
        if reranked.get("results"):
            reranked = ask_user_corrections(reranked, uid=args.uid)
            save_log(reranked)

    else:
        # interactive mode
        if not has_index:
            print("\n❌ Cannot run interactive mode without FAISS index. "
                  "Run: python main.py --build-index")
            sys.exit(1)
        interactive_mode()


if __name__ == "__main__":
    main()
