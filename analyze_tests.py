"""
SayFit Alpha – Test Analyzer
=============================
Evaluates the results of a --test-folder run.

Requires: main.py must have been run with --test-folder so that each
run directory contains a meta.json.

Usage:
    python analyze_tests.py
        → analyzes outputs/<today>/testrun_001/ (latest batch)

    python analyze_tests.py --batch-dir outputs/2026-03-10/testrun_001
        → analyzes a specific batch folder directly

    python analyze_tests.py --runs-dir outputs/2026-03-10
        → finds all testrun_* folders in a date directory
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _bar(value: float, max_value: float, width: int = 20, char: str = "█") -> str:
    if max_value == 0:
        filled = 0
    else:
        filled = int(round(value / max_value * width))
    return char * filled + "░" * (width - filled)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  0.0%"
    return f"{n / total * 100:5.1f}%"


# ── Load run data ─────────────────────────────────────────────────────────────

def collect_runs(
    batch_dir: Path,
    last_n: int | None = None,
    specific_dirs: list[Path] | None = None,
) -> list[dict]:
    """
    Collects run dicts from a batch directory.
    - specific_dirs: explicit list of run paths (passed from main.py)
    - batch_dir + last_n: auto-discover all subdirs with meta.json
    """
    if specific_dirs is not None:
        test_run_dirs = sorted(specific_dirs)
    else:
        # any subdir that has a meta.json is a valid test run
        all_subdirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
        test_run_dirs = [d for d in all_subdirs if (d / "meta.json").exists()]

        if not test_run_dirs:
            print(
                f"⚠️  No test runs found in {batch_dir}.\n"
                "   Hint: Run 'python main.py --test-folder ...' first."
            )
            sys.exit(1)

        if last_n is not None:
            test_run_dirs = test_run_dirs[-last_n:]

    runs = []
    for run_dir in test_run_dirs:
        meta    = load_json(run_dir / "meta.json") or {}
        step1   = load_json(run_dir / "step1_extraction_output.json") or {}
        step1_5 = load_json(run_dir / "step1_5_ontology_output.json") or {}
        step2   = load_json(run_dir / "step2_retrieval_output.json") or {}
        step3   = load_json(run_dir / "step3_reranker_output.json") or {}

        runs.append({
            "run_dir":    run_dir.name,
            "meta":       meta,
            "extraction": step1,
            "ontology":   step1_5,
            "retrieval":  step2,
            "reranker":   step3,
        })

    return runs


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_run(run: dict) -> dict:
    meta    = run["meta"]
    step1   = run["extraction"]
    step1_5 = run["ontology"]
    step2   = run["retrieval"]
    step3   = run["reranker"]

    results  = step3.get("results", [])
    items_ex = step1.get("items", {})

    # ── ontology: predicted category per item_name ────────────────────────
    ont_items = step1_5.get("ontology", {})
    ont_by_name: dict[str, dict] = {}
    for v in ont_items.values():
        ont_by_name[v.get("item_name", "").lower()] = v

    # ── retrieval: top adjusted_score + cat_match rate per query ──────────
    retrieval_items = step2.get("items", [])
    top_scores:    dict[str, float] = {}
    cat_match_rates: dict[str, float] = {}  # fraction of top-K with cat_match
    for ri in retrieval_items:
        query      = ri.get("query", "")
        candidates = ri.get("candidates", [])
        if candidates:
            top_scores[query] = candidates[0].get("adjusted_score", 0.0)
            matched = sum(1 for c in candidates if c.get("cat_match", False))
            cat_match_rates[query] = matched / len(candidates)

    item_details = []
    for res in results:
        item_name  = res.get("item_name", "?")
        matched    = res.get("matched_name", "?")
        confidence = res.get("confidence", "?")
        conf_note  = res.get("confidence_note", "")
        grams      = res.get("amount_grams", None)
        source     = res.get("source", "?")
        nutr       = res.get("nutrition", {})
        proc_desc  = res.get("processing_description", "")

        # map back to ontology prediction
        ont = ont_by_name.get(item_name.lower(), {})
        cat_l1     = ont.get("predicted_cat_l1", "")
        cat_l2     = ont.get("predicted_cat_l2", "")
        ont_source = ont.get("source", "")   # "llm" | "lookup" | ""

        query_key  = f"{item_name} ({proc_desc})" if proc_desc else item_name
        top_score  = top_scores.get(query_key, None)
        cat_match_rate = cat_match_rates.get(query_key, None)
        if top_score is None:
            for k, v in top_scores.items():
                if item_name.lower() in k.lower():
                    top_score = v
                    cat_match_rate = cat_match_rates.get(k)
                    break

        item_details.append({
            "item_name":      item_name,
            "matched":        matched,
            "confidence":     confidence,
            "conf_note":      conf_note,
            "grams":          grams,
            "source":         source,
            "top_score":      top_score,
            "cat_l1":         cat_l1,
            "cat_l2":         cat_l2,
            "ont_source":     ont_source,
            "cat_match_rate": cat_match_rate,
            "calories":       nutr.get("calories"),
            "protein":        nutr.get("protein"),
            "fat":            nutr.get("fat"),
            "carbs":          nutr.get("carbs"),
        })

    return {
        "run_dir":     run["run_dir"],
        "test_file":   meta.get("test_file", "unknown"),
        "input_text":  meta.get("input_text", ""),
        "n_extracted": len(items_ex),
        "n_results":   len(results),
        "success":     len(results) > 0,
        "items":       item_details,
    }


# ── Output ────────────────────────────────────────────────────────────────────

SEP  = "─" * 72
DSEP = "═" * 72

CONF_EMOJI = {"high": "✅", "medium": "🟡", "low": "🔴"}
SRC_SHORT  = {"usda": "USDA", "openfoodfacts": "OFF "}


def print_header(title: str, subtitle: str = ""):
    print()
    print(DSEP)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(DSEP)


def print_per_test(analyzed: list[dict]):
    print_header("RESULTS PER TEST", f"{len(analyzed)} tests analyzed")

    for i, a in enumerate(analyzed, 1):
        print()
        status = "✅" if a["success"] else "❌"
        print(f"  TEST {i}/{len(analyzed)}  {status}  │  {a['test_file']}  [{a['run_dir']}]")
        text = a["input_text"]
        if len(text) > 68:
            text = text[:65] + "..."
        print(f"  Input:  \"{text}\"")
        print(f"  Extracted: {a['n_extracted']}  │  With result: {a['n_results']}")

        if not a["items"]:
            print("  ⚠️  No result – pipeline returned no items.")
            print(SEP)
            continue

        print()
        hdr = (f"  {'#':<3} {'Item':<16} {'→ Match':<20} {'Conf':>4} "
               f"{'g':>5} {'kcal':>5} {'Score':>6} {'Cat-L1':<20} {'Ont':>5}")
        print(hdr)
        print("  " + "─" * 82)

        for j, item in enumerate(a["items"], 1):
            conf_ico   = CONF_EMOJI.get(item["confidence"], "?")
            score_str  = f"{item['top_score']:.3f}"  if item["top_score"]      is not None else "  n/a"
            grams_str  = f"{item['grams']:.0f}"      if item["grams"]          is not None else "  ?"
            cal_str    = f"{item['calories']:.0f}"   if item["calories"]       is not None else "  ?"
            cat_str    = item["cat_l1"][:19]         if item["cat_l1"]         else "?"
            ont_src    = item["ont_source"][:5]      if item["ont_source"]     else "?"
            match_pct  = (f"{item['cat_match_rate']*100:.0f}%"
                          if item["cat_match_rate"] is not None else "")

            row = (
                f"  {j:<3} "
                f"{item['item_name'][:15]:<16} "
                f"{item['matched'][:19]:<20} "
                f"{conf_ico:>4} "
                f"{grams_str:>5} "
                f"{cal_str:>5} "
                f"{score_str:>6} "
                f"{cat_str:<20} "
                f"{ont_src:>5}"
            )
            print(row)

            # secondary line: cat_l2 + category match rate + conf note
            details = []
            if item["cat_l2"]:
                details.append(f"L2: {item['cat_l2']}")
            if match_pct:
                details.append(f"cat-match: {match_pct} of top-K")
            if item["conf_note"]:
                details.append(item["conf_note"])
            if details:
                print(f"       ↳ {' │ '.join(details)}")

        print(SEP)


def print_summary(analyzed: list[dict]):
    total      = len(analyzed)
    successful = sum(1 for a in analyzed if a["success"])
    failed     = total - successful

    all_items = [item for a in analyzed for item in a["items"]]
    n_items   = len(all_items)

    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for item in all_items:
        c = item["confidence"]
        if c in conf_counts:
            conf_counts[c] += 1

    src_counts = {}
    for item in all_items:
        s = item["source"]
        src_counts[s] = src_counts.get(s, 0) + 1

    scores    = [item["top_score"] for item in all_items if item["top_score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    avg_items = n_items / total if total else 0.0

    low_conf_items = [
        (a["test_file"], item)
        for a in analyzed
        for item in a["items"]
        if item["confidence"] == "low"
    ]
    failed_tests = [a for a in analyzed if not a["success"]]

    print_header("SUMMARY")

    print(f"  Tests total:         {total}")
    print(f"  Successful:          {successful}/{total}  ({_pct(successful, total)})")
    print(f"  Failed:              {failed}")
    print()
    print(f"  Items total:         {n_items}")
    print(f"  Avg items / test:    {avg_items:.1f}")
    print()

    print("  Confidence distribution:")
    for level in ("high", "medium", "low"):
        n   = conf_counts[level]
        bar = _bar(n, n_items)
        ico = CONF_EMOJI.get(level, "?")
        print(f"    {ico} {level:<8} {n:>3}  {bar}  {_pct(n, n_items)}")
    print()

    print("  Database distribution:")
    for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
        bar = _bar(n, n_items)
        print(f"    {SRC_SHORT.get(src, src):<6}  {n:>3}  {bar}  {_pct(n, n_items)}")
    print()

    print("  Retrieval score (top-1 candidate):")
    print(f"    Avg {avg_score:.3f}  │  Min {min_score:.3f}  │  Max {max_score:.3f}")
    if scores:
        score_bar = _bar(avg_score, 1.0, width=30)
        print(f"    {score_bar}  (scale 0–1)")
    print()

    # ── Ontology filter stats ─────────────────────────────────────────────
    llm_classified    = sum(1 for i in all_items if i["ont_source"] == "llm")
    lookup_classified = sum(1 for i in all_items if i["ont_source"] == "lookup")
    unclassified      = n_items - llm_classified - lookup_classified

    cat_match_vals = [i["cat_match_rate"] for i in all_items if i["cat_match_rate"] is not None]
    avg_cat_match  = sum(cat_match_vals) / len(cat_match_vals) if cat_match_vals else 0.0

    print("  Ontology filter (Step 1.5):")
    print(f"    Source – LLM: {llm_classified}  │  Lookup: {lookup_classified}  │  None: {unclassified}")
    print(f"    Avg category-match rate in top-K candidates: {avg_cat_match*100:.1f}%")
    print()

    # ── L1 category breakdown ─────────────────────────────────────────────
    cat_counts_l1: dict[str, int] = {}
    for item in all_items:
        c = item["cat_l1"] or "unknown"
        cat_counts_l1[c] = cat_counts_l1.get(c, 0) + 1

    print("  L1 category distribution (predicted):")
    for cat, n in sorted(cat_counts_l1.items(), key=lambda x: -x[1]):
        bar = _bar(n, n_items, width=16)
        print(f"    {cat:<28} {n:>3}  {bar}  {_pct(n, n_items)}")
    print()

    if low_conf_items or failed_tests:
        print(SEP)
        print("  ⚠️  WARNINGS")
        print()

        if failed_tests:
            print("  Failed tests (no result):")
            for a in failed_tests:
                print(f"    ✗  {a['test_file']}  – \"{a['input_text'][:50]}\"")
            print()

        if low_conf_items:
            print("  Low-confidence items:")
            for test_file, item in low_conf_items:
                note = f" → {item['conf_note']}" if item["conf_note"] else ""
                print(f"    🔴  [{test_file}]  {item['item_name']} → {item['matched']}{note}")
        print()

    print(DSEP)
    print()


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(analyzed: list[dict], runs_dir: Path, filename: str | None = None) -> Path:
    report = {
        "generated_at": datetime.now().isoformat(),
        "runs_dir": str(runs_dir),
        "total_tests": len(analyzed),
        "tests": analyzed,
    }
    if filename is None:
        filename = f"test_analysis_{datetime.now().strftime('%H%M%S')}.json"
    out_path = runs_dir / filename
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📄 Report saved: {out_path}")
    return out_path


# ── Public API for main.py ────────────────────────────────────────────────────

class _Tee:
    """Writes simultaneously to stdout and a file."""
    def __init__(self, file):
        self._file   = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def run_analysis(run_dirs: list[Path], batch_dir: Path, report_path: Path) -> None:
    """
    Called from main.py after a --test-folder run.
    Prints analysis to console AND saves it as a .txt file.
    """
    runs     = collect_runs(batch_dir, specific_dirs=run_dirs)
    analyzed = [analyze_run(r) for r in runs]

    with open(report_path, "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = _Tee(f)
        try:
            print_per_test(analyzed)
            print_summary(analyzed)
        finally:
            sys.stdout = old_stdout

    print(f"  📄 Report saved: {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SayFit Test Analyzer – evaluates --test-folder runs."
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Path to a specific testrun_NNN/ folder (e.g. outputs/2026-03-10/testrun_001)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Date folder to scan for testrun_* batches (e.g. outputs/2026-03-10). "
             "Uses the latest batch found. Default: outputs/<today>",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Only analyze the last N test runs within the batch (optional)",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Also save a JSON report alongside the batch folder",
    )
    args = parser.parse_args()

    # Resolve batch directory
    if args.batch_dir:
        batch_dir = args.batch_dir
    else:
        if args.runs_dir:
            date_dir = args.runs_dir
        else:
            today    = datetime.now().strftime("%Y-%m-%d")
            date_dir = Path("outputs") / today

        if not date_dir.exists():
            print(f"❌ Directory not found: {date_dir}")
            sys.exit(1)

        batches = sorted(date_dir.glob("testrun_*"))
        if not batches:
            print(f"❌ No testrun_* folders found in {date_dir}")
            sys.exit(1)

        batch_dir = batches[-1]   # latest batch

    if not batch_dir.exists():
        print(f"❌ Batch folder not found: {batch_dir}")
        sys.exit(1)

    print(f"\n🔍 Analyzing: {batch_dir}")

    runs     = collect_runs(batch_dir, last_n=args.last_n)
    analyzed = [analyze_run(r) for r in runs]

    if args.save_report:
        report_path = batch_dir.parent / f"{batch_dir.name}.txt"
        run_analysis(
            run_dirs=[batch_dir / r["run_dir"] for r in runs],
            batch_dir=batch_dir,
            report_path=report_path,
        )
    else:
        print_per_test(analyzed)
        print_summary(analyzed)


if __name__ == "__main__":
    main()
