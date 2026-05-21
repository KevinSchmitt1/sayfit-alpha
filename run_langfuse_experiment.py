"""
Full-pipeline Langfuse experiment runner (Langfuse SDK v4).

Reads langfuse_dataset.csv, uploads it to a Langfuse dataset (if not already
present), then runs the complete SayFit pipeline (Steps 1→3) on every item
using langfuse.run_experiment() and logs scored results as a named run.

Usage
-----
    # First run: upload dataset + run experiment
    python run_langfuse_experiment.py --run-name "baseline-v1"

    # Upload dataset only (no pipeline run)
    python run_langfuse_experiment.py --upload-only

    # Run without re-uploading (dataset already in Langfuse)
    python run_langfuse_experiment.py --run-name "prompt-v2" --no-upload

    # Only test the first 3 rows (faster iteration)
    python run_langfuse_experiment.py --run-name "smoke" --limit 3

Scores logged per item
----------------------
    kcal_accuracy   : 1 − |pred − exp| / exp  (clamped to [0, 1]; higher = better)
    protein_accuracy: same for protein_g
    fat_accuracy    : same for fat_g
    carbs_accuracy  : same for carbs_g
    item_count_match: 1.0 if predicted item count == expected item count, else 0.0
"""

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import llm_client  # noqa: E402


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _rel_accuracy(predicted: float, expected: float) -> float:
    if expected == 0:
        return 1.0 if predicted == 0 else 0.0
    return max(0.0, 1.0 - abs(predicted - expected) / expected)


def _pred_totals(result: dict) -> dict:
    totals = {"kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0}
    for r in result.get("results", []):
        nutr = r.get("nutrition", {})
        totals["kcal"]      += nutr.get("calories") or 0
        totals["protein_g"] += nutr.get("protein")  or 0
        totals["fat_g"]     += nutr.get("fat")       or 0
        totals["carbs_g"]   += nutr.get("carbs")     or 0
    return totals


# ── Langfuse task ─────────────────────────────────────────────────────────────

def make_pipeline_task():
    """Return the task function (imported late to avoid triggering index load)."""
    from main import run_pipeline  # noqa: PLC0415

    def pipeline_task(*, item, **kwargs):
        input_data = item.input if hasattr(item, "input") else item.get("input", {})
        text = input_data["text"] if isinstance(input_data, dict) else input_data

        buf = StringIO()
        with redirect_stdout(buf):
            result = run_pipeline(
                text=text,
                uid="langfuse_experiment",
                use_llm=True,
            )
        return result

    return pipeline_task


# ── Langfuse evaluators ───────────────────────────────────────────────────────

def run_averages(*, item_results):
    """Run-level evaluator: computes per-metric averages across all items.

    These are stored as dataset-run-level scores (with dataset_run_id), which
    is what Langfuse's comparison view aggregates and displays.
    """
    from langfuse import Evaluation  # noqa: PLC0415

    metric_names = ["kcal_accuracy", "protein_accuracy", "fat_accuracy", "carbs_accuracy", "item_count_match"]
    sums: dict[str, float] = {m: 0.0 for m in metric_names}
    counts: dict[str, int] = {m: 0 for m in metric_names}

    for item_result in item_results:
        for evaluation in item_result.evaluations:
            if evaluation.name in sums and isinstance(evaluation.value, (int, float)):
                sums[evaluation.name] += evaluation.value
                counts[evaluation.name] += 1

    return [
        Evaluation(
            name=name,
            value=round(sums[name] / counts[name], 4) if counts[name] > 0 else 0.0,
            comment=f"avg over {counts[name]} items",
        )
        for name in metric_names
    ]


def evaluator_suite(*, input, output, expected_output, metadata, **kwargs):
    from langfuse import Evaluation  # noqa: PLC0415

    if not isinstance(output, dict):
        return [Evaluation(name="kcal_accuracy", value=0.0, comment="pipeline returned no dict")]

    expected = (
        json.loads(expected_output)
        if isinstance(expected_output, str)
        else (expected_output or {})
    )

    pred   = _pred_totals(output)
    exp_t  = expected.get("total", {})
    n_pred = len(output.get("results", []))
    n_exp  = len(expected.get("items", []))

    return [
        Evaluation(
            name="kcal_accuracy",
            value=round(_rel_accuracy(pred["kcal"],      exp_t.get("kcal", 0)),      4),
            comment=f"pred {pred['kcal']:.0f} kcal vs exp {exp_t.get('kcal', 0):.0f} kcal",
        ),
        Evaluation(
            name="protein_accuracy",
            value=round(_rel_accuracy(pred["protein_g"], exp_t.get("protein_g", 0)), 4),
            comment=f"pred {pred['protein_g']:.1f}g vs exp {exp_t.get('protein_g', 0):.1f}g",
        ),
        Evaluation(
            name="fat_accuracy",
            value=round(_rel_accuracy(pred["fat_g"],     exp_t.get("fat_g", 0)),     4),
            comment=f"pred {pred['fat_g']:.1f}g vs exp {exp_t.get('fat_g', 0):.1f}g",
        ),
        Evaluation(
            name="carbs_accuracy",
            value=round(_rel_accuracy(pred["carbs_g"],   exp_t.get("carbs_g", 0)),   4),
            comment=f"pred {pred['carbs_g']:.1f}g vs exp {exp_t.get('carbs_g', 0):.1f}g",
        ),
        Evaluation(
            name="item_count_match",
            value=1.0 if n_pred == n_exp else 0.0,
            comment=f"predicted {n_pred} items, expected {n_exp}",
        ),
    ]


# ── Dataset upload ────────────────────────────────────────────────────────────

def upload_dataset(lf, dataset_name: str, csv_path: Path) -> None:
    try:
        lf.create_dataset(
            name=dataset_name,
            description="SayFit full-pipeline evaluation dataset (meal descriptions → nutrition)",
        )
        print(f"  Created dataset '{dataset_name}' in Langfuse.")
    except Exception:
        print(f"  Dataset '{dataset_name}' already exists – reusing it.")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    for i, row in enumerate(rows, 1):
        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={"text": row["input"]},
            expected_output=json.loads(row["expected_output"]),
            metadata={
                "difficulty":     row.get("difficulty", ""),
                "challenge_type": row.get("challenge_type", ""),
                "total_kcal":     float(row.get("total_kcal", 0)),
                "total_protein_g": float(row.get("total_protein_g", 0)),
                "total_fat_g":    float(row.get("total_fat_g", 0)),
                "total_carbs_g":  float(row.get("total_carbs_g", 0)),
            },
        )
        print(f"  [{i}/{len(rows)}] Uploaded: {row['input'][:60]}")

    print(f"\n  Done – {len(rows)} items in dataset '{dataset_name}'.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run a full-pipeline Langfuse experiment.")
    parser.add_argument("--run-name",     default="pipeline-run",          help="Experiment run name in Langfuse UI")
    parser.add_argument("--dataset-name", default="sayfit-pipeline-eval",  help="Langfuse dataset name")
    parser.add_argument("--csv",          default=str(ROOT / "data" / "langfuse_dataset.csv"), help="Path to dataset CSV")
    parser.add_argument("--upload",       action="store_true", help="Upload/update dataset before running")
    parser.add_argument("--upload-only",  action="store_true", help="Upload dataset, then exit")
    parser.add_argument("--limit",        type=int, default=None, help="Only run on first N items")
    parser.add_argument("--openai",       action="store_true",   help="Use OpenAI instead of Groq")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Dataset CSV not found: {csv_path}")
        sys.exit(1)

    config.DEV_MODE = True  # suppresses spinner threads during batch run
    llm_client.configure(use_local=False, use_openai=args.openai)

    from langfuse import Langfuse  # noqa: PLC0415
    lf = Langfuse()

    if args.upload or args.upload_only:
        print(f"\n📤 Uploading dataset from {csv_path.name} …")
        upload_dataset(lf, args.dataset_name, csv_path)

    if args.upload_only:
        return

    dataset = lf.get_dataset(args.dataset_name)
    items   = dataset.items
    if args.limit:
        items = items[: args.limit]

    print(f"\n🧪 Starting experiment '{args.run_name}' on {len(items)} items …\n")

    result = lf.run_experiment(
        name=args.dataset_name,
        run_name=args.run_name,
        data=items,
        task=make_pipeline_task(),
        evaluators=[evaluator_suite],
        run_evaluators=[run_averages],
        max_concurrency=1,  # sequential – avoids FAISS / LLM rate-limit issues
    )

    print(result.format())
    if result.dataset_run_url:
        print(f"\n🔗 View in Langfuse: {result.dataset_run_url}\n")


if __name__ == "__main__":
    main()
