"""
Step 3 – Standalone runner
===========================
Usage:
    python -m step3_reranker.run                            # uses example inputs
    python -m step3_reranker.run --extraction e.json --retrieval r.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from step3_reranker.reranker import rerank_all  # noqa: E402

EXAMPLE_INPUT = Path(__file__).parent / "example_input.json"
EXAMPLE_OUTPUT = Path(__file__).parent / "example_output.json"


def main():
    parser = argparse.ArgumentParser(description="Step 3 – Reranker & Portion Estimator (LLM)")
    parser.add_argument("--extraction", type=str, default=None,
                        help="Path to Step 1 extraction output JSON")
    parser.add_argument("--retrieval", type=str, default=None,
                        help="Path to Step 2 retrieval output JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("  Step 3 – Reranker & Portion Estimator (LLM)")
    print("=" * 60)

    # load inputs
    if args.extraction and args.retrieval:
        ext_path = Path(args.extraction)
        ret_path = Path(args.retrieval)
    elif EXAMPLE_INPUT.exists():
        # example_input.json contains both extraction + retrieval merged
        print(f"\n📂 Loading combined example input: {EXAMPLE_INPUT}")
        with open(EXAMPLE_INPUT) as f:
            combined = json.load(f)
        extraction_data = combined["extraction"]
        retrieval_data = combined["retrieval"]

        result = rerank_all(extraction_data, retrieval_data)

        output_path = Path(args.output) if args.output else config.OUTPUTS_DIR / "step3_reranker_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Output saved to: {output_path}")
        print(json.dumps(result, indent=2))
        return
    else:
        # try loading from default output locations
        ext_path = config.OUTPUTS_DIR / "step1_extraction_output.json"
        ret_path = config.OUTPUTS_DIR / "step2_retrieval_output.json"

    for p, label in [(ext_path, "extraction"), (ret_path, "retrieval")]:
        if not p.exists():
            print(f"❌ {label} file not found: {p}")
            sys.exit(1)

    print(f"\n📂 Loading extraction output: {ext_path}")
    with open(ext_path) as f:
        extraction_data = json.load(f)

    print(f"📂 Loading retrieval output: {ret_path}")
    with open(ret_path) as f:
        retrieval_data = json.load(f)

    result = rerank_all(extraction_data, retrieval_data)

    output_path = Path(args.output) if args.output else config.OUTPUTS_DIR / "step3_reranker_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Output saved to: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
