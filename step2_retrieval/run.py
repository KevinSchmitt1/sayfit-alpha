"""
Step 2 – Standalone runner
===========================
Usage:
    python -m step2_retrieval.run                         # uses example input
    python -m step2_retrieval.run --input queries.json    # custom input
    python -m step2_retrieval.run --build-index           # build FAISS index first
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

EXAMPLE_INPUT = Path(__file__).parent / "example_input.json"
EXAMPLE_OUTPUT = Path(__file__).parent / "example_output.json"


def main():
    parser = argparse.ArgumentParser(description="Step 2 – Food Retrieval (RAG)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input JSON (step1 output or query list)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON")
    parser.add_argument("--build-index", action="store_true",
                        help="Build the FAISS index before retrieving")
    parser.add_argument("--top-k", type=int, default=None, help="Number of candidates per query")
    args = parser.parse_args()

    print("=" * 60)
    print("  Step 2 – Food Retrieval (RAG)")
    print("=" * 60)

    # optionally build index first
    if args.build_index:
        from step2_retrieval.build_index import build_index
        build_index()

    from step2_retrieval.retriever import retrieve

    input_path = Path(args.input) if args.input else EXAMPLE_INPUT
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n📂 Loading input: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    # accept both a plain list of queries or a step1-style output
    if isinstance(data, list):
        queries = data
    elif "queries" in data:
        queries = data["queries"]
    else:
        print("❌ Input must contain a 'queries' list or be a JSON array.")
        sys.exit(1)

    result = retrieve(queries, top_k=args.top_k)

    output_path = Path(args.output) if args.output else config.OUTPUTS_DIR / "step2_retrieval_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Output saved to: {output_path}")

    # print summary
    for item in result["items"]:
        print(f"\n  Query: {item['query']}")
        for i, c in enumerate(item["candidates"][:3]):
            print(f"    #{i+1} {c['item_name']} ({c['source']}) — score={c['adjusted_score']:.3f}")
        if len(item["candidates"]) > 3:
            print(f"    … and {len(item['candidates'])-3} more candidates")


if __name__ == "__main__":
    main()
