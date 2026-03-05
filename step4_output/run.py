"""
Step 4 – Standalone runner
===========================
Usage:
    python -m step4_output.run                          # uses example input
    python -m step4_output.run --input reranker.json    # custom input
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from step4_output.formatter import format_output, save_log  # noqa: E402

EXAMPLE_INPUT = Path(__file__).parent / "example_input.json"


def main():
    parser = argparse.ArgumentParser(description="Step 4 – Output Formatter")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to Step 3 reranker output JSON")
    parser.add_argument("--save", action="store_true", help="Save log to outputs/")
    args = parser.parse_args()

    print("=" * 60)
    print("  Step 4 – Output Formatter")
    print("=" * 60)

    input_path = Path(args.input) if args.input else EXAMPLE_INPUT
    if not input_path.exists():
        # try default pipeline output
        input_path = config.OUTPUTS_DIR / "step3_reranker_output.json"
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print(f"\n📂 Loading input: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    output = format_output(data)
    print(output)

    if args.save:
        save_log(data)


if __name__ == "__main__":
    main()
