"""
Step 1 – Standalone runner
===========================
Usage:
    python -m step1_extraction.run                        # uses example input
    python -m step1_extraction.run --input my_input.json  # custom input
    python -m step1_extraction.run --output results.json  # custom output path
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from step1_extraction.extractor import extract_items, extract_from_file  # noqa: E402

EXAMPLE_INPUT = Path(__file__).parent / "example_input.json"
EXAMPLE_OUTPUT = Path(__file__).parent / "example_output.json"


def main():
    parser = argparse.ArgumentParser(description="Step 1 – Food Item Extraction (LLM)")
    parser.add_argument("--input", type=str, default=None, help="Path to input JSON (voice recorder output)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else EXAMPLE_INPUT

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Step 1 – Food Item Extraction (LLM)")
    print("=" * 60)

    result = extract_from_file(input_path)

    # save output
    output_path = Path(args.output) if args.output else config.OUTPUTS_DIR / "step1_extraction_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Output saved to: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
