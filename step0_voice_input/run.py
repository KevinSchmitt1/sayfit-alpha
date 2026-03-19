"""
Step 0 – Standalone runner
===========================
Usage:
    python -m step0_voice_input.run                          # record from mic (10s default)
    python -m step0_voice_input.run --duration 15            # record for 15 seconds
    python -m step0_voice_input.run --wav path/to/file.wav   # transcribe existing .wav
    python -m step0_voice_input.run --uid my_user_id         # set user ID
    python -m step0_voice_input.run --output my_output.json  # custom output path
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from step0_voice_input.voice_recorder import (  # noqa: E402
    record_and_transcribe,
    transcribe_wav,
)

EXAMPLE_OUTPUT = Path(__file__).parent / "example_output.json"


def main():
    parser = argparse.ArgumentParser(description="Step 0 – Voice Input (Record & Transcribe)")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to an existing .wav file to transcribe (skips recording)")
    parser.add_argument("--duration", type=int, default=None,
                        help=f"Recording duration in seconds (default: {config.WHISPER_RECORD_SECONDS})")
    parser.add_argument("--uid", type=str, default="PLACEHOLDER_UserID",
                        help="User ID to embed in the output JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the output JSON")
    args = parser.parse_args()

    if args.wav:
        result = transcribe_wav(args.wav, uid=args.uid)
    else:
        duration = args.duration or config.WHISPER_RECORD_SECONDS
        result = record_and_transcribe(duration=duration, uid=args.uid)

    # save output
    output_path = Path(args.output) if args.output else config.OUTPUTS_DIR / "step0_voice_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
