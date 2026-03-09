"""
SayFit Alpha – Central Configuration
=====================================
Loads environment variables and defines paths / model settings used by all modules.
Everything is configurable via .env or by editing the defaults below.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
CALIBRATION_DIR = DATA_DIR / "calibrations"
OUTPUTS_DIR = ROOT_DIR / "outputs"
INPUTS_DIR = ROOT_DIR / "inputs"

# ensure dirs exist
for d in [DATA_DIR, INDEX_DIR, CALIBRATION_DIR, OUTPUTS_DIR, INPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv(ROOT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    print("[WARNING] GROK_API_KEY not found in .env – LLM calls will fail.")

# ── LLM settings (Groq) ─────────────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Model for structured extraction tasks (JSON output)
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "llama-3.3-70b-versatile")
# Model for reasoning / reranking / coaching
REASONING_MODEL = os.getenv("REASONING_MODEL", "llama-3.3-70b-versatile")
# Temperature for extraction (low = deterministic)
EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.1"))
# Temperature for reasoning (slightly higher for natural language)
REASONING_TEMPERATURE = float(os.getenv("REASONING_TEMPERATURE", "0.3"))

# ── Embedding settings ───────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))

# ── Retrieval settings ───────────────────────────────────────────────────────
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))

# ── Data files ───────────────────────────────────────────────────────────────
OFF_CSV = DATA_DIR / "off_nutrition_clean.csv"
USDA_CSV = DATA_DIR / "usda_nutrition_clean.csv"

# ── Whisper / Voice input settings ───────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
WHISPER_RECORD_SECONDS = int(os.getenv("WHISPER_RECORD_SECONDS", "10"))
WHISPER_TARGET_DB = float(os.getenv("WHISPER_TARGET_DB", "-20"))

# ── Portion defaults file ───────────────────────────────────────────────────
PORTION_DEFAULTS_FILE = ROOT_DIR / "portion_defaults.json"

# ── Calibration file ────────────────────────────────────────────────────────
CALIBRATION_FILE = CALIBRATION_DIR / "user_prefs.json"


def print_config():
    """Print current configuration for debugging."""
    print("=" * 60)
    print("  SayFit Alpha – Configuration")
    print("=" * 60)
    print(f"  ROOT_DIR           : {ROOT_DIR}")
    print(f"  GROQ API KEY       : {'***' + GROQ_API_KEY[-6:] if len(GROQ_API_KEY) > 6 else '(not set)'}")
    print(f"  EXTRACTION_MODEL   : {EXTRACTION_MODEL}")
    print(f"  REASONING_MODEL    : {REASONING_MODEL}")
    print(f"  EMBEDDING_MODEL    : {EMBEDDING_MODEL_NAME}")
    print(f"  TOP_K_CANDIDATES   : {TOP_K_CANDIDATES}")
    print(f"  WHISPER_MODEL      : {WHISPER_MODEL}")
    print(f"  WHISPER_SAMPLE_RATE: {WHISPER_SAMPLE_RATE} Hz")
    print(f"  WHISPER_TARGET_DB  : {WHISPER_TARGET_DB} dB")
    print(f"  USDA CSV rows      : {USDA_CSV}")
    print(f"  OFF CSV rows       : {OFF_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
