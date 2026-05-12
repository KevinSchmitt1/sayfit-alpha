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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not GROQ_API_KEY and not OPENAI_API_KEY:
    print("[WARNING] Neither GROQ_API_KEY nor OPENAI_API_KEY found in .env – LLM calls will fail.")

# ── LLM settings (Groq) ─────────────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Model for structured extraction tasks (JSON output)
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "llama-3.3-70b-versatile")
# Model for reasoning / reranking / coaching
REASONING_MODEL = os.getenv("REASONING_MODEL", "llama-3.3-70b-versatile")

# ── LLM settings (OpenAI) ────────────────────────────────────────────────────
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_EXTRACTION_MODEL = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")
OPENAI_REASONING_MODEL  = os.getenv("OPENAI_REASONING_MODEL",  "gpt-4o-mini")
# Temperature for extraction (low = deterministic)
EXTRACTION_TEMPERATURE = float(os.getenv("EXTRACTION_TEMPERATURE", "0.1"))
# Temperature for reasoning (slightly higher for natural language)
REASONING_TEMPERATURE = float(os.getenv("REASONING_TEMPERATURE", "0.3"))

# ── Embedding settings ───────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "512"))

# ── Retrieval settings ───────────────────────────────────────────────────────
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))
# Multi-query pooling: run 2–3 query variants per item and merge the candidate sets.
# Increases recall without touching the reranker. Set to "false" to disable.
MULTI_QUERY_POOLING = os.getenv("MULTI_QUERY_POOLING", "true").lower() == "true"

# ── Data files ───────────────────────────────────────────────────────────────
# usda_final.csv: rows with item_name, macros, cat_l1, cat_l2 (cat_l3 optional/removed)
USDA_FINAL_CSV = DATA_DIR / "usda_final.csv"
COMBINED_FINAL_CSV = DATA_DIR / "combined_final.csv" 

# ── Ontology filter (Step 1.5) ───────────────────────────────────────────────
# Legacy single-boost (kept for heuristic fallback path).
ONTOLOGY_CATEGORY_BOOST = float(os.getenv("ONTOLOGY_CATEGORY_BOOST", "1.15"))

# Tiered boost multipliers for ranked LLM category hints.
# Rank 1 (most likely category) → strongest boost.
# Rank 2 / 3 → progressively smaller boosts so they don't compete with rank 1.
ONTOLOGY_BOOST_RANK1 = float(os.getenv("ONTOLOGY_BOOST_RANK1", "1.30"))
ONTOLOGY_BOOST_RANK2 = float(os.getenv("ONTOLOGY_BOOST_RANK2", "1.10"))
ONTOLOGY_BOOST_RANK3 = float(os.getenv("ONTOLOGY_BOOST_RANK3", "1.03"))

# ── Whisper / Voice input settings ───────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
WHISPER_RECORD_SECONDS = int(os.getenv("WHISPER_RECORD_SECONDS", "15"))
WHISPER_TARGET_DB = float(os.getenv("WHISPER_TARGET_DB", "-20"))

# ── Developer mode ──────────────────────────────────────────────────────────
# When True, all step modules print verbose debug output.
# Set via `python main.py --devmode`, or by the SAYFIT_DEVMODE=1 env var
# (used when Step 0 runs as an isolated subprocess).
DEV_MODE: bool = os.getenv("SAYFIT_DEVMODE", "0") == "1"

# ── Default user ─────────────────────────────────────────────────────────────
# Used as the pre-filled suggestion when the program prompts for a user ID.
# Override via SAYFIT_DEFAULT_USER env var or the --default-uid CLI flag.
DEFAULT_USER_ID: str = os.getenv("SAYFIT_DEFAULT_USER", "default_user")

# ── Portion / ontology data files ──────────────────────────────────────────
PORTION_DEFAULTS_FILE = DATA_DIR / "portion_defaults.json"
FOOD_ONTOLOGY_FILE    = DATA_DIR / "food_ontology_300.json"

# ── Calibration file ────────────────────────────────────────────────────────
CALIBRATION_FILE = CALIBRATION_DIR / "user_prefs.json"

# ── Local LLM (Ollama) ───────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH = DATA_DIR / "sayfit_meals.db"


def print_config(active_extraction_model: str = "", active_reasoning_model: str = ""):
    """Print current configuration for debugging."""
    ext_model = active_extraction_model or EXTRACTION_MODEL
    rea_model = active_reasoning_model or REASONING_MODEL
    print("=" * 60)
    print("  SayFit Alpha – Configuration")
    print("=" * 60)
    print(f"  ROOT_DIR           : {ROOT_DIR}")
    print(f"  GROQ API KEY       : {'***' + GROQ_API_KEY[-6:] if len(GROQ_API_KEY) > 6 else '(not set)'}")
    print(f"  EXTRACTION_MODEL   : {ext_model}")
    print(f"  REASONING_MODEL    : {rea_model}")
    print(f"  EMBEDDING_MODEL    : {EMBEDDING_MODEL_NAME}")
    print(f"  TOP_K_CANDIDATES   : {TOP_K_CANDIDATES}")
    print(f"  WHISPER_MODEL      : {WHISPER_MODEL}")
    print(f"  WHISPER_SAMPLE_RATE: {WHISPER_SAMPLE_RATE} Hz")
    print(f"  WHISPER_TARGET_DB  : {WHISPER_TARGET_DB} dB")
    print(f"  USDA final CSV     : {USDA_FINAL_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
