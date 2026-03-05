# SayFit Alpha 🍽️

**Voice-to-nutrition food logger** – tell the system what you ate and get instant macro tracking.

SayFit uses a RAG (Retrieval-Augmented Generation) pipeline to convert natural language food descriptions into accurate nutritional data by searching USDA and OpenFoodFacts databases.

---

## Quick Start

```bash
# 1. Clone the repo
git clone <repo-url> && cd sayfit-alpha

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
#    Create a .env file in the project root with:
echo 'GROK_API_KEY=your_groq_api_key_here' > .env

# 5. Extract the food databases (if not already done)
unzip food_dbs.zip -d data/

# 6. Build the FAISS index (one-time, takes a few minutes)
python main.py --build-index

# 7. Run the pipeline!
python main.py
```

---

## How It Works

```
Voice / Text Input
       │
       ▼
┌─────────────────────────────┐
│  Step 0: Voice Input         │   Record mic / load .wav → Whisper
│  "i ate 2 eggs and a pizza" │   → {text, date_time, UID}
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 1: Extraction (LLM)   │   Groq API → structured food items
│  "2 eggs and a pizza"       │   → {egg: qty=2, pizza: qty=1}
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 2: Retrieval (RAG)    │   FAISS vector search over 1.8M foods
│  Queries → Top-20 matches   │   USDA + OpenFoodFacts databases
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 3: Reranker (LLM)     │   Pick best match, estimate portions,
│  Validate + calculate       │   apply user calibrations
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 4: Output              │   Formatted table + daily totals
│  Review & correct            │   User correction → learning engine
└─────────────────────────────┘
```

---

## Project Structure

```
sayfit-alpha/
├── main.py                     # Full pipeline (interactive or file-based)
├── config.py                   # Central configuration (.env, paths, models)
├── portion_defaults.json       # Default portion sizes for common foods
├── requirements.txt
├── .env                        # API keys (not committed to git)
│
├── step0_voice_input/          # Voice recording & transcription
│   ├── voice_recorder.py       # Record, normalise, dB-adjust, transcribe
│   ├── run.py                  # Standalone runner
│   └── example_output.json     # Example transcription output
│
├── step1_extraction/           # LLM food item extraction
│   ├── extractor.py            # Core extraction logic
│   ├── run.py                  # Standalone runner
│   ├── example_input.json      # Example voice transcript
│   └── example_output.json     # Example extracted items
│
├── step2_retrieval/            # RAG retriever (FAISS + embeddings)
│   ├── build_index.py          # One-time index builder
│   ├── retriever.py            # Vector search + scoring
│   ├── run.py                  # Standalone runner
│   ├── example_input.json      # Example queries
│   └── example_output.json     # Example retrieved candidates
│
├── step3_reranker/             # Second-layer LLM (reranker + portions)
│   ├── reranker.py             # Reranking + portion estimation
│   ├── calibration.py          # User learning engine
│   ├── run.py                  # Standalone runner
│   ├── example_input.json      # Example combined input
│   └── example_output.json     # Example final items
│
├── step4_output/               # Output formatting
│   ├── formatter.py            # Table rendering + daily totals
│   ├── run.py                  # Standalone runner
│   └── example_input.json      # Example reranker output
│
├── data/
│   ├── off_nutrition_clean.csv # OpenFoodFacts database
│   ├── usda_nutrition_clean.csv# USDA database
│   ├── faiss_index/            # Built FAISS index (generated)
│   └── calibrations/           # User calibration data (generated)
│
├── inputs/                     # Place voice recorder JSONs here
└── outputs/                    # Pipeline outputs & logs
```

---

## Usage Modes

### Interactive Mode (default)
```bash
python main.py
```
Type what you ate and get instant results. The system will ask you to review and correct if needed.

### Direct Text Mode
```bash
python main.py --text "i had 2 eggs and a banana for breakfast"
```

### File Input Mode
```bash
python main.py --input inputs/my_meal.json
```
Where the input JSON looks like:
```json
{
  "text": "i ate a pepperoni pizza and 3 eggs",
  "date_time": "2026-03-05T12:30:00",
  "UID": "user_001"
}
```

### Voice Recording Mode
```bash
python main.py --record                    # record 10s from mic → full pipeline
python main.py --record --duration 15      # record 15 seconds
python main.py --wav path/to/audio.wav     # transcribe existing .wav → full pipeline
```

### Show Configuration
```bash
python main.py --show-config
```

---

## Running Individual Steps

Each step can run standalone for debugging or development:

```bash
# Step 0: Record from microphone and transcribe
python -m step0_voice_input.run                    # record from mic
python -m step0_voice_input.run --wav audio.wav     # transcribe .wav file

# Step 1: Extract food items from text
python -m step1_extraction.run --input step1_extraction/example_input.json

# Step 2: Build the FAISS index (one-time)
python -m step2_retrieval.run --build-index

# Step 2: Retrieve candidates for queries
python -m step2_retrieval.run --input step2_retrieval/example_input.json

# Step 3: Rerank candidates and estimate portions
python -m step3_reranker.run

# Step 4: Format and display results
python -m step4_output.run
```

Each step reads from example input files by default, so you can test them independently.

---

## Configuration

All settings are in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROK_API_KEY` | (required) | Your Groq API key |
| `EXTRACTION_MODEL` | `llama-3.3-70b-versatile` | Model for extraction tasks |
| `REASONING_MODEL` | `llama-3.3-70b-versatile` | Model for reranking/reasoning |
| `EXTRACTION_TEMPERATURE` | `0.1` | Temperature for extraction |
| `REASONING_TEMPERATURE` | `0.3` | Temperature for reasoning |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `EMBEDDING_BATCH_SIZE` | `512` | Batch size for encoding |
| `TOP_K_CANDIDATES` | `20` | Candidates per query |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `WHISPER_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `WHISPER_RECORD_SECONDS` | `10` | Default mic recording duration |
| `WHISPER_TARGET_DB` | `-20` | Target dB RMS for audio normalisation |

---

## User Calibration (Learning Engine)

The system learns from your corrections. If you say "a portion of noodles" and correct it to 500g, the system saves this preference in `data/calibrations/user_prefs.json`. Next time, it will use your preferred amount automatically.

Calibrations are stored per User ID and per food item.

---

## Databases

- **USDA**: ~1.8M food entries (generic foods)
- **OpenFoodFacts**: ~66K entries (branded/packaged foods)

Both databases provide nutrition per 100g (calories, protein, fat, carbs). The FAISS index combines both for unified search.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FAISS index not found` | Run `python main.py --build-index` |
| `GROK_API_KEY not set` | Add your key to `.env` |
| `CSV files not found` | Run `unzip food_dbs.zip -d data/` |
| Slow index building | Normal for ~1.8M entries (3-10 min). Only needed once. |
| Wrong food matched | Correct it interactively – the system will learn! |
| Microphone not found | Check `sounddevice` can see your mic: `python -c "import sounddevice; print(sounddevice.query_devices())"` |
| Whisper too slow | Use a smaller model: set `WHISPER_MODEL=tiny` in `.env` |