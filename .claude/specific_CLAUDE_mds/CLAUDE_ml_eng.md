# SayFit Alpha – CLAUDE.md

This document covers the two actively developed areas of the project:
**ML Engineering** (existing pipeline components) and
**UI/UX** (Next.js 16 frontend, SSE streaming).

The overall pipeline architecture is documented in `README.md` and `structure.md`.

---

## Project Structure (Overview)

```
sayfit-alpha/
├── frontend/
│   └── src/
│       ├── app/                   # Next.js App Router pages
│       │   ├── page.tsx           # Main page (meal logger)
│       │   ├── history/page.tsx   # Meal history
│       │   └── profile/page.tsx   # User profile + macro goals
│       ├── components/            # React components
│       └── lib/api.ts             # API client + TypeScript types
├── step0_voice_input/             # Whisper ASR
├── step1_extraction/              # LLM extraction
├── step1_5_ontology_filter/       # SentenceTransformer + ontology
├── step2_retrieval/               # FAISS index + retrieval
├── step3_reranker/                # LLM reranker + calibration
├── step4_output/                  # Terminal formatter
├── step5_database/                # SQLite persistence
├── api.py                         # FastAPI backend (main API)
└── config.py                      # Central config file
```

---

## Pipeline – Existing ML Components

The pipeline runs through six steps. Several of them already use ML or model-based components today.

```
Voice / Text / JSON Input
         │
         ▼
┌─────────────────────────────┐
│ Step 0: Voice Input         │  Whisper ASR → {text, date_time, UID}
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 1: Extraction (LLM)    │  Groq / OpenAI / Ollama / heuristic fallback
│                             │  → structured food items with quantity + description
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 1.5: Ontology Filter   │  SentenceTransformer embeddings + category heuristics
│                             │  → L1/L2 category, portion hints
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 2: Retrieval (FAISS)   │  all-MiniLM-L6-v2 embeddings, top-20 candidates
│                             │  multi-query pooling, ontology boosts, name penalties
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 3: Reranker            │  LLM picks best match, estimates grams, calculates macros
│                             │  calibration layer applies stored user preferences
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 4: Output              │  Terminal table + totals, user review + corrections
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 5: Database            │  SQLite — meals, items, calibrations, user profiles
└─────────────────────────────┘
```

### What is already ML in the current pipeline

**Step 0 – Whisper ASR** (`step0_voice_input/voice_recorder.py`)
OpenAI Whisper (`base` by default, configurable via `WHISPER_MODEL` in `.env`) transcribes `.wav` audio to text. Audio is normalized to a target dB RMS before transcription.

**Step 1 – LLM Extraction** (`step1_extraction/extractor.py`)
An instruction-tuned LLM (Groq `llama-3.3-70b-versatile` by default) parses raw text into structured food items with `item_name`, `quantity_raw`, `unit_hint`, `description`, and `category_ranks`. Falls back to a rule-based heuristic in `--no-llm` mode.

**Step 1.5 – SentenceTransformer + Ontology** (`step1_5_ontology_filter/ontology_filter.py`)
Uses `all-MiniLM-L6-v2` embeddings to match food names against `data/food_ontology_300.json`. The ontology provides ranked L1/L2 categories and portion hints. In `--no-llm` mode this is the only source of category information, falling back to a keyword heuristic.

**Step 2 – FAISS Retrieval** (`step2_retrieval/retriever.py`)
Embeds food queries with `all-MiniLM-L6-v2` (batch size 512), searches a pre-built FAISS index over `data/combined_final.csv` (USDA + OpenFoodFacts), and returns the top-20 candidates. Retrieval quality is improved by:
- **Multi-query pooling** — 2–3 query variants per item, candidate sets merged (toggle via `MULTI_QUERY_POOLING`)
- **Ontology-based category boosts** — rank-1 category gets ×1.30, rank-2 ×1.10, rank-3 ×1.03 (configurable in `.env`)
- **Name-match penalties** — penalises candidates whose name is a weak lexical match for the query

**Step 3 – LLM Reranker + Calibration** (`step3_reranker/reranker.py`, `step3_reranker/calibration.py`)
An LLM picks the best candidate from the top-20, validates it against the extraction description (e.g. "raw fruit" cannot match "Peach Pie"), and estimates the final portion in grams. The calibration layer (`data/calibrations/user_prefs.json`) stores per-user corrections and applies them automatically on future runs — this is the learning engine of the system.

**Step 4 – Output Formatter** (`step4_output/formatter.py`)
Renders the final results as a terminal table with per-item macros and daily totals. The user can review, edit amounts, remove items, add missed foods, and confirm or discard before saving.

**Step 5 – Database** (`step5_database/database.py`)
Persists meals, meal items, calibrations, and user profiles to SQLite (`data/sayfit_meals.db`). Also exposes the data via the FastAPI backend (`api.py`) so the frontend can read meal history, daily totals, and the user profile.

---

## UI/UX (Frontend)

### Stack

- **Framework:** Next.js 16 (App Router) + React 19
- **Language:** TypeScript (strict)
- **Styling:** Tailwind CSS v4 + inline styles
- **API:** Fetch + SSE (Server-Sent Events) → FastAPI backend on `localhost:8000`

> **Warning:** Next.js 16 has breaking changes compared to older versions. Read `node_modules/next/dist/docs/` before making changes. The `AGENTS.md` in the frontend folder is the authoritative warning.

---

### Pages and Routing

| Route | File | Description |
|-------|------|-------------|
| `/` | `app/page.tsx` | Main page: meal logger, streaming, review, save |
| `/history` | `app/history/page.tsx` | Meal history with delete functionality |
| `/profile` | `app/profile/page.tsx` | User profile, macro goals, TDEE calculation |

---

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `TerminalInput` | `components/TerminalInput.tsx` | Text input for meal description |
| `PipelineStream` | `components/PipelineStream.tsx` | Displays SSE steps in real time |
| `NutritionTable` | `components/NutritionTable.tsx` | Results table with edit/remove/add |
| `DailySummary` | `components/DailySummary.tsx` | Daily overview (calories, macros) |
| `MealHistory` | `components/MealHistory.tsx` | Meal list for `/history` |
| `DevPanel` | `components/DevPanel.tsx` | Developer overlay with timing + SSE log |

---

### API Client (`lib/api.ts`)

Backend URL: `process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"`

**Key functions:**

```typescript
// SSE pipeline streaming
streamLog(text, uid, llm, onEvent, signal?, options?)

// Save a meal
saveMeal(results, uid, input_text) → Promise<string>  // meal_id

// Load daily overview
getToday(uid, date?) → Promise<TodayData>

// Search single food item (NutritionTable: "Add item")
searchFood(food_name, amount_grams, uid, llm) → Promise<NutritionItem | null>

// Read / save profile
getProfile(uid) / saveProfile(uid, profile)

// Delete
deleteMeal(meal_id) / deleteMealItem(meal_id, item_id)

// Update grams after the fact
updateItemGrams(meal_id, item_id, new_grams)
```

**SSE event types:**

```typescript
type SseEvent =
  | { event: "step";   step: number; total: number; msg: string }
  | { event: "detail"; key: string; data: unknown }
  | { event: "done";   results: NutritionItem[]; totals: Totals }
  | { event: "error";  msg: string };
```

---

### Key Data Types

```typescript
interface NutritionItem {
  item_name: string;
  matched_name: string;
  amount_grams: number;
  unit: string;
  confidence: "high" | "medium" | "low";
  confidence_note: string;
  nutrition: { calories: number; protein: number; fat: number; carbs: number };
  nutrition_per_100g: { calories: number; protein: number; fat: number; carbs: number };
  quantity_raw: string | null;
  date_time: string;
}
```

---

### Main Page State Machine

```
idle → streaming → done → (save → saved | discard → idle)
```

- **idle:** Input visible, menu visible
- **streaming:** Input disabled, PipelineStream shows steps
- **done:** NutritionTable visible
- **saved:** Second input appears (log another meal)

---

### Grams Recalculation (Pre-Save)

When the user edits `amount_grams` in `NutritionTable`, macros are recalculated:

```typescript
// Preferred: from nutrition_per_100g (more accurate)
item.nutrition = scale(nutrition_per_100g, newGrams / 100)

// Fallback: proportional to old value
item.nutrition = scale(item.nutrition, newGrams / item.amount_grams)
```

This pattern in `handleUpdateGrams` (page.tsx) should not be changed without good reason.

---

### Developer Mode

Toggle via the `[dev]` button (top right). When active:
- `DevPanel` shows all SSE events + API calls with timing
- `onDevEntry(label, type, data?)` is passed to `streamLog` and `saveMeal`

---

### Running the Frontend

```bash
cd frontend
npm install
npm run dev    # → http://localhost:3000
```

Backend must be running separately:
```bash
# from repo root
python api.py  # → http://localhost:8000
```

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the backend runs on a different port.

---

### Frontend Conventions

- Keep state centralized in `page.tsx` — components receive props, no global state manager.
- Use `"use client"` only when necessary (interactive components); prefer Server Components for static pages.
- TypeScript: no `any` types. Always type new API responses in `lib/api.ts`.
- Always guard SSE connections with an `AbortController` (already implemented via `abortRef`).
- New API endpoints always get a corresponding typed function in `lib/api.ts` before use.
