# Frontend API Guide — SayFit Alpha

> **Audience:** ML engineer building the Next.js frontend.
> This document is the authoritative contract between the FastAPI backend and the UI.
> It covers every endpoint, all TypeScript types, the voice input flow, error handling, and CORS.

---

## Quick Start

```bash
# Backend — Docker (recommended, matches production)
docker compose up --build      # first time: builds image, installs whisper + ffmpeg (~5 min)
docker compose up              # subsequent starts: fast

# Backend — local uvicorn (no Docker, faster iteration)
PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# Frontend (once scaffolded)
cd frontend && npm run dev    # → http://localhost:3000
```

> **First Docker build takes longer than usual** because `openai-whisper` pulls in PyTorch (~1.5 GB)
> and `ffmpeg` is installed as a system package. Subsequent builds are fast (layers are cached).

**Base URL:**
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
```

**CORS:** Already configured. The API allows `http://localhost:3000` out of the box — no browser preflight issues in dev.

---

## TypeScript Types

These map 1-to-1 with the Pydantic schemas in `api/schemas.py`.

```typescript
// ── Core food logging ────────────────────────────────────────────────────────

interface FoodItem {
  item_id: string
  item_name: string    // matched food name (e.g. "BANANA, RAW"), not raw user input
  calories: number
  protein: number      // grams
  fat: number          // grams
  carbs: number        // grams
  grams: number        // portion size
}

interface Meal {
  meal_id: string
  date_time: string    // ISO 8601 timestamp, server-generated
  uid: string
  items: FoodItem[]
}

interface DailyMacros {
  meal_date: string    // "YYYY-MM-DD"
  calories: number
  protein: number
  fat: number
  carbs: number
  meal_count: number
}

interface MealHistory {
  user_id: string
  days: number
  daily_breakdown: DailyMacros[]
  average_calories: number
  average_protein: number
}

// ── Voice transcription ──────────────────────────────────────────────────────

interface TranscribeResponse {
  text: string         // Whisper output, leading/trailing whitespace stripped
}

// ── Recipe suggestions ───────────────────────────────────────────────────────

interface RecipePreferences {
  target_calories?: number        // null = use full remaining daily budget
  taste?: "savory" | "sweet" | "any"   // default "any"
  max_time_minutes?: number       // null = no limit
  ingredients?: string[]          // on-hand ingredients to prefer
  few_ingredients?: boolean       // true = limit to max 5 ingredients
}

interface RecipeSuggestRequest {
  source: "spoonacular" | "kaggle" | "combo"   // default "spoonacular"
  preferences?: RecipePreferences
}

interface RecipeNutrition {
  calories: number
  protein: number
  fat: number
  carbs: number
}

interface RecipeSuggestion {
  title: string
  fit_score: number         // 0–100, higher = better macro match
  scale_factor: number      // informational only — DO NOT pass to log-suggestion
  ready_in_minutes: number
  source_url: string
  ingredients: string[]
  ingredient_count: number
  source: "spoonacular" | "kaggle"
  nutrition: RecipeNutrition   // already scaled — use directly for display
}

interface RecipeSuggestResponse {
  remaining: RecipeNutrition   // user's remaining macro budget for today
  suggestions: RecipeSuggestion[]
  message?: string             // present when suggestions is empty (explain why)
}

interface RecipeLogRequest {
  title: string          // recipe title, used as the meal name in history
  portions: number       // > 0, default 1.0
  nutrition: RecipeNutrition
}
```

---

## Endpoint Reference

Base URL: `http://localhost:8000`

### POST /log — Log a meal from text

Runs the full pipeline (extraction → FAISS retrieval → reranking → DB save).
Accepts plain text; returns the structured meal with item UUIDs.

> **This is the endpoint called after voice transcription.** The voice flow populates a text
> input and the user submits it here — see [Voice Input Integration](#voice-input-integration).

```typescript
// Request
POST /log
Content-Type: application/json
{ uid: string, text: string }

// Response 200
Meal

// Example
const res = await fetch(`${API_URL}/log`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ uid: "user123", text: "two eggs and oatmeal with banana" }),
})
const meal: Meal = await res.json()
```

**Note:** `POST /log` is synchronous — it blocks until the full pipeline finishes (~5–15s depending on LLM latency). If you want to show step-by-step progress to the user, see [SSE — Open Decision](#sse--open-decision) below.

---

### GET /meals/{uid}/today — Today's meals

Returns all meals logged today with full item detail.

```typescript
// Response 200
Meal[]

const meals: Meal[] = await fetch(`${API_URL}/meals/${uid}/today`).then(r => r.json())
```

---

### GET /meals/{uid}?days=N — Meal history

Daily macro summary for the last N days (default 30).

```typescript
// Response 200
MealHistory

const history: MealHistory = await fetch(`${API_URL}/meals/${uid}?days=7`).then(r => r.json())
```

---

### PATCH /meals/{uid}/items/{item_id} — Rescale a food item

Adjusts a food item's portion size. Macros are recalculated proportionally by the server.

```typescript
// Request body
{ meal_id: string, grams: number }   // grams > 0

// Response 204 No Content
// Follow up with GET /meals/{uid}/today to refresh state

await fetch(`${API_URL}/meals/${uid}/items/${item_id}`, {
  method: "PATCH",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ meal_id, grams: 200 }),
})
```

---

### POST /meals/{uid}/items — Add a missed item

Adds a food item to an existing meal. Nutrition is resolved server-side via FAISS.

```typescript
// Request body
{ meal_id: string, item_name: string, grams: number }

// Response 201
FoodItem

const item: FoodItem = await fetch(`${API_URL}/meals/${uid}/items`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ meal_id, item_name: "apple", grams: 150 }),
}).then(r => r.json())
```

---

### DELETE /meals/{uid}/items/{item_id} — Remove a food item

Soft-deletes an item. Meal totals are recalculated by the server.

```typescript
// meal_id is a query parameter — NOT in the body
// Response 204 No Content

await fetch(`${API_URL}/meals/${uid}/items/${item_id}?meal_id=${meal_id}`, {
  method: "DELETE",
})
```

---

### DELETE /meals/{uid}/{meal_id} — Delete a whole meal

Soft-deletes the meal and all its items.

```typescript
// Response 204 No Content

await fetch(`${API_URL}/meals/${uid}/${meal_id}`, { method: "DELETE" })
```

---

### POST /transcribe — Transcribe audio (voice input)

Accepts an audio file upload (WebM, Opus, WAV, MP3, etc.) and returns transcribed text.
Recording happens in the browser; this endpoint handles transcription only.

```typescript
// Request: multipart/form-data with field "file" (audio blob)
// Response 200
TranscribeResponse    // { text: string }

async function transcribeAudio(audioBlob: Blob): Promise<string> {
  const form = new FormData()
  form.append("file", audioBlob, "recording.webm")
  const res = await fetch(`${API_URL}/transcribe`, { method: "POST", body: form })
  if (!res.ok) throw new Error(`Transcription failed: ${res.status}`)
  const data: TranscribeResponse = await res.json()
  return data.text
}
```

**Important:** The first call to `/transcribe` after a server restart may take ~5 seconds
(Whisper `base` model loads into memory). Subsequent calls are fast (~1–3s for short clips).

---

### POST /recipes/{uid}/suggest — Get recipe suggestions

Returns recipe suggestions that fit the user's remaining macro budget for today.

```typescript
// Request
POST /recipes/{uid}/suggest
Content-Type: application/json
RecipeSuggestRequest

// Response 200
RecipeSuggestResponse

const res = await fetch(`${API_URL}/recipes/${uid}/suggest`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    source: "kaggle",              // "kaggle" = no API key needed; "spoonacular" needs SPOONACULAR_API_KEY
    preferences: { taste: "savory", max_time_minutes: 30 },
  }),
})
const data: RecipeSuggestResponse = await res.json()
```

**Note:** `source: "spoonacular"` requires a `SPOONACULAR_API_KEY` env var on the server. If missing, the API returns 400. `source: "kaggle"` works with no external key — use it as the default UI option.

---

### POST /recipes/{uid}/log-suggestion — Log a chosen recipe as a meal

Saves the selected recipe as a meal entry using the same DB tables as `/log`.

```typescript
// Request
POST /recipes/{uid}/log-suggestion
Content-Type: application/json
RecipeLogRequest

// Response 201
Meal

// Usage — pass the suggestion's nutrition directly, never pass scale_factor
const meal: Meal = await fetch(`${API_URL}/recipes/${uid}/log-suggestion`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    title: suggestion.title,
    portions: 1.0,
    nutrition: suggestion.nutrition,    // already scaled — use as-is
  }),
}).then(r => r.json())
```

---

## Voice Input Integration

This is the key new feature. The full flow:

```
1. User clicks mic button
        ↓
2. Browser: navigator.mediaDevices.getUserMedia({ audio: true })
        ↓
3. Browser: MediaRecorder records → WebM/Opus blob
        ↓
4. User clicks stop (or recording timer expires)
        ↓
5. POST /transcribe  (multipart, field "file" = the blob)
        ↓
6. Response: { "text": "two eggs and oatmeal with banana" }
        ↓
7. Populate the text input with the transcribed text
        ↓
8. User reviews / edits if needed, then confirms
        ↓
9. POST /log  (existing flow, unchanged)
```

### Minimal MediaRecorder implementation

```typescript
async function startVoiceCapture(
  onTranscribed: (text: string) => void,
  onError: (err: Error) => void,
): Promise<() => void> {           // returns a stop() function
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" })
  const chunks: BlobPart[] = []

  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data)
  }

  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop())   // release mic
    const blob = new Blob(chunks, { type: "audio/webm" })
    try {
      const text = await transcribeAudio(blob)
      onTranscribed(text)
    } catch (err) {
      onError(err as Error)
    }
  }

  recorder.start()
  return () => recorder.stop()    // call this on second button click or timeout
}
```

### Permission handling

```typescript
// getUserMedia throws DOMException if permission is denied
try {
  const stopRecording = await startVoiceCapture(setText, setError)
} catch (err) {
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    setError("Microphone permission denied. Please allow mic access in your browser.")
  }
}
```

### Browser compatibility

| Browser | Audio format | Notes |
|---------|-------------|-------|
| Chrome / Edge | WebM (Opus) | Default from MediaRecorder — works perfectly |
| Firefox | WebM (Opus) or OGG | Both work — no conversion needed |
| Safari ≥ 14.1 | MP4 / AAC | Pass `mimeType: "audio/mp4"` if supported; Whisper handles it |

Check support before using:
```typescript
const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/mp4"
```

---

## Error Handling

All errors follow FastAPI's standard error envelope:

```typescript
interface ApiError {
  detail: string | Array<{ loc: string[]; msg: string; type: string }>
}
```

| Status | Meaning | Common cause |
|--------|---------|-------------|
| 200 / 201 | Success | — |
| 204 | Success, no body | PATCH / DELETE responses |
| 400 | Bad request | Missing `SPOONACULAR_API_KEY` on server; pipeline failure |
| 404 | Not found | No nutrition data found for a food item (POST /meals/{uid}/items) |
| 422 | Validation error | Missing required field; wrong type; empty audio file; unsupported file type |
| 500 | Server error | Pipeline exception — check API logs |

**POST /log specific:** If the pipeline fails internally, you get a 500 with a detail string. Always show the user a friendly retry message rather than the raw error.

**POST /transcribe specific:**
- 422 if the `file` field is missing entirely
- 422 if the file is empty (zero bytes)
- 422 if the content-type is not audio (e.g. application/pdf)
- 500 if Whisper crashes internally (rare)

---

## SSE — Open Decision

The API currently uses **synchronous request-response** for `POST /log`.
The pipeline takes 5–15 seconds. There are two UI approaches:

**Option A — Loading spinner (current, simplest)**
Show a spinner while `POST /log` is pending. No API changes needed.

**Option B — SSE streaming (richer UX)**
Show each pipeline step as it completes ("Extracting items… Retrieving matches… Done").
Requires a new `POST /log/stream` endpoint returning `text/event-stream`. Not yet built.

Check with Kevin before implementing Option B — it requires backend work.

---

## Docker Port Map

| Service | Host port | Container port | Notes |
|---------|-----------|----------------|-------|
| `api` | 8000 | 8000 | FastAPI |
| `prometheus` | 9090 | 9090 | Metrics scraping |
| `grafana` | 3001 | 3000 | Dashboard (note: not 3000) |
| `frontend` | 3000 | 3000 | **Reserved for Next.js — not yet in compose** |

Grafana is on **3001** to leave 3000 free for the frontend. Don't change this.

When you add the `frontend` service to `docker-compose.yml`, use `NEXT_PUBLIC_API_URL=http://localhost:8000` for browser-side fetches and `API_URL=http://api:8000` for server-side Next.js fetches (internal Docker DNS).
