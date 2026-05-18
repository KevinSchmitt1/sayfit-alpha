════════════════════════════════════════════════
CHECKPOINT — 2026-05-18
════════════════════════════════════════════════

PROJECT: sayfit-alpha
TOPIC:   Unit test improvements, test structure reorganisation, FastAPI skeleton

────────────────────────────────────────────────
ORIGINAL GOAL
────────────────────────────────────────────────
Started by discussing whether to test "chicken eh breast i guess" in the
extractor. Evolved into a broader test quality session, then transitioned
into starting the FastAPI layer — the next planned milestone.

────────────────────────────────────────────────
WHAT ACTUALLY HAPPENED
────────────────────────────────────────────────
- Added `@pytest.mark.heuristic` / `@pytest.mark.llm_path` marks to test_extractor.py
- Added 5 new LLM-path tests (markdown fencing, malformed JSON, multi-item, voice noise, delegation)
- Created `pyproject.toml` to register pytest marks
- Moved all tests from `tests/` into `tests/unit/` — conftest moved too
- Updated CI to run `pytest tests/unit/ -q`
- Updated `CLAUDE_sw_mlops.md` — testing section fully rewritten to reflect new structure
- Created `tests/unit/test_ontology_filter.py` — 19 tests across heuristic and llm_path marks
- Discovered data quality bug: "banana" → "snacks" in `combined_final.csv`
- Documented bug in `bugs.md` with architectural root cause and fix proposals
- Created `api/schemas.py` — `MealCreate`, `FoodItem`, `Meal` (user built it Socratically)
- Created `api/main.py` — `POST /log` endpoint skeleton, calls `run_pipeline()` + `save_meal()`

────────────────────────────────────────────────
WHAT SHIFTED (vs original thinking)
────────────────────────────────────────────────
- **Ontology filter tests** — was: skip them, heuristic tests are enough / now: added them,
  and the process revealed the banana/snacks data quality bug
- **`seeds_only` fixture** — was: just a test isolation tool / now: also serves as ground
  truth for correct category mappings, exposes CSV vs seed disagreements
- **CSV as taxonomy source** — was: assumed CSV categories are correct / now: established
  that `_L1_SEEDS` + LLM system prompt are the source of truth; CSV must conform to them
- **`item_id` in API response** — was: assumed pipeline provides it / now: `save_meal()`
  generates `meal_id` but does not return individual `item_id`s — gap identified

────────────────────────────────────────────────
CURRENT STATE
────────────────────────────────────────────────
✅ Done      — `tests/unit/` structure with marks and conftest
✅ Done      — 171 unit tests passing
✅ Done      — CI updated to `tests/unit/`
✅ Done      — `pyproject.toml` with mark registration
✅ Done      — `test_ontology_filter.py` — 19 tests
✅ Done      — `bugs.md` — banana/snacks data quality bug documented
✅ Done      — `api/schemas.py` — MealCreate, FoodItem, Meal
✅ Done      — `api/main.py` — POST /log skeleton, compiles, not yet tested
⚠️  Uncertain — `item_id` in `FoodItem` response — currently `""`, save_meal() doesn't return item IDs
⚠️  Uncertain — `api/main.py` not yet run or tested — wiring may have issues
⏳ Planned   — remaining 5 endpoints (GET /meals, PATCH, POST item, DELETE)
⏳ Planned   — `tests/unit/test_api.py` — FastAPI endpoint tests
⏳ Planned   — `tests/integration/` layer — real API calls, real LLM
⏳ Planned   — Docker packaging
⏳ Planned   — fix banana/snacks CSV bug (data engineer task)

────────────────────────────────────────────────
OPEN QUESTIONS
────────────────────────────────────────────────
- `save_meal()` returns `meal_id` but not `item_id`s — do we need to query
  the DB after saving to get them, or change `save_meal()` to return them?
- `run_pipeline()` writes files to `outputs/` — is that acceptable in an API
  context, or should file I/O be suppressed for API requests?
- FastAPI not yet installed in `requirements.txt` — needs to be added before CI runs

────────────────────────────────────────────────
RECOMMENDED NEXT STEP
────────────────────────────────────────────────
Add `fastapi` and `uvicorn` to `requirements.txt`, then run the API locally
with `uvicorn api.main:app --reload` and call `POST /log` with a test input
to see if the wiring actually works end to end. Fix whatever breaks before
adding more endpoints.