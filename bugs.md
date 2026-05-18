# Bugs

## [DATA QUALITY] Wrong category mappings in combined_final.csv

**Discovered:** 2026-05-18, surfaced by unit test isolation in `tests/unit/test_ontology_filter.py`

**Problem:**
`combined_final.csv` contains incorrect `cat_l1` mappings for some foods.
Example confirmed: `"banana"` → `"snacks"` instead of `"fruits"`.

This matters because `classify_item_name()` tries an exact CSV lookup first (step 1), before
falling back to the hard-coded `_L1_SEEDS`. A wrong CSV entry silently wins, sending a wrong
`category_hint` to the retriever. The retriever then applies its ontology boost to the wrong
category, potentially surfacing a worse food match — no error, no crash, just degraded results.

**Scope:** Unknown. Banana is confirmed. Other staple foods (apple, egg, rice, etc.) may also
be miscategorised. The full extent needs a data audit.

**Owner:** Data engineer (owns `combined_final.csv` and the data pipeline).

**Architectural root cause:**
The CSV is being used as the category taxonomy, but it should be the other way around.
The canonical category list already exists in two places in the code:
- `_L1_SEEDS` in `step1_5_ontology_filter/ontology_filter.py` (hand-curated)
- The allowed categories list in the LLM system prompt in `step1_extraction/extractor.py:64`

These are the source of truth. `combined_final.csv` must conform to them — it cannot define them.
When the CSV disagrees with the seeds, the CSV is wrong.

The FAISS index is built from the CSV, and the retriever boosts candidates whose category
matches the hint. If the CSV categories are wrong, the hint and the index speak the same
wrong language — the boost fires for the wrong category. Fixing the CSV fixes both.

**Proposed fix:**

Short term (unblock now, SW/MLOps):
- Add a small correction dict in `step1_5_ontology_filter/ontology_filter.py` that overrides
  known wrong CSV mappings before the lookup runs:
  ```python
  _CSV_CORRECTIONS = {"banana": ("fruits", "", ""), ...}
  ```
  Cheap, explicit, immediately testable.

Medium term (proper fix, data engineer):
- Audit `combined_final.csv` category distribution against `_L1_SEEDS`.
- Add `scripts/validate_categories.py`: flags every row where `cat_l1` disagrees with
  the canonical seed list. Output: list of foods to reclassify.
- Fix the CSV and rebuild the FAISS index.

Long term (data engineer + CI):
- CI runs the validation script on every CSV update so wrong categories can never
  reach the ontology filter silently again.

**How to verify the fix:** The `seeds_only` fixture in `tests/unit/test_ontology_filter.py`
proves the correct mapping (seeds are hand-curated). Once the CSV is fixed, remove the
`seeds_only` fixture and let the tests run against real CSV data — they should still pass.

