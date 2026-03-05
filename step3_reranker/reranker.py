"""
Step 3 – Second-Layer LLM  (Reranker + Portion Estimator)
==========================================================
Takes the extracted items (Step 1) and the retrieved candidates (Step 2),
uses an LLM to:
  1. Pick the best-matching candidate per item (validate against description).
  2. Estimate the amount in grams (using portion defaults + user calibrations).
  3. Calculate final macro values.

Standalone:  python -m step3_reranker.run [--input ...]
"""

import json
import sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from step3_reranker.calibration import get_user_preference  # noqa: E402

# ── Groq client ─────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=config.GROQ_API_KEY,
    base_url=config.GROQ_BASE_URL,
)

# ── Portion defaults ────────────────────────────────────────────────────────
_portion_defaults: dict | None = None


def _load_portion_defaults() -> dict:
    global _portion_defaults
    if _portion_defaults is not None:
        return _portion_defaults
    if config.PORTION_DEFAULTS_FILE.exists():
        with open(config.PORTION_DEFAULTS_FILE) as f:
            _portion_defaults = json.load(f)
    else:
        _portion_defaults = {}
    return _portion_defaults


# ── System prompt for the reranker LLM ─────────────────────────────────────
SYSTEM_PROMPT = """\
You are a nutrition-matching assistant. You will receive:
1. An extracted food item (name, quantity, description/processing degree).
2. A list of candidate matches from a nutrition database (each with name, \
   source, brand, and nutrition per 100 g).
3. Portion defaults and any user calibration data.

Your tasks:
A) SELECT the single best candidate that matches the extracted item. Use the \
   description to validate: e.g. "peach (raw fruit)" should NOT match \
   "PEACH PIE". Prefer generic/plain items when description is "unspecified".
B) ESTIMATE the total amount in grams:
   - Use the spoken quantity if given (e.g. "3" eggs × 60 g each = 180 g).
   - Use portion defaults / user calibration if no quantity is spoken.
   - For fractional amounts like "half a pizza", compute accordingly.
C) CALCULATE the final macros by scaling the per-100g values to the \
   estimated grams.

Return ONLY valid JSON matching this schema:
{
  "item_name": "<original extracted name>",
  "matched_name": "<chosen candidate name>",
  "matched_doc_id": "<doc_id of chosen candidate>",
  "source": "<usda or openfoodfacts>",
  "brand": "<brand or empty>",
  "amount_grams": <number>,
  "unit": "g",
  "processing_description": "<description from extraction>",
  "confidence": "<high|medium|low>",
  "confidence_note": "<brief explanation if medium/low>",
  "nutrition": {
    "calories": <number>,
    "protein": <number>,
    "fat": <number>,
    "carbs": <number>
  }
}

Rules:
- All nutrition values are the TOTAL for the estimated grams (not per 100g).
- Round nutrition to 1 decimal place.
- If no candidate is a good match, set confidence to "low" and pick the \
  closest anyway, explaining in confidence_note.
- Output ONLY the JSON, nothing else.
"""


def rerank_single_item(
    extracted_item: dict,
    candidates: list[dict],
    uid: str = "",
) -> dict:
    """
    Use the LLM to pick the best candidate and estimate portion for one item.

    Parameters
    ----------
    extracted_item : dict
        From Step 1, e.g. {"item_name": "egg", "quantity_raw": "3", "description": "boiled"}
    candidates : list[dict]
        From Step 2, each with name, nutrition_per_100g, etc.
    uid : str
        User ID for calibration lookup.

    Returns
    -------
    dict with matched item, amount, and computed nutrition.
    """
    item_name = extracted_item.get("item_name", "")
    quantity_raw = extracted_item.get("quantity_raw")
    description = extracted_item.get("description", "unspecified")

    # look up portion defaults + user calibrations
    defaults = _load_portion_defaults()
    portion_hint = defaults.get(item_name.lower(), {})
    user_pref = get_user_preference(uid, item_name) if uid else None

    # build the user message for the LLM
    user_msg = json.dumps({
        "extracted_item": {
            "item_name": item_name,
            "quantity_raw": quantity_raw,
            "description": description,
        },
        "candidates": [
            {
                "doc_id": c["doc_id"],
                "source": c["source"],
                "item_name": c["item_name"],
                "brand": c.get("brand", ""),
                "adjusted_score": c.get("adjusted_score", 0),
                "nutrition_per_100g": c["nutrition_per_100g"],
                "portion_info": c.get("portion_info", {}),
            }
            for c in candidates[:20]  # max 20 candidates
        ],
        "portion_defaults": portion_hint if portion_hint else None,
        "user_calibration": user_pref,
    }, indent=2)

    response = client.chat.completions.create(
        model=config.REASONING_MODEL,
        temperature=config.REASONING_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)

    # attach date_time from extraction
    result["date_time"] = extracted_item.get("date_time", "")

    return result


def rerank_all(
    extraction_output: dict,
    retrieval_output: dict,
) -> dict:
    """
    Run the reranker for every extracted item.

    Parameters
    ----------
    extraction_output : dict
        Full output from Step 1 (has "items" and "queries").
    retrieval_output : dict
        Full output from Step 2 (has "items" list with candidates per query).

    Returns
    -------
    dict with "results" list of finalised food items.
    """
    print("\n🧠 [Step 3] Reranking & estimating portions …")

    items_dict = extraction_output.get("items", {})
    retrieval_items = retrieval_output.get("items", [])

    # map query → candidates for lookup
    query_to_candidates = {}
    for ri in retrieval_items:
        query_to_candidates[ri["query"]] = ri["candidates"]

    queries = extraction_output.get("queries", [])
    uid = ""
    # get uid from first item
    for v in items_dict.values():
        uid = v.get("uid", "")
        break

    results = []
    for i, (item_key, item_data) in enumerate(items_dict.items()):
        item_name = item_data.get("item_name", "")
        print(f"\n   [{i+1}/{len(items_dict)}] Processing: \"{item_name}\"")

        # find corresponding candidates
        query = queries[i] if i < len(queries) else ""
        candidates = query_to_candidates.get(query, [])

        if not candidates:
            print(f"   ⚠️  No candidates found for \"{item_name}\" – skipping LLM, using unknown estimate.")
            results.append({
                "item_name": item_name,
                "matched_name": "UNKNOWN",
                "matched_doc_id": "",
                "source": "",
                "brand": "",
                "amount_grams": 100,
                "unit": "g",
                "processing_description": item_data.get("description", "unspecified"),
                "confidence": "low",
                "confidence_note": "No database candidates found. Macros are estimated.",
                "nutrition": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0},
                "date_time": item_data.get("date_time", ""),
            })
            continue

        result = rerank_single_item(item_data, candidates, uid=uid)
        conf = result.get("confidence", "")
        print(f"   → Matched: \"{result.get('matched_name', '')}\" | "
              f"{result.get('amount_grams', '?')}g | "
              f"confidence={conf}")
        if conf in ("low", "medium") and result.get("confidence_note"):
            print(f"     ℹ️  {result['confidence_note']}")
        results.append(result)

    print(f"\n   ✅ Reranking complete – {len(results)} items finalised.")
    return {"results": results}
