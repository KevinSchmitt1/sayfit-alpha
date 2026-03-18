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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import llm_client  # noqa: E402
from step3_reranker.calibration import get_user_preference  # noqa: E402


def _log(*args, **kwargs):
    """Print only when developer mode is active."""
    if config.DEV_MODE:
        print(*args, **kwargs)

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
1. An extracted food item (name, quantity).
2. A list of candidate matches from a nutrition database (each with name, \
   source, brand, and nutrition per 100 g).
3. Portion defaults and any user calibration data.

Your tasks:
A) SELECT the single best candidate that matches the extracted item. Use the \
   description to validate: e.g. "peach (raw fruit)" should NOT match \
   "PEACH PIE". Prefer generic/plain items when description is "unspecified".
B) DETERMINE the total amount in grams:
   - A pre-resolved "portion_hint" is provided with fields "grams", "unit", \
   and "source". It already accounts for the spoken quantity and user \
   preferences. Use "portion_hint.grams" as the final amount_grams.
   - EXCEPTION: if "quantity_raw" contains an explicit gram/ml weight \
   (e.g. "150g", "200ml"), override portion_hint and use that exact value.
   - Do not second-guess portion_hint unless you detect a clear error.
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

    # Portion hint pre-resolved by Step 1.5 (preferred); fall back to old lookup
    portion_hint = extracted_item.get("portion_hint")
    if not portion_hint:
        defaults = _load_portion_defaults()
        flat_hint = defaults.get(item_name.lower(), {})
        user_pref = get_user_preference(uid, item_name) if uid else None
        if user_pref or flat_hint:
            portion_hint = {
                "grams": user_pref["preferred_grams"] if user_pref else flat_hint.get("default_grams", 100),
                "unit": "g",
                "source": "user_pref" if user_pref else "portion_defaults",
            }

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
        "portion_hint": portion_hint,
    }, indent=2)

    response = llm_client.get_client().chat.completions.create(
        model=llm_client.reasoning_model(),
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

    # attach nutrition_per_100g from matched candidate so correction step can rescale
    matched_doc_id = result.get("matched_doc_id", "")
    result["nutrition_per_100g"] = {}
    for c in candidates:
        if c.get("doc_id") == matched_doc_id:
            result["nutrition_per_100g"] = c.get("nutrition_per_100g", {})
            break

    return result


def _parse_quantity(quantity_raw) -> float | None:
    """
    Konvertiert eine Rohmengenangabe in einen numerischen Multiplikator.
    Gibt None zurück wenn unklar (→ Portion-Default wird direkt verwendet).
    """
    if quantity_raw is None:
        return None
    q = str(quantity_raw).strip().lower()

    # Direkte Gramm-/ml-Angabe: "150g", "200 ml"
    import re
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:g|ml|gram|grams)$", q)
    if m:
        return float(m.group(1))  # wird als absolute Menge behandelt, kein Multiplikator

    word_map = {
        "a": 1, "an": 1, "one": 1, "half": 0.5, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "a few": 2, "some": 1,
    }
    if q in word_map:
        return word_map[q]

    try:
        return float(q)
    except ValueError:
        return None


def _is_absolute_grams(quantity_raw) -> bool:
    """True wenn quantity_raw bereits eine direkte Grammangabe ist (z.B. '150g')."""
    import re
    if quantity_raw is None:
        return False
    return bool(re.match(r"^\d+(?:\.\d+)?\s*(?:g|ml|gram|grams)$",
                          str(quantity_raw).strip().lower()))


def _description_penalty(description: str, candidate_name: str) -> float:
    """
    Einfache Kandidaten-Abwertung basierend auf der Zubereitungsart.
    Gibt einen Multiplikator 0.5–1.0 zurück.
    """
    desc = description.lower()
    cand = candidate_name.lower()

    # Rohes Lebensmittel sollte nicht auf verarbeitetes Produkt matchen
    if desc in ("raw fruit", "fresh", "raw") and any(
        w in cand for w in ("pie", "cake", "jam", "juice", "dried", "fried", "baked")
    ):
        return 0.5

    # Gegrilltes / gebratenes sollte nicht auf rohen Match fallen (weniger wichtig)
    return 1.0


def rerank_single_item_heuristic(
    extracted_item: dict,
    candidates: list[dict],
) -> dict:
    """
    Regelbasierter Ersatz für den LLM-Reranker (kein API-Aufruf nötig).

    Strategie:
      • Kandidat mit bestem adjusted_score × description_penalty
      • Portion aus quantity_raw (falls Zahlenangabe) oder portion_defaults
      • Nährwerte = per_100g × amount_grams / 100
    """
    item_name   = extracted_item.get("item_name", "")
    quantity_raw = extracted_item.get("quantity_raw")
    description  = extracted_item.get("description", "unspecified")

    defaults = _load_portion_defaults()

    # ── 1. Besten Kandidaten wählen ───────────────────────────────────────────
    best = max(
        candidates,
        key=lambda c: c.get("adjusted_score", 0) * _description_penalty(description, c.get("item_name", "")),
    )
    score = best.get("adjusted_score", 0)

    # ── 2. Portion bestimmen ──────────────────────────────────────────────────
    # Priority: explicit grams → portion_hint (from Step 1.5) → flat defaults
    portion_hint_resolved = extracted_item.get("portion_hint", {})
    quantity_parsed = extracted_item.get("quantity_parsed")

    if _is_absolute_grams(quantity_raw):
        import re
        amount_grams = float(re.match(r"^(\d+(?:\.\d+)?)", str(quantity_raw).strip()).group(1))
        unit = "g"
    elif portion_hint_resolved.get("grams"):
        amount_grams = float(portion_hint_resolved["grams"])
        unit = portion_hint_resolved.get("unit", "g")
    else:
        # Fallback: flat defaults + quantity multiplier
        flat_hint = defaults.get(item_name.lower(), {})
        default_grams = flat_hint.get("default_grams", 100)
        unit = flat_hint.get("unit", "g")
        multiplier = (float(quantity_parsed) if quantity_parsed is not None
                      else _parse_quantity(quantity_raw))
        amount_grams = default_grams * (multiplier if multiplier is not None else 1.0)

    # ── 3. Nährwerte berechnen ────────────────────────────────────────────────
    per100 = best.get("nutrition_per_100g", {})
    factor = amount_grams / 100.0
    nutrition = {
        "calories": round((per100.get("calories") or 0) * factor, 1),
        "protein":  round((per100.get("protein")  or 0) * factor, 1),
        "fat":      round((per100.get("fat")       or 0) * factor, 1),
        "carbs":    round((per100.get("carbs")     or 0) * factor, 1),
    }

    # ── 4. Konfidenz ──────────────────────────────────────────────────────────
    has_default = item_name.lower() in defaults
    if score >= 0.75 and has_default:
        confidence = "high"
        conf_note  = ""
    elif score >= 0.55:
        confidence = "medium"
        conf_note  = "Portion aus Standardwerten geschätzt." if not has_default else ""
    else:
        confidence = "low"
        conf_note  = f"Schwacher Retrieval-Score ({score:.3f}); Ergebnis möglicherweise ungenau."

    return {
        "item_name":              item_name,
        "matched_name":           best.get("item_name", ""),
        "matched_doc_id":         best.get("doc_id", ""),
        "source":                 best.get("source", ""),
        "brand":                  best.get("brand", ""),
        "amount_grams":           round(amount_grams),
        "unit":                   unit,
        "processing_description": description,
        "confidence":             confidence,
        "confidence_note":        conf_note,
        "nutrition":              nutrition,
        "nutrition_per_100g":     per100,
        "date_time":              extracted_item.get("date_time", ""),
    }


def rerank_all(
    extraction_output: dict,
    retrieval_output: dict,
    use_llm: bool = True,
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
    mode_label = "LLM" if use_llm else "heuristic (no LLM)"
    _log(f"\n🧠 [Step 3] Reranking & portion estimation [{mode_label}] …")

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
        _log(f"\n   [{i+1}/{len(items_dict)}] Processing: \"{item_name}\"")

        # find corresponding candidates
        query = queries[i] if i < len(queries) else ""
        candidates = query_to_candidates.get(query, [])

        if not candidates:
            _log(f"   ⚠️  No candidates found for \"{item_name}\" – skipping LLM, using unknown estimate.")
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
                "nutrition_per_100g": {},
                "date_time": item_data.get("date_time", ""),
            })
            continue

        if use_llm:
            result = rerank_single_item(item_data, candidates, uid=uid)
        else:
            result = rerank_single_item_heuristic(item_data, candidates)
        conf = result.get("confidence", "")
        _log(f"   → Matched: \"{result.get('matched_name', '')}\" | "
             f"{result.get('amount_grams', '?')}g | "
             f"confidence={conf}")
        if conf in ("low", "medium") and result.get("confidence_note"):
            _log(f"     ℹ️  {result['confidence_note']}")
        results.append(result)

    _log(f"\n   ✅ Reranking complete – {len(results)} items finalised.")
    return {"results": results}
