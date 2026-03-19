"""
Step 1 – Food Item Extraction (LLM)
====================================
Takes raw transcribed text (JSON from voice recorder) and uses a Groq-hosted
LLM to extract structured food items with descriptions and search queries.

Can run standalone:
    python -m step1_extraction.run [--input path/to/input.json]

Input  : JSON  {"text": "...", "date_time": "...", "UID": "..."}
Output : JSON  {"items": {...}, "queries": [...]}
"""

import json
import re
import sys
from pathlib import Path

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import llm_client  # noqa: E402


def _log(*args, **kwargs):
    """Print only when developer mode is active."""
    if config.DEV_MODE:
        print(*args, **kwargs)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a food-logging assistant. The user will give you a sentence describing \
what they ate or drank. Your job:

1. Extract every distinct food/drink item mentioned.
2. For each item, return:
   - "item_name": a short, clear name (e.g. "pepperoni pizza")
   - "quantity_raw": the amount as spoken (e.g. "3", "half a", "a bowl of"), \
     or null if not mentioned
   - "quantity_parsed": numeric value of the quantity (e.g. "three" → 3, \
     "half a" → 0.5, "2 full" → 2, "a bowl of" → 1, "three and a half" → 3.5), \
     or null if quantity_raw is null. Encode fractions directly as decimals.
   - "unit_hint": the serving unit if explicitly mentioned (e.g. "slice", \
     "glass", "bowl", "cup", "piece", "plate", "handful", "can", "bottle", \
     "some", "a bit", "a little", "a few"), \
     or null if no specific unit or vague quantity was spoken.
   - "description": describe the processing degree or context \
     (e.g. "frozen", "homemade", "raw fruit", "fried", "boiled", "grilled", \
      "restaurant", "canned", "fresh"). If unknown, write "unspecified".
   - "category_ranks": a ranked list of 1–3 food database categories that \
     best describe this item. Use ONLY categories from the list below. \
     Order from most to least likely. Use ONE category for simple, \
     unambiguous items (e.g. "banana" → ["fruits"]). Use TWO or THREE for \
     composite or ambiguous dishes where the right database match is \
     uncertain (e.g. "spaghetti bolognese" → \
     ["prepared & frozen meals", "grains & pasta", "meat"]).
3. Produce a list of search queries optimised for a food-nutrition database \
   lookup. Each query should be the item name plus the description in \
   parentheses, e.g. "pepperoni pizza (frozen)".

Allowed categories (use these exact strings):
  dairy & eggs, meat, poultry, fish & seafood, vegetables, fruits,
  grains & pasta, baked goods, snacks, sweets & confectionery, beverages,
  prepared & frozen meals, condiments & sauces, fats & oils,
  legumes & beans, soups, plant-based alternatives, supplements

Return ONLY valid JSON matching this schema exactly:
{
  "items": {
    "item1": {
      "item_name": "<string>",
      "quantity_raw": "<string or null>",
      "quantity_parsed": <number or null>,
      "unit_hint": "<string or null>",
      "description": "<string>",
      "category_ranks": ["<category>"]
    }
  },
  "queries": ["<query1>", "<query2>"]
}

Rules:
- Number items sequentially: item1, item2, …
- Normalise to singular where appropriate (3 eggs → item_name "egg", quantity_raw "3").
- If the user mentions a combined dish (e.g. "chicken salad"), treat it as ONE item.
- "category_ranks" must contain only strings from the allowed categories list.
- Do NOT invent nutritional data.
- Output ONLY the JSON object, nothing else.
"""


_QUANTITY_WORDS = {
    "a", "an", "one", "two", "three", "four", "five", "six",
    "half", "some", "few", "couple", "handful",
}

_FILLER_PHRASES = re.compile(
    r"\b(i (had|ate|drank|think i had|think i ate)|had|not sure (how much|what was in it)|"
    r"maybe|i think|on the side|also|with|we (shared|went to|had)|"
    r"and we|finished with|a bunch of|some kind of|not totally sure)\b",
    re.IGNORECASE,
)

_SPLIT_PATTERN = re.compile(r",\s*(?:and\s+)?|(?<!\w)\band\b(?!\w)|\balso\b", re.IGNORECASE)


def _clean_segment(seg: str) -> str:
    seg = _FILLER_PHRASES.sub(" ", seg)
    seg = re.sub(r"\s{2,}", " ", seg).strip(" .,;")
    return seg


def _parse_quantity(seg: str) -> tuple[str | None, str]:
    """Trennt optionale Mengenangabe vom Lebensmittelnamen."""
    tokens = seg.split()
    if not tokens:
        return None, seg

    # "150g chicken" → quantity_raw="150g", name="chicken"
    if re.match(r"^\d+(?:\.\d+)?\s*(?:g|ml|gram|grams|kg)?$", tokens[0], re.IGNORECASE):
        return tokens[0], " ".join(tokens[1:]) or tokens[0]

    # "two eggs" / "a banana"
    if tokens[0].lower() in _QUANTITY_WORDS and len(tokens) > 1:
        return tokens[0], " ".join(tokens[1:])

    return None, seg


def extract_items_heuristic(text: str, date_time: str = "", uid: str = "") -> dict:
    """
    Regelbasierter Ersatz für den LLM-Extraktor (kein API-Aufruf nötig).
    Teilt den Text an Kommas / 'and' / 'also' auf und bereinigt Füllwörter.
    """
    _log("\n🔍 [Step 1] Extracting food items from text [heuristic] …")
    _log(f"   Input text: \"{text}\"")

    raw_segments = _SPLIT_PATTERN.split(text)
    items: dict = {}
    queries: list[str] = []

    idx = 1
    for seg in raw_segments:
        seg = _clean_segment(seg)
        if len(seg) < 2:
            continue

        quantity_raw, name = _parse_quantity(seg)
        name = name.strip().lower()
        if not name:
            continue

        key = f"item{idx}"
        items[key] = {
            "item_name":    name,
            "quantity_raw": quantity_raw,
            "description":  "unspecified",
            "date_time":    date_time,
            "uid":          uid,
        }
        queries.append(f"{name} (unspecified)")
        idx += 1

    _log(f"   ✅ Extracted {len(items)} item(s): {queries}")
    return {"items": items, "queries": queries}


def extract_items(text: str, date_time: str = "", uid: str = "", use_llm: bool = True) -> dict:
    """
    Call the extraction LLM and return structured items + queries.

    Parameters
    ----------
    text : str
        Raw transcription text, e.g. "i ate a pepperoni pizza and 3 eggs"
    date_time : str
        Timestamp from the transcription.
    uid : str
        User identifier.

    Returns
    -------
    dict with keys "items" and "queries", enriched with date_time per item.
    """
    if not use_llm:
        return extract_items_heuristic(text, date_time=date_time, uid=uid)

    _log("\n🔍 [Step 1] Extracting food items from text …")
    _log(f"   Input text: \"{text}\"")

    response = llm_client.get_client().chat.completions.create(
        model=llm_client.extraction_model(),
        temperature=config.EXTRACTION_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # parse JSON (handle markdown fences if present)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)

    # normalise: some models return items as a list instead of a keyed dict
    if isinstance(result.get("items"), list):
        result["items"] = {
            f"item{i+1}": item
            for i, item in enumerate(result["items"])
        }

    # attach metadata to each item
    for key in result.get("items", {}):
        result["items"][key]["date_time"] = date_time
        result["items"][key]["uid"] = uid

    n = len(result.get("items", {}))
    _log(f"   ✅ Extracted {n} item(s): {result.get('queries', [])}")
    return result


# ── convenience: load from file ─────────────────────────────────────────────
def extract_from_file(input_path: str | Path) -> dict:
    """Load a voice-recorder JSON and run extraction."""
    input_path = Path(input_path)
    print(f"\n📂 Loading input file: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    return extract_items(
        text=data["text"],
        date_time=data.get("date_time", ""),
        uid=data.get("UID", ""),
    )
