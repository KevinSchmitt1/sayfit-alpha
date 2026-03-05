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
import sys
from pathlib import Path
from openai import OpenAI

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── Groq client ─────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=config.GROQ_API_KEY,
    base_url=config.GROQ_BASE_URL,
)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a food-logging assistant. The user will give you a sentence describing \
what they ate or drank. Your job:

1. Extract every distinct food/drink item mentioned.
2. For each item, return:
   - "item_name": a short, clear name (e.g. "pepperoni pizza")
   - "quantity_raw": the amount as spoken (e.g. "3", "half a", "a bowl of"), \
     or null if not mentioned
   - "description": describe the processing degree or context \
     (e.g. "frozen", "homemade", "raw fruit", "fried", "boiled", "grilled", \
      "restaurant", "canned", "fresh"). If unknown, write "unspecified".
3. Produce a list of search queries optimised for a food-nutrition database \
   lookup. Each query should be the item name plus the description in \
   parentheses, e.g. "pepperoni pizza (frozen)".

Return ONLY valid JSON matching this schema exactly:
{
  "items": {
    "item1": {
      "item_name": "<string>",
      "quantity_raw": "<string or null>",
      "description": "<string>"
    }
  },
  "queries": ["<query1>", "<query2>"]
}

Rules:
- Number items sequentially: item1, item2, …
- Normalise to singular where appropriate (3 eggs → item_name "egg", quantity_raw "3").
- If the user mentions a combined dish (e.g. "chicken salad"), treat it as ONE item.
- Do NOT invent nutritional data.
- Output ONLY the JSON object, nothing else.
"""


def extract_items(text: str, date_time: str = "", uid: str = "") -> dict:
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
    print("\n🔍 [Step 1] Extracting food items from text …")
    print(f"   Input text: \"{text}\"")

    response = client.chat.completions.create(
        model=config.EXTRACTION_MODEL,
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

    # attach metadata to each item
    for key in result.get("items", {}):
        result["items"][key]["date_time"] = date_time
        result["items"][key]["uid"] = uid

    n = len(result.get("items", {}))
    print(f"   ✅ Extracted {n} item(s): {result.get('queries', [])}")
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
