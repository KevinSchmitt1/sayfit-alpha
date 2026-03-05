
# General description of the program

Build a readme.md and requirements.txt. 
Build the program, so that every user can download the repo and use it witht he readme.md.

Dont let the program be a blackbox, add describing print-lines (so that the user sees whats going on).

__IMPORTANT:__
Build different modules for the steps, so that they can act as standalones. Make it work such as the standalone modules can work with the outputs of the prior modules. Integrate example outputs for each step, so we can adjust the steps by ourselves.
Avoid hard coding and make use of the smart LLM's (.env: GROK_API_KEY use deepseek).
Make it as adjustable and useable for human devs as possible.
You dont need to do UI building, its fine if it runs in terminal.

Would be nice if it would be somehow not too cryptic to use the whole thing.
# Architecture Overview


## 1. Voice Input 

ASR output JSON:

voice_recorder.py: used for transcribing and parsing .wav to .json

- includes a normalizer and a db-adjuster

```json
{"text": "i ate a pepperoni pizza and 3 eggs",
 "date_time": "timestamp",
 "UID": "PLACEHOLDER_UserID" }
```

## 2. Item Extraction (LLM, not RAG yet)
Input: json from step 1.

Goal: turn raw text (json) into a structured list (json) and a query to give to retriever to search for data out of the RAG (use USDA and openfoodfacts database).

json-output: keep userID for later data management
```json 
{"item1": 
  {"item-name": "peperoni pizza",
  "date_time": "timestamp",
  "description": "processing degree (f.e. raw fruit, fried, salad, pie, frozen, restaurant, homemade, etc.)"},
"item2":
  {"item-name": "egg",
  "date-time": "timestamp",
  "description": "processing degree (raw, boiled, fried, large, small, etc.)"}
}
```
You can adjust the output to a matching format, but keep the features as we want them.

query output:
```
["peperoni pizza (frozen)", "egg (baked)"]
```

Save json for later use and give query to retriever (RAG).


Use an instruction LLM (groq) with a strict JSON schema.

## 3. Retrieval (your RAG)

For each extracted item, retrieve candidate foods from your databases (pick 20 best candidates):

- USDA (generic foods)
- OpenFoodFacts (branded/packaged foods)

The databases are in the food_dbs.zip and will be extracted to 'data/off_nutrition_clean.csv', 'data/usda_nutrition_clean.csv' -> use these first for the RAG.

Use embeddings + vector search to get top K candidates (20). Build batching and vector score system. Add penalties for foods with non exact matches:

F.e.: query input "avocado" has best similarity with hit "avocado" not "AVOCADO OIL, AVOCADO"

Retriever output only gives macros and kcal per 100g, second layer llm handles to portion sizes and predictions of the amounts/units.

json
```
{
  "items": [
    {
      "name": "peperoni pizza",
      "amount": 100.0,
      "unit": "g",
      "matched_name": "Peperoni Pizza",
      "nutrition": {
        "calories": 389.0,
        "protein": 16.9,
        "fat": 10.6,
        "carbs": 40.4
      }
    },
    {
      "name": "eggs",
      "amount": 100.0,
      "unit": "g",
      "matched_name": "Egg",
      "nutrition": {
        "calories": 300,
        "protein": 12.4,
        "fat": 9.8,
        "carbs": 0.4
      }
    }
  ] 
}
```

## 4. Second layer LLM (extractor and reasoning)

Look at 20 best candidates from retriever, compare json from first extraction step with json from retriever. Use description for validation of the picked foods:
f.e. input peach (description: raw fruit) cannot match "PEACH PIE".
Built in some sanity checks/verification steps if the foods are accurate enough for the user.

Make assumptions/predictions on the mentioned amounts in the json-text inputs from transcription:
f.e. "half a pizza" --> assume weight in "g" by searching for default pizza weight

API key to use from .env

Output-json
```
{Data Output :
"item-name": output of first layer llm
"matched-name": output of retriever item
kcal: calculated by amount
protein: calculated by amount
fat: calculated by amount
carbs: calculated by amount
"amount": (in g calculated with predicted amount)
"unit": 
processing degree: (description of the first layer llm)
Date: timestamp
}
```

Give a user output with the amounts of the extracted foods and the corresponding macros/kcal.
Ask the user if this is correct or to specify the input if something is inaccurate.
Build a learn engine for the llm (database of calibrations by the user) with the past user input:
f.e. learn if the user always tells "a portion of noodles" and corrects the llm several times that this means 500g for the user.

## 5. Output

Return:

- Table of items + matched food name + grams + macros
- Daily totals


## What "RAG" Means in This System

Your knowledge base is not documents; it is structured food rows.

You ground the LLM by only allowing it to:

- Output structured items (step 1)
- Select from retrieved candidates (step 3)
- Never invent macros, if something is unknown, predict/assume the macros/kcal but ask the user/tell them that its unknown



## How to Integrate Data into RAG

Build an index table with one row per food entry.

For retrieval you need:

- `doc_id` (unique)
- `source` (`"usda"` or `"off"`)
- `name` (`food_name`)
- `brand` (OFF only)
- `macros` (`kcal/protein/carbs/fat` per 100g)
- `text_for_embedding`

Examples:

- USDA: `"description"`
- OFF: `"product_name | brands | categories_en"`

Then embed `text_for_embedding` and store vectors in FAISS/Chroma.

## Handling "pizza" Correctly (important realism)

`"pepperoni pizza"` is ambiguous. Support uncertainty:

If best-match confidence is low:

- Ask follow-up: "Was it homemade, restaurant, or frozen packaged? Approx grams/slices?"

Or use a default assumption:

- `1 serving pepperoni pizza = 150g` (store this in portion defaults)

For a capstone, it's acceptable to implement defaults and allow user corrections.

## Minimal Pipeline: Component Responsibilities

### Transcription tool #0
- Input audio (as .wav) convert to json.
- OpenAIwhisper + sounddevice

### LLM #1 (Extraction)

- Input: raw text
- Output: JSON items ('name', 'amount', 'unit')

### Retriever (RAG)

- Input: item query text
- Output: top K candidate food rows from USDA/OFF

### LLM #2 (optional reranker)

- Input: item + candidates
- Output: chosen candidate ID + grams assumption if needed

### Deterministic Calculator

- Input: chosen rows + portion grams
- Output: macros per item + totals

### LLM #3 (coaching)

- Input: totals + goal
- Output: tips/substitutions
