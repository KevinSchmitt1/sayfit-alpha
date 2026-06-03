# FAISS Index Output Contract — sayfit-alpha interface

When the data pipeline builds the FAISS index, it must output **two files** into `data/faiss_index/`:

```
data/faiss_index/
├── food.index        ← FAISS binary index (faiss.write_index)
└── food_meta.pkl     ← pandas DataFrame pickled (df.to_pickle)
```

Both files are required. `sayfit-alpha` will refuse to start if either is missing.

---

## Why two files?

FAISS stores only vectors — it has no concept of what food a vector represents.
When a query returns vector positions `[412, 7, 3891, ...]`, `food_meta.pkl` is the lookup table
that maps position → food name, macros, category.

**Row `i` in the pkl must correspond to vector `i` in the index.**  
Do not shuffle or reindex the DataFrame between `index.add(embeddings)` and `df.to_pickle(...)`.

---

## Required columns in `food_meta.pkl`

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | str | Unique row ID, e.g. `"usda_0"` |
| `source` | str | Dataset origin, e.g. `"usda"`, `"off"` |
| `item_name` | str | Human-readable food name returned to the user |
| `kcal_100g` | float | |
| `protein_100g` | float | |
| `carbs_100g` | float | |
| `fat_100g` | float | |
| `cat_l1` | str | Ontology category level 1 |
| `cat_l2` | str | Ontology category level 2 |
| `cat_l3` | str | Ontology category level 3 |
| `text_for_embedding` | str | The string that was actually encoded into the vector |

Optional columns (include if present): `item_id`, `brand`, `serving_size`, `quantity`, `portion_description`, `gram_weight`

---

## Minimal save snippet

```python
import faiss
import pandas as pd

# ... build embeddings and df with correct columns ...

# 1. Save FAISS index
faiss.write_index(index, "data/faiss_index/food.index")

# 2. Save metadata — row order MUST match vector order in the index
keep_cols = [
    "doc_id", "source", "item_name",
    "kcal_100g", "protein_100g", "carbs_100g", "fat_100g",
    "cat_l1", "cat_l2", "cat_l3",
    "text_for_embedding",
]
df[keep_cols].to_pickle("data/faiss_index/food_meta.pkl")
```

---

## What sayfit-alpha does with these files

```python
# retriever.py (simplified)
index = faiss.read_index("data/faiss_index/food.index")
meta  = pd.read_pickle("data/faiss_index/food_meta.pkl")

scores, indices = index.search(query_vector, k=20)
results = meta.iloc[indices[0]]   # row lookup by position
```

---

## Embedding model

The same model must be used for both building the index and querying it:

```
sentence-transformers/all-MiniLM-L6-v2
```

Configured in `sayfit-alpha/config.py` as `EMBEDDING_MODEL_NAME`.  
Use `normalize_embeddings=True` when encoding (cosine similarity via inner product on normalised vectors).
