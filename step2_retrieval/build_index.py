"""
Step 2 – Build FAISS Index
===========================
Reads usda_final.csv (item_name, macros, cat_l1, cat_l2, cat_l3), creates
sentence embeddings and stores everything in a FAISS index + metadata pickle.

This is a ONE-TIME setup step.  Run it before using the retriever:

    python -m step2_retrieval.build_index
    # or:
    python main.py --build-index

The resulting files are saved under  data/faiss_index/
"""

import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _clean(val) -> str:
    """Return empty string for NaN / None values."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def load_and_prepare_data() -> pd.DataFrame:
    """Load usda_final.csv and prepare it for embedding."""

    # ── USDA final (with real ontology categories) ────────────────────────
    #csv_path = config.USDA_FINAL_CSV
    csv_path = config.COMBINED_FINAL_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"usda_final.csv not found at {csv_path}.\n"
            "Place the file in the data/ directory and try again."
        )

    print(f"📂 Loading USDA final: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"   ✅ {len(df):,} rows loaded")

    # build embedding text from item_name only (no brand in this dataset)
    df["text_for_embedding"] = df["item_name"].fillna("").str.strip()
    df["source"] = df.get("source", pd.Series(["usda"] * len(df)))
    df["doc_id"] = "usda_" + df.index.astype(str)

    # drop rows with no embeddable text
    df = df[df["text_for_embedding"].str.len() > 0].reset_index(drop=True)

    # ── Log ontology distribution ─────────────────────────────────────────
    print("\n🏷️  Ontology category distribution (L1):")
    cat_counts = df["cat_l1"].fillna("(missing)").value_counts()
    for cat, cnt in cat_counts.items():
        print(f"     {cat:35s}: {cnt:,}")

    print(f"\n📊 Total entries to index: {len(df):,}")
    return df


def build_index(batch_size: int | None = None):
    """Build the FAISS index and save it alongside the metadata."""
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE

    df = load_and_prepare_data()
    texts = df["text_for_embedding"].tolist()

    # ── load embedding model ─────────────────────────────────────────────
    print(f"\n🤖 Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    print(f"   Device: {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"   Embedding dimension: {dim}")

    # ── encode in batches ────────────────────────────────────────────────
    print(f"\n⏳ Encoding {len(texts):,} entries (batch_size={batch_size}) …")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via inner product
    )
    elapsed = time.time() - t0
    print(f"   ✅ Encoding done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} entries/sec)")

    embeddings = np.array(embeddings, dtype="float32")

    # ── build FAISS index (inner product = cosine on normalised vectors) ─
    print("\n🏗️  Building FAISS index …")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"   ✅ Index contains {index.ntotal:,} vectors")

    # ── save index + metadata ────────────────────────────────────────────
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = config.INDEX_DIR / "food.index"
    meta_path  = config.INDEX_DIR / "food_meta.pkl"

    faiss.write_index(index, str(index_path))
    print(f"   💾 FAISS index saved: {index_path}")

    # keep all columns relevant for retrieval + ontology
    keep_cols = [
        "doc_id", "source", "item_name",
        "kcal_100g", "protein_100g", "carbs_100g", "fat_100g",
        "cat_l1", "cat_l2", "cat_l3",
        "text_for_embedding",
    ]
    # optional columns present in some variants of the data
    for c in ["item_id", "brand", "serving_size", "quantity",
              "portion_description", "gram_weight"]:
        if c in df.columns:
            keep_cols.append(c)

    keep_cols = [c for c in keep_cols if c in df.columns]
    meta_df = df[keep_cols].copy()
    meta_df.to_pickle(str(meta_path))
    print(f"   💾 Metadata saved : {meta_path}  ({len(meta_df):,} rows, {len(keep_cols)} cols)")
    print("\n🎉 Index build complete!")


if __name__ == "__main__":
    build_index()
