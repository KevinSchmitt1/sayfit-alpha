"""
Step 2 – Build FAISS Index
===========================
Reads the USDA and OpenFoodFacts CSVs, creates sentence embeddings for each
food entry, and stores everything in a FAISS index + metadata pickle.

This is a ONE-TIME setup step.  Run it before using the retriever:

    python -m step2_retrieval.build_index

The resulting files are saved under  data/faiss_index/
"""

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _clean(val):
    """Return empty string for NaN / None values."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def load_and_prepare_data(sources: list[str] | None = None) -> pd.DataFrame:
    """Load CSV files and create a unified dataframe with embedding text.

    Parameters
    ----------
    sources : list[str] | None
        Which databases to include: ["off"], ["usda"], or ["off", "usda"].
        Defaults to ["off"] (OpenFoodFacts only – fast & lightweight).
    """
    if sources is None:
        sources = ["off"]
    sources = [s.lower() for s in sources]
    frames = []

    # ── OpenFoodFacts ────────────────────────────────────────────────────
    if "off" in sources:
        if config.OFF_CSV.exists():
            print(f"📂 Loading OpenFoodFacts: {config.OFF_CSV}")
            off = pd.read_csv(config.OFF_CSV, dtype=str, low_memory=False)
            off.columns = [c.strip() for c in off.columns]
            # build embedding text: "product_name | brand"
            off["text_for_embedding"] = off.apply(
                lambda r: " | ".join(filter(None, [_clean(r.get("item_name", "")),
                                                    _clean(r.get("brand", ""))])),
                axis=1,
            )
            off["source"] = "openfoodfacts"
            off["doc_id"] = "off_" + off.index.astype(str)
            frames.append(off)
            print(f"   ✅ {len(off):,} rows from OpenFoodFacts")
        else:
            print(f"⚠️  OpenFoodFacts CSV not found at {config.OFF_CSV}")

    # ── USDA ──────────────────────────────────────────────────────────────
    if "usda" in sources:
        if config.USDA_CSV.exists():
            print(f"📂 Loading USDA: {config.USDA_CSV}")
            usda = pd.read_csv(config.USDA_CSV, dtype=str, low_memory=False)
            usda.columns = [c.strip() for c in usda.columns]
            # build embedding text: item_name (+ brand if present)
            usda["text_for_embedding"] = usda.apply(
                lambda r: " | ".join(filter(None, [_clean(r.get("item_name", "")),
                                                    _clean(r.get("brand", ""))])),
                axis=1,
            )
            usda["source"] = "usda"
            usda["doc_id"] = "usda_" + usda.index.astype(str)
            frames.append(usda)
            print(f"   ✅ {len(usda):,} rows from USDA")
        else:
            print(f"⚠️  USDA CSV not found at {config.USDA_CSV}")

    if not frames:
        raise FileNotFoundError("No food CSVs found. Extract food_dbs.zip into data/")

    df = pd.concat(frames, ignore_index=True)

    # drop rows where we have no text to embed
    df = df[df["text_for_embedding"].str.len() > 0].reset_index(drop=True)
    print(f"\n📊 Total food entries to index: {len(df):,}")
    return df


def build_index(batch_size: int | None = None, sources: list[str] | None = None):
    """Build the FAISS index and save it alongside the metadata.

    Parameters
    ----------
    batch_size : int | None
        Encoding batch size (default from config).
    sources : list[str] | None
        ["off"], ["usda"], or ["off", "usda"]. Default: ["off"].
    """
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE

    df = load_and_prepare_data(sources=sources)
    texts = df["text_for_embedding"].tolist()

    # ── load embedding model ─────────────────────────────────────────────
    print(f"\n🤖 Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    # Use MPS (Apple Silicon GPU) or CUDA if available for faster encoding
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
    meta_path = config.INDEX_DIR / "food_meta.pkl"

    faiss.write_index(index, str(index_path))
    print(f"   💾 FAISS index saved: {index_path}")

    # metadata: keep all columns we need for retrieval
    keep_cols = [
        "doc_id", "source", "item_id", "item_name", "brand",
        "kcal_100g", "protein_100g", "carbs_100g", "fat_100g",
        "text_for_embedding",
    ]
    # add optional columns if present
    for c in ["serving_size", "quantity", "portion_description", "gram_weight"]:
        if c in df.columns:
            keep_cols.append(c)
    keep_cols = [c for c in keep_cols if c in df.columns]
    meta_df = df[keep_cols].copy()
    meta_df.to_pickle(str(meta_path))
    print(f"   💾 Metadata saved: {meta_path}")
    print("\n🎉 Index build complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS food index")
    parser.add_argument("--sources", nargs="+", default=["off"],
                        choices=["off", "usda"],
                        help="Which databases to index (default: off). Use 'off usda' for both.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Encoding batch size")
    args = parser.parse_args()
    build_index(batch_size=args.batch_size, sources=args.sources)
