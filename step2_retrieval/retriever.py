"""
Step 2 – Food Retriever (FAISS vector search)
===============================================
Given a list of query strings (from Step 1), search the FAISS index for the
top-K most relevant food entries, apply a scoring penalty for non-exact
matches, and return structured candidate lists.

Usage (as library):
    from step2_retrieval.retriever import retrieve
    results = retrieve(["pepperoni pizza (frozen)", "egg (boiled)"])

Standalone:
    python -m step2_retrieval.run [--input step1_output.json]
"""

import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ── lazy-loaded singletons ──────────────────────────────────────────────────
_model = None
_index = None
_meta = None


def _load_resources():
    """Load embedding model, FAISS index, and metadata (once)."""
    global _model, _index, _meta

    if _model is not None:
        return

    index_path = config.INDEX_DIR / "food.index"
    meta_path = config.INDEX_DIR / "food_meta.pkl"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {config.INDEX_DIR}.\n"
            "Run  python -m step2_retrieval.build_index  first."
        )

    print("\n📦 [Step 2] Loading retrieval resources …")
    print(f"   FAISS index : {index_path}")
    _index = faiss.read_index(str(index_path))
    print(f"   Index size  : {_index.ntotal:,} vectors")

    _meta = pd.read_pickle(str(meta_path))
    print(f"   Metadata    : {len(_meta):,} rows")

    print(f"   Embedding   : {config.EMBEDDING_MODEL_NAME}")
    _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    print("   ✅ Resources loaded.")


# ── scoring helpers ─────────────────────────────────────────────────────────

def _extract_core_name(query: str) -> str:
    """Strip parenthetical descriptions from query, e.g. 'egg (boiled)' -> 'egg'."""
    return re.sub(r"\s*\(.*?\)", "", query).strip().lower()


def _compute_name_penalty(query_core: str, candidate_name: str) -> float:
    """
    Penalise candidates where the food name doesn't closely match the query.

    Returns a multiplier in [0.5, 1.0]:
      - 1.0  if candidate matches query closely
      - lower if the candidate has extra qualifier-words that change meaning
    """
    cand = candidate_name.lower().strip()

    # exact match → no penalty
    if query_core == cand:
        return 1.0

    # query is a substring at word boundary → small penalty
    if query_core in cand:
        # penalise based on how many extra words
        extra_words = len(cand.split()) - len(query_core.split())
        penalty = max(0.7, 1.0 - extra_words * 0.06)
        return penalty

    # candidate is a substring of query → slight penalty
    if cand in query_core:
        return 0.85

    # no substring match → moderate penalty
    return 0.6


def retrieve(
    queries: list[str],
    top_k: int | None = None,
    category_hints: list[str] | None = None,
    category_boost: float | None = None,
) -> dict:
    """
    Retrieve top-K candidates for each query.

    Parameters
    ----------
    queries : list[str]
        Search queries, e.g. ["pepperoni pizza (frozen)", "egg (boiled)"]
    top_k : int
        Number of candidates per query (default from config).
    category_hints : list[dict] | None
        Predicted categories per query from Step 1.5, e.g.
        [{"cat_l1": "dairy & eggs", "cat_l2": "eggs"}, ...].
        L1 match → ×boost; L2 match (finer) → ×boost again (stacked).
        No hard filtering — candidates outside the category are still kept.
    category_boost : float
        Multiplicative boost per matching level (default 1.15 = +15 %).
        L1+L2 match → ×1.15×1.15 ≈ +32 %.  Set to 1.0 to disable.

    Returns
    -------
    dict with "items" list, each containing name, candidates, and nutrition.
    """
    _load_resources()
    top_k = top_k or config.TOP_K_CANDIDATES
    category_boost = category_boost if category_boost is not None else config.ONTOLOGY_CATEGORY_BOOST
    # retrieve more than needed so we can re-rank
    search_k = min(top_k * 3, _index.ntotal)

    has_cat_l1 = "cat_l1" in _meta.columns
    has_cat_l2 = "cat_l2" in _meta.columns
    has_cat_l3 = "cat_l3" in _meta.columns
    hints = category_hints or []

    rank_boosts = [
        config.ONTOLOGY_BOOST_RANK1,
        config.ONTOLOGY_BOOST_RANK2,
        config.ONTOLOGY_BOOST_RANK3,
    ]

    print(f"\n🔎 [Step 2] Retrieving candidates (top_k={top_k}) …")
    if hints:
        print(f"   🏷️  Ontology hints active (rank boosts={rank_boosts}) – "
              f"{len(hints)} hint(s) provided")

    # encode all queries at once
    query_vecs = _model.encode(queries, normalize_embeddings=True)
    query_vecs = np.array(query_vecs, dtype="float32")

    # FAISS search
    scores, indices = _index.search(query_vecs, search_k)

    results = {"items": []}

    for q_idx, query in enumerate(queries):
        core_name = _extract_core_name(query)
        hint = hints[q_idx] if q_idx < len(hints) else {}
        if isinstance(hint, dict):
            hint_ranked_l1: list[str] = hint.get("ranked_l1", [])
            # backwards-compat: if no ranked_l1, fall back to flat cat_l1
            if not hint_ranked_l1 and hint.get("cat_l1", "") not in ("", "other"):
                hint_ranked_l1 = [hint["cat_l1"]]
            hint_l2: str = hint.get("cat_l2", "")
        else:
            hint_ranked_l1 = [hint] if hint and hint != "other" else []
            hint_l2 = ""
        hint_l1 = hint_ranked_l1[0] if hint_ranked_l1 else ""
        print(f"   Query: \"{query}\" → core=\"{core_name}\""
              + (f", hints={hint_ranked_l1}" if hint_ranked_l1 else ""))

        candidates = []
        for rank in range(search_k):
            doc_idx = int(indices[q_idx][rank])
            if doc_idx < 0:
                continue
            sim_score = float(scores[q_idx][rank])
            row = _meta.iloc[doc_idx]
            cand_name = str(row.get("item_name", ""))

            # categories from metadata (populated at index-build time)
            cand_l1 = str(row.get("cat_l1", "")) if has_cat_l1 else ""
            cand_l2 = str(row.get("cat_l2", "")) if has_cat_l2 else ""
            cand_l3 = str(row.get("cat_l3", "")) if has_cat_l3 else ""

            # apply name-match penalty
            penalty = _compute_name_penalty(core_name, cand_name)
            adjusted_score = sim_score * penalty

            # apply ontology boost (soft – no hard exclusion)
            # ranked L1 hints: rank-1 match → strongest boost, rank-2 → less, etc.
            # Only the best (lowest-rank) matching hint fires.
            cat_match_rank = next(
                (i for i, l1h in enumerate(hint_ranked_l1)
                 if l1h and l1h != "other" and cand_l1 == l1h),
                -1,
            )
            cat_match_l2 = bool(hint_l2 and cand_l2 and cand_l2 == hint_l2)
            if cat_match_rank >= 0:
                boost = rank_boosts[cat_match_rank] if cat_match_rank < len(rank_boosts) else rank_boosts[-1]
                adjusted_score *= boost
            if cat_match_l2:
                adjusted_score *= category_boost  # additional boost for L2 match

            candidates.append({
                "doc_id": str(row.get("doc_id", "")),
                "source": str(row.get("source", "")),
                "item_name": cand_name,
                "brand": str(row.get("brand", "")) if pd.notna(row.get("brand")) else "",
                "cat_l1": cand_l1,
                "cat_l2": cand_l2,
                "cat_l3": cand_l3,
                "cat_match": cat_match_rank >= 0 or cat_match_l2,
                "raw_score": round(sim_score, 4),
                "adjusted_score": round(adjusted_score, 4),
                "nutrition_per_100g": {
                    "calories": _safe_float(row.get("kcal_100g")),
                    "protein": _safe_float(row.get("protein_100g")),
                    "fat": _safe_float(row.get("fat_100g")),
                    "carbs": _safe_float(row.get("carbs_100g")),
                },
                "portion_info": {
                    "serving_size": str(row.get("serving_size", "")) if pd.notna(row.get("serving_size", None)) else "",
                    "portion_description": str(row.get("portion_description", "")) if pd.notna(row.get("portion_description", None)) else "",
                    "gram_weight": _safe_float(row.get("gram_weight")),
                },
            })

        # sort by adjusted score and keep top_k
        candidates.sort(key=lambda c: c["adjusted_score"], reverse=True)
        candidates = candidates[:top_k]

        best = candidates[0] if candidates else None
        if best:
            print(f"   → Best match: \"{best['item_name']}\" "
                  f"[{best['cat_l1']} / {best['cat_l2']}] "
                  f"(score={best['adjusted_score']:.3f})")
        else:
            print("   → No matches")

        results["items"].append({
            "query": query,
            "core_name": core_name,
            "category_hint": {"cat_l1": hint_l1, "ranked_l1": hint_ranked_l1, "cat_l2": hint_l2},
            "candidates": candidates,
        })

    print(f"   ✅ Retrieved candidates for {len(queries)} queries.")
    return results


def _safe_float(val) -> float | None:
    """Convert to float, return None for missing values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None
