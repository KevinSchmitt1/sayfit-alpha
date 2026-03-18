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


def _log(*args, **kwargs):
    """Print only when developer mode is active."""
    if config.DEV_MODE:
        print(*args, **kwargs)

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

    _log("\n📦 [Step 2] Loading retrieval resources …")
    _log(f"   FAISS index : {index_path}")
    _index = faiss.read_index(str(index_path))
    _log(f"   Index size  : {_index.ntotal:,} vectors")

    _meta = pd.read_pickle(str(meta_path))
    _log(f"   Metadata    : {len(_meta):,} rows")

    _log(f"   Embedding   : {config.EMBEDDING_MODEL_NAME}")
    # Reuse Step 1.5's model if it was already loaded (avoids two instances)
    def _load_st_silent():
        """Load SentenceTransformer, suppressing C-level fd output in normal mode."""
        if config.DEV_MODE:
            return SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        import os as _os
        _devnull = _os.open(_os.devnull, _os.O_WRONLY)
        _saved_out, _saved_err = _os.dup(1), _os.dup(2)
        _os.dup2(_devnull, 1); _os.dup2(_devnull, 2)
        try:
            return SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        finally:
            _os.dup2(_saved_out, 1); _os.dup2(_saved_err, 2)
            _os.close(_saved_out); _os.close(_saved_err); _os.close(_devnull)

    try:
        import step1_5_ontology_filter.ontology_filter as _ont
        if _ont._embed_model is not None:
            _model = _ont._embed_model
            _log("   ♻️  Reusing embedding model from Step 1.5")
        else:
            _model = _load_st_silent()
            _ont._embed_model = _model
    except ImportError:
        _model = _load_st_silent()
    _log("   ✅ Resources loaded.")

    # Share model with Step 1.5 so it doesn't load a second instance
    try:
        import step1_5_ontology_filter.ontology_filter as _ont
        _ont._embed_model = _model
    except ImportError:
        pass


# ── scoring helpers ─────────────────────────────────────────────────────────

def _extract_core_name(query: str) -> str:
    """Strip parenthetical descriptions from query, e.g. 'egg (boiled)' -> 'egg'."""
    return re.sub(r"\s*\(.*?\)", "", query).strip().lower()


def _build_query_variants(query: str, core_name: str, hint_l1: str = "") -> list[str]:
    """
    Generate 2–4 search variants for multi-query pooling.

    Examples:
        "pepperoni pizza (frozen)"  →  ["pepperoni pizza (frozen)", "pepperoni pizza",
                                         "frozen pepperoni pizza",
                                         "prepared & frozen meals pepperoni pizza"]
        "egg (boiled)"              →  ["egg (boiled)", "egg", "boiled egg"]
        "banana (unspecified)"      →  ["banana (unspecified)", "banana"]
    """
    variants = [query]

    # core name only (drop description parens) if it differs
    if core_name != query.lower().strip():
        variants.append(core_name)

    # rephrased: "item (desc)" → "desc item" — skip "unspecified"
    m = re.search(r"\(([^)]+)\)", query)
    if m:
        desc = m.group(1).strip().lower()
        if desc and desc != "unspecified":
            rephrased = f"{desc} {core_name}"
            if rephrased not in variants:
                variants.append(rephrased)

    # L1-category-prefixed variant: shifts the embedding towards the correct
    # semantic space even without rebuilding the index
    if hint_l1 and hint_l1 not in ("", "other"):
        cat_variant = f"{hint_l1} {core_name}"
        if cat_variant not in variants:
            variants.append(cat_variant)

    return variants


def _compute_name_penalty(query_core: str, candidate_name: str) -> float:
    """
    Penalise candidates where the food name doesn't closely match the query.

    Returns a multiplier in [0.55, 1.0]:
      - 1.0  exact match
      - 0.85 candidate is a substring of the query
      - 0.70–0.85 query is a substring of candidate (with extra words)
      - 0.65–0.85 token overlap (shared keywords like 'bolognese')
      - 0.55 no shared tokens at all
    """
    cand = candidate_name.lower().strip()
    q = query_core.lower().strip()

    # exact match → no penalty
    if q == cand:
        return 1.0

    # query is a substring of candidate → small penalty based on extra words
    if q in cand:
        extra_words = len(cand.split()) - len(q.split())
        return max(0.7, 1.0 - extra_words * 0.06)

    # candidate is a substring of query → slight penalty
    if cand in q:
        return 0.85

    # token overlap: reward shared meaningful keywords (e.g. 'bolognese')
    q_tokens = set(q.split())
    c_tokens = set(cand.split())
    shared = q_tokens & c_tokens
    if shared:
        # fraction of query tokens found in candidate, scaled to [0.65, 0.85]
        overlap_ratio = len(shared) / len(q_tokens)
        return 0.65 + 0.20 * overlap_ratio

    # no shared tokens at all → harshest penalty
    return 0.55


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

    use_pooling = config.MULTI_QUERY_POOLING

    _log(f"\n🔎 [Step 2] Retrieving candidates (top_k={top_k}, pooling={use_pooling}) …")
    if hints:
        _log(f"   🏷️  Ontology hints active (rank boosts={rank_boosts}) – "
              f"{len(hints)} hint(s) provided")

    # Pre-compute core names for all queries
    core_names = [_extract_core_name(q) for q in queries]

    # Build flat variant list + index ranges per original query
    all_variants: list[str] = []
    variant_ranges: list[tuple[int, int]] = []
    for i, (q, core) in enumerate(zip(queries, core_names)):
        # extract the top-ranked L1 hint for this query (for category-prefixed variant)
        _h = hints[i] if i < len(hints) else {}
        if isinstance(_h, dict):
            _l1_for_variant = (_h.get("ranked_l1") or [_h.get("cat_l1", "")])[0]
        else:
            _l1_for_variant = _h if _h and _h != "other" else ""
        vlist = _build_query_variants(q, core, hint_l1=_l1_for_variant) if use_pooling else [q]
        start = len(all_variants)
        all_variants.extend(vlist)
        variant_ranges.append((start, start + len(vlist)))

    # Encode all variants in one batch
    all_vecs = _model.encode(all_variants, normalize_embeddings=True)
    all_vecs = np.array(all_vecs, dtype="float32")

    # FAISS search for every variant at once
    all_scores, all_indices = _index.search(all_vecs, search_k)

    results = {"items": []}

    for q_idx, query in enumerate(queries):
        core_name = core_names[q_idx]
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

        v_start, v_end = variant_ranges[q_idx]
        n_variants = v_end - v_start
        variant_labels = all_variants[v_start:v_end]
        _log(f"   Query: \"{query}\" → core=\"{core_name}\""
              + (f", hints={hint_ranked_l1}" if hint_ranked_l1 else "")
              + (f", variants={variant_labels[1:]}" if n_variants > 1 else ""))

        # Merge candidates from all variants: doc_id → best-scored candidate
        merged: dict[str, dict] = {}

        for v_offset in range(n_variants):
            v_abs = v_start + v_offset
            for rank in range(search_k):
                doc_idx = int(all_indices[v_abs][rank])
                if doc_idx < 0:
                    continue
                sim_score = float(all_scores[v_abs][rank])
                row = _meta.iloc[doc_idx]
                cand_name = str(row.get("item_name", ""))
                doc_id = str(row.get("doc_id", ""))

                cand_l1 = str(row.get("cat_l1", "")) if has_cat_l1 else ""
                cand_l2 = str(row.get("cat_l2", "")) if has_cat_l2 else ""
                cand_l3 = str(row.get("cat_l3", "")) if has_cat_l3 else ""

                # name penalty always uses the original core_name (consistent across variants)
                penalty = _compute_name_penalty(core_name, cand_name)
                adjusted_score = sim_score * penalty

                # ontology boost (soft — no hard exclusion)
                cat_match_rank = next(
                    (i for i, l1h in enumerate(hint_ranked_l1)
                     if l1h and l1h != "other" and cand_l1 == l1h),
                    -1,
                )
                cat_match_l2 = bool(hint_l2 and cand_l2 and cand_l2 == hint_l2)
                if cat_match_rank >= 0:
                    boost = rank_boosts[cat_match_rank] if cat_match_rank < len(rank_boosts) else rank_boosts[-1]
                    adjusted_score *= boost
                # L2 boost only stacks when the candidate's L1 is the TOP-ranked category.
                # Prevents wrong-category items from gaining L2 boost via a coincidental
                # L2 match (e.g. raw 'spaghetti' sharing 'pasta & noodles' L2 with a
                # prepared bolognese query whose L2 was mis-predicted).
                if cat_match_l2 and cat_match_rank == 0:
                    adjusted_score *= category_boost

                # keep best adjusted_score per doc_id across variants
                if doc_id not in merged or adjusted_score > merged[doc_id]["adjusted_score"]:
                    merged[doc_id] = {
                        "doc_id": doc_id,
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
                    }

        # sort merged pool by adjusted_score, keep top_k
        candidates = sorted(merged.values(), key=lambda c: c["adjusted_score"], reverse=True)[:top_k]

        best = candidates[0] if candidates else None
        if best:
            _log(f"   → Best match: \"{best['item_name']}\" "
                  f"[{best['cat_l1']} / {best['cat_l2']}] "
                  f"(score={best['adjusted_score']:.3f}, pool={len(merged)} unique)")
        else:
            _log("   → No matches")

        results["items"].append({
            "query": query,
            "core_name": core_name,
            "category_hint": {"cat_l1": hint_l1, "ranked_l1": hint_ranked_l1, "cat_l2": hint_l2},
            "candidates": candidates,
        })

    _log(f"   ✅ Retrieved candidates for {len(queries)} queries.")
    return results


def _safe_float(val) -> float | None:
    """Convert to float, return None for missing values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return None
