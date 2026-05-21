from unittest.mock import patch

import pytest

from step1_5_ontology_filter.ontology_filter import (
    apply_ontology_filter,
    classify_item_name,
    resolve_portion_hint,
)


# ── Heuristic path ────────────────────────────────────────────────────────────
# Tests for rule-based classification internals. No CSV, no embeddings needed:
# the L1 seed keyword table (_L1_SEEDS) is hard-coded in memory.


@pytest.fixture
def seeds_only(monkeypatch):
    """Patch out CSV-derived structures so classify_item_name uses only the
    hard-coded _L1_SEEDS (step 4). This isolates seed logic from CSV data quality."""
    import step1_5_ontology_filter.ontology_filter as ont
    monkeypatch.setattr(ont, "_loaded", True)
    monkeypatch.setattr(ont, "_exact", {})
    monkeypatch.setattr(ont, "_name_index", [])
    monkeypatch.setattr(ont, "_l2_keywords", [])


@pytest.mark.heuristic
class TestClassifyItemName:
    def test_known_seed_maps_to_correct_l1(self, seeds_only):
        l1, _, _ = classify_item_name("banana")
        assert l1 == "fruits"

    def test_multi_word_seed_match(self, seeds_only):
        # "chicken breast" is explicitly in _L1_SEEDS under "poultry"
        l1, _, _ = classify_item_name("chicken breast")
        assert l1 == "poultry"

    def test_completely_unknown_returns_other(self, seeds_only):
        l1, l2, l3 = classify_item_name("zzzunknownfoodzzz")
        assert l1 == "other"
        assert l2 == ""
        assert l3 == ""

    def test_salmon_maps_to_fish(self, seeds_only):
        l1, _, _ = classify_item_name("salmon")
        assert l1 == "fish & seafood"

    def test_oat_maps_to_grains(self, seeds_only):
        l1, _, _ = classify_item_name("oatmeal")
        assert l1 == "grains & pasta"


@pytest.mark.heuristic
class TestResolvePortion:
    def test_explicit_grams_bypasses_all_tiers(self):
        result = resolve_portion_hint("chicken", 150, "g", uid="", cat_l1="poultry")
        assert result["grams"] == 150.0
        assert result["source"] == "explicit_grams"

    def test_explicit_ml_treated_same_as_grams(self):
        result = resolve_portion_hint("orange juice", 200, "ml", uid="", cat_l1="beverages")
        assert result["grams"] == 200.0
        assert result["source"] == "explicit_grams"

    def test_category_default_used_as_fallback(self):
        # No user prefs (uid=""), no ontology file in CI → falls through to Tier 3
        result = resolve_portion_hint("banana", None, None, uid="", cat_l1="fruits")
        assert result["grams"] > 0
        assert result["source"] in ("ontology_default", "ontology_unit", "category_default", "portion_defaults", "fallback")

    def test_vague_quantity_applies_fraction_multiplier(self):
        # "some" is a vague quantity → multiplier becomes 0.2, bypasses unit lookup
        baseline = resolve_portion_hint("chicken", None, None, uid="", cat_l1="poultry")
        vague = resolve_portion_hint("chicken", None, "some", uid="", cat_l1="poultry")
        assert vague["grams"] < baseline["grams"]

    def test_unknown_category_falls_back_to_100g(self):
        result = resolve_portion_hint("xyzzy_food", None, None, uid="", cat_l1="")
        assert result["source"] == "fallback"
        assert result["grams"] == 100.0

    def test_quantity_multiplier_applied(self):
        single = resolve_portion_hint("chicken", 1, "g", uid="", cat_l1="poultry")
        triple = resolve_portion_hint("chicken", 3, "g", uid="", cat_l1="poultry")
        assert triple["grams"] == pytest.approx(single["grams"] * 3)


# ── LLM path ──────────────────────────────────────────────────────────────────
# Tests for apply_ontology_filter() wiring. classify_l2_semantic is patched to
# avoid loading the SentenceTransformer model during unit tests.

_SINGLE_ITEM_EXTRACTION = {
    "items": {
        "item1": {
            "item_name": "banana",
            "quantity_raw": None,
            "quantity_parsed": None,
            "unit_hint": None,
            "description": "raw fruit",
            "category_ranks": ["fruits"],
            "uid": "",
        }
    },
    "queries": ["banana (raw fruit)"],
}

_TWO_ITEM_EXTRACTION = {
    "items": {
        "item1": {
            "item_name": "chicken breast",
            "quantity_raw": "150g",
            "quantity_parsed": 150,
            "unit_hint": "g",
            "description": "grilled",
            "category_ranks": ["poultry"],
            "uid": "",
        },
        "item2": {
            "item_name": "rice",
            "quantity_raw": None,
            "quantity_parsed": None,
            "unit_hint": None,
            "description": "boiled",
            "category_ranks": ["grains & pasta"],
            "uid": "",
        },
    },
    "queries": ["chicken breast (grilled)", "rice (boiled)"],
}


@pytest.mark.llm_path
class TestApplyOntologyFilter:
    def test_output_has_ontology_and_category_hints_keys(self):
        with patch("step1_5_ontology_filter.ontology_filter.classify_l2_semantic", return_value=""):
            result = apply_ontology_filter(_SINGLE_ITEM_EXTRACTION)
        assert "ontology" in result
        assert "category_hints" in result

    def test_category_hints_count_matches_queries(self):
        with patch("step1_5_ontology_filter.ontology_filter.classify_l2_semantic", return_value=""):
            result = apply_ontology_filter(_TWO_ITEM_EXTRACTION)
        assert len(result["category_hints"]) == len(_TWO_ITEM_EXTRACTION["queries"])

    def test_llm_ranks_set_source_to_llm(self):
        # When category_ranks is provided by the LLM, source should be "llm"
        with patch("step1_5_ontology_filter.ontology_filter.classify_l2_semantic", return_value=""):
            result = apply_ontology_filter(_SINGLE_ITEM_EXTRACTION)
        assert result["ontology"]["item1"]["source"] == "llm"

    def test_heuristic_source_when_no_category_ranks(self):
        # When category_ranks is absent, falls back to rule-based classification
        extraction = {
            "items": {
                "item1": {
                    "item_name": "banana",
                    "quantity_raw": None,
                    "quantity_parsed": None,
                    "unit_hint": None,
                    "description": "unspecified",
                    "category_ranks": [],
                    "uid": "",
                }
            },
            "queries": ["banana (unspecified)"],
        }
        result = apply_ontology_filter(extraction)
        assert result["ontology"]["item1"]["source"] == "rules"

    def test_llm_l1_propagated_to_category_hints(self):
        with patch("step1_5_ontology_filter.ontology_filter.classify_l2_semantic", return_value=""):
            result = apply_ontology_filter(_SINGLE_ITEM_EXTRACTION)
        assert result["category_hints"][0]["cat_l1"] == "fruits"

    def test_portion_hint_attached_to_each_item(self):
        with patch("step1_5_ontology_filter.ontology_filter.classify_l2_semantic", return_value=""):
            result = apply_ontology_filter(_SINGLE_ITEM_EXTRACTION)
        assert "portion_hint" in result["ontology"]["item1"]
        assert "grams" in result["ontology"]["item1"]["portion_hint"]

    def test_empty_items_returns_empty_ontology(self):
        result = apply_ontology_filter({"items": {}, "queries": []})
        assert result["ontology"] == {}
        assert result["category_hints"] == []

    def test_original_extraction_keys_preserved(self):
        # apply_ontology_filter extends the dict — it must not drop existing keys
        with patch("step1_5_ontology_filter.ontology_filter.classify_l2_semantic", return_value=""):
            result = apply_ontology_filter(_SINGLE_ITEM_EXTRACTION)
        assert "items" in result
        assert "queries" in result
