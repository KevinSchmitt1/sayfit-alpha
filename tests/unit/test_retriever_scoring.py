import pytest

from step2_retrieval.retriever import (
    _extract_core_name,
    _compute_name_penalty,
    _build_query_variants,
    _safe_float,
)


class TestExtractCoreName:
    def test_no_parens_unchanged(self):
        assert _extract_core_name("banana") == "banana"

    def test_strips_single_parens(self):
        assert _extract_core_name("banana (raw)") == "banana"

    def test_strips_different_parens_content(self):
        assert _extract_core_name("banana (fruit)") == "banana"

    def test_lowercases_result(self):
        assert _extract_core_name("Egg (Boiled)") == "egg"

    def test_multiword_query(self):
        assert _extract_core_name("pepperoni pizza (frozen)") == "pepperoni pizza"

    def test_keep_multiword_query_outside_parens(self):
        assert _extract_core_name("raw nuts") == "raw nuts"


class TestNamePenalty:
    def test_exact_match_returns_1(self):
        assert _compute_name_penalty("egg", "egg") == 1.0

    def test_query_substring_of_candidate_one_extra_word(self):
        # formula: max(0.7, 1.0 - extra_words * 0.06) → 1.0 - 1*0.06 = 0.94
        assert _compute_name_penalty("egg", "boiled egg") == pytest.approx(0.94)

    def test_candidate_substring_of_query(self):
        # "egg" in "egg boiled" → candidate is substring of query → 0.85
        assert _compute_name_penalty("egg boiled", "egg") == 0.85

    def test_no_shared_tokens(self):
        assert _compute_name_penalty("banana", "beef steak") == 0.55

    def test_partial_token_overlap(self):
        # "bolognese pasta" vs "pasta carbonara": shared={"pasta"}, q_tokens=2
        # formula: 0.65 + 0.20 * (1/2) = 0.75 — verify this against the source!
        result = _compute_name_penalty("bolognese pasta", "pasta carbonara")
        assert result == pytest.approx(0.75)

class TestBuildQueryVariants:
    def test_with_descriptor_in_parens(self):
        # docstring example: "egg (boiled)" → 3 variants
        variants = _build_query_variants("egg (boiled)", "egg")
        assert variants == ["egg (boiled)", "egg", "boiled egg"]

    def test_unspecified_descriptor_skipped(self):
        # "unspecified" is not a useful search term — only 2 variants
        variants = _build_query_variants("banana (unspecified)", "banana")
        assert variants == ["banana (unspecified)", "banana"]

    def test_no_parens_single_variant(self):
        # no parens, no hint → nothing to add, just the original
        variants = _build_query_variants("spaghetti bolognese", "spaghetti bolognese")
        assert variants == ["spaghetti bolognese"]

    def test_multi_word_meal_with_hint_l1(self):
        # hint_l1 adds a category-prefixed variant to shift the embedding space
        variants = _build_query_variants(
            "spaghetti bolognese", "spaghetti bolognese", hint_l1="prepared & frozen meals"
        )
        assert variants == [
            "spaghetti bolognese",
            "prepared & frozen meals spaghetti bolognese",
        ]

    def test_full_four_variants(self):
        # docstring example: parens + hint_l1 → 4 variants
        variants = _build_query_variants(
            "pepperoni pizza (frozen)", "pepperoni pizza", hint_l1="prepared & frozen meals"
        )
        assert variants == [
            "pepperoni pizza (frozen)",
            "pepperoni pizza",
            "frozen pepperoni pizza",
            "prepared & frozen meals pepperoni pizza",
        ]

    def test_no_duplicate_variants(self):
        # if core_name == query, it should not be added twice
        variants = _build_query_variants("banana", "banana")
        assert len(variants) == len(set(variants))    


class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float("3.14") == 3.14

    def test_invalid_float_returns_none(self):
        assert _safe_float("not a number") is None

    def test_empty_string_returns_none(self):
        assert _safe_float("") is None

    def test_numpy_nan_returns_none(self):
        assert _safe_float(float("nan")) is None
    
    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_wrong_string_returns_none(self):
        assert _safe_float("123abc") is None
