import pytest

from step3_reranker.reranker import (
    _description_penalty,
    _is_absolute_grams,
    _parse_quantity,
    rerank_all,
    rerank_single_item_heuristic,
)

FAKE_CANDIDATE = {
    "doc_id": "abc123",
    "item_name": "banana raw",
    "source": "usda",
    "brand": "",
    "adjusted_score": 0.85,
    "nutrition_per_100g": {"calories": 89.0, "protein": 1.1, "fat": 0.3, "carbs": 23.0},
}

FAKE_CANDIDATE_LOW = {
    "doc_id": "ghi789",
    "item_name": "mystery food",
    "source": "usda",
    "brand": "",
    "adjusted_score": 0.40,
    "nutrition_per_100g": {"calories": 100.0, "protein": 5.0, "fat": 2.0, "carbs": 15.0},
}


class TestParseQuantity:
    def test_none_returns_none(self):
        assert _parse_quantity(None) is None

    def test_integer_string(self):
        assert _parse_quantity("3") == 3.0

    def test_float_string(self):
        assert _parse_quantity("2.5") == 2.5

    def test_word_two(self):
        assert _parse_quantity("two") == 2.0

    def test_word_half(self):
        assert _parse_quantity("half") == 0.5

    def test_word_a(self):
        assert _parse_quantity("a") == 1.0

    def test_gram_suffix(self):
        assert _parse_quantity("150g") == 150.0

    def test_ml_suffix(self):
        assert _parse_quantity("200ml") == 200.0

    def test_unknown_word_returns_none(self):
        assert _parse_quantity("handful") is None


class TestIsAbsoluteGrams:
    def test_gram_suffix_true(self):
        assert _is_absolute_grams("150g") is True

    def test_ml_suffix_true(self):
        assert _is_absolute_grams("200ml") is True

    def test_gram_with_space_true(self):
        assert _is_absolute_grams("200 g") is True

    def test_plain_number_false(self):
        assert _is_absolute_grams("3") is False

    def test_none_false(self):
        assert _is_absolute_grams(None) is False

    def test_word_quantity_false(self):
        assert _is_absolute_grams("two") is False


class TestDescriptionPenalty:
    def test_raw_vs_pie_returns_half(self):
        assert _description_penalty("raw fruit", "peach pie") == 0.5

    def test_raw_vs_plain_returns_one(self):
        assert _description_penalty("raw fruit", "peach raw") == 1.0

    def test_unspecified_vs_raw_returns_one(self):
        assert _description_penalty("unspecified", "banana raw") == 1.0

    def test_fresh_vs_fried_returns_half(self):
        assert _description_penalty("fresh", "fried apple") == 0.5

    def test_raw_vs_juice_returns_half(self):
        assert _description_penalty("raw", "apple juice") == 0.5


class TestRerankSingleItemHeuristic:
    def _item(self, name, qty_raw=None, description="unspecified", portion_hint=None):
        item = {"item_name": name, "quantity_raw": qty_raw, "description": description}
        if portion_hint:
            item["portion_hint"] = portion_hint
        return item

    def test_returns_required_keys(self):
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [FAKE_CANDIDATE]
        )
        for key in ("item_name", "matched_name", "amount_grams", "nutrition", "confidence"):
            assert key in result

    def test_picks_highest_score_candidate(self):
        low_c = {**FAKE_CANDIDATE, "doc_id": "low", "item_name": "banana chips", "adjusted_score": 0.50}
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [low_c, FAKE_CANDIDATE]
        )
        assert result["matched_name"] == "banana raw"

    def test_explicit_grams_used_as_amount(self):
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [FAKE_CANDIDATE]
        )
        assert result["amount_grams"] == 200

    def test_nutrition_scaled_to_explicit_grams(self):
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [FAKE_CANDIDATE]
        )
        assert result["nutrition"]["calories"] == pytest.approx(178.0, abs=0.2)

    def test_portion_hint_used_when_no_explicit_grams(self):
        result = rerank_single_item_heuristic(
            self._item("salmon", portion_hint={"grams": 150, "unit": "g"}),
            [FAKE_CANDIDATE],
        )
        assert result["amount_grams"] == 150

    def test_confidence_high_with_good_score_and_default(self):
        # banana is in portion_defaults.json; score=0.85 >= 0.75 → high
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [FAKE_CANDIDATE]
        )
        assert result["confidence"] == "high"

    def test_confidence_medium_with_moderate_score(self):
        med_c = {**FAKE_CANDIDATE, "adjusted_score": 0.60}
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [med_c]
        )
        assert result["confidence"] == "medium"

    def test_confidence_low_with_weak_score(self):
        result = rerank_single_item_heuristic(
            self._item("salmon", qty_raw="200g"), [FAKE_CANDIDATE_LOW]
        )
        assert result["confidence"] == "low"

    def test_description_penalty_selects_better_match(self):
        # pie_candidate score 0.90 penalised → 0.45; raw_candidate 0.85 unpenalised → 0.85
        pie_c = {**FAKE_CANDIDATE, "doc_id": "pie", "item_name": "peach pie", "adjusted_score": 0.90}
        raw_c = {**FAKE_CANDIDATE, "doc_id": "raw", "item_name": "banana raw", "adjusted_score": 0.85}
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g", description="raw fruit"),
            [pie_c, raw_c],
        )
        assert result["matched_name"] == "banana raw"

    def test_fallback_default_grams_used_when_no_explicit_qty(self):
        # banana default = 120g, no qty_raw → amount_grams = 120 * 1.0
        result = rerank_single_item_heuristic(
            self._item("banana"), [FAKE_CANDIDATE]
        )
        assert result["amount_grams"] == 120

    def test_fallback_multiplier_applied_to_default_grams(self):
        # banana default = 120g, quantity_raw="two" → 120 * 2 = 240g
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="two"), [FAKE_CANDIDATE]
        )
        assert result["amount_grams"] == 240

    def test_fallback_unknown_item_uses_100g_default(self):
        # food not in portion_defaults.json → falls back to hard-coded 100g
        result = rerank_single_item_heuristic(
            self._item("xyzunknownfood"), [FAKE_CANDIDATE]
        )
        assert result["amount_grams"] == 100

    def test_quantity_parsed_takes_priority_over_quantity_raw_in_fallback(self):
        # quantity_parsed is pre-computed; should override _parse_quantity(quantity_raw)
        item = {**self._item("banana", qty_raw="two"), "quantity_parsed": 3.0}
        result = rerank_single_item_heuristic(item, [FAKE_CANDIDATE])
        # 120 * 3.0 = 360, not 120 * 2.0
        assert result["amount_grams"] == 360

    def test_none_values_in_nutrition_per_100g_treated_as_zero(self):
        candidate_nulls = {
            **FAKE_CANDIDATE,
            "nutrition_per_100g": {"calories": None, "protein": None, "fat": None, "carbs": None},
        }
        result = rerank_single_item_heuristic(
            self._item("banana", qty_raw="200g"), [candidate_nulls]
        )
        assert result["nutrition"]["calories"] == 0.0
        assert result["nutrition"]["protein"] == 0.0

    def test_date_time_propagated_to_result(self):
        item = {**self._item("banana", qty_raw="200g"), "date_time": "2026-01-01"}
        result = rerank_single_item_heuristic(item, [FAKE_CANDIDATE])
        assert result["date_time"] == "2026-01-01"


class TestRerankAll:
    def _extraction(self, name, qty_raw=None, query=None):
        q = query or f"{name} (unspecified)"
        return {
            "items": {
                "item1": {"item_name": name, "quantity_raw": qty_raw, "description": "unspecified"}
            },
            "queries": [q],
        }

    def _retrieval(self, query, candidates):
        return {"items": [{"query": query, "candidates": candidates}]}

    def test_returns_dict_with_results_key(self):
        ext = self._extraction("banana", qty_raw="200g", query="banana (unspecified)")
        ret = self._retrieval("banana (unspecified)", [FAKE_CANDIDATE])
        output = rerank_all(ext, ret, use_llm=False)
        assert "results" in output
        assert isinstance(output["results"], list)

    def test_one_item_in_one_result_out(self):
        ext = self._extraction("banana", qty_raw="200g", query="banana (unspecified)")
        ret = self._retrieval("banana (unspecified)", [FAKE_CANDIDATE])
        output = rerank_all(ext, ret, use_llm=False)
        assert len(output["results"]) == 1

    def test_no_candidates_returns_unknown_entry(self):
        ext = self._extraction("mystery food", query="mystery food (unspecified)")
        ret = self._retrieval("mystery food (unspecified)", [])
        output = rerank_all(ext, ret, use_llm=False)
        result = output["results"][0]
        assert result["matched_name"] == "UNKNOWN"
        assert result["confidence"] == "low"

    def test_two_items_both_processed(self):
        extraction = {
            "items": {
                "item1": {"item_name": "banana", "quantity_raw": "200g", "description": "unspecified"},
                "item2": {"item_name": "egg", "quantity_raw": "100g", "description": "unspecified"},
            },
            "queries": ["banana (unspecified)", "egg (unspecified)"],
        }
        egg_candidate = {**FAKE_CANDIDATE, "doc_id": "egg1", "item_name": "egg boiled"}
        retrieval = {
            "items": [
                {"query": "banana (unspecified)", "candidates": [FAKE_CANDIDATE]},
                {"query": "egg (unspecified)", "candidates": [egg_candidate]},
            ]
        }
        output = rerank_all(extraction, retrieval, use_llm=False)
        assert len(output["results"]) == 2
        names = [r["item_name"] for r in output["results"]]
        assert "banana" in names
        assert "egg" in names
