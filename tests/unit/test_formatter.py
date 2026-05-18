import pytest

from step4_output.formatter import _pad, format_output, render_summary, render_table

HIGH_CONF_RESULT = {
    "item_name": "banana",
    "matched_name": "banana raw",
    "amount_grams": 120,
    "unit": "g",
    "quantity_raw": "1",
    "processing_description": "unspecified",
    "confidence": "high",
    "confidence_note": "",
    "nutrition": {"calories": 106.8, "protein": 1.3, "fat": 0.4, "carbs": 27.6},
}

LOW_CONF_RESULT = {
    "item_name": "mystery item",
    "matched_name": "UNKNOWN",
    "amount_grams": 100,
    "unit": "g",
    "quantity_raw": None,
    "processing_description": "unspecified",
    "confidence": "low",
    "confidence_note": "No candidates found.",
    "nutrition": {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0},
}


class TestPad:
    def test_left_align_pads_with_spaces(self):
        assert _pad("hi", 5) == "hi   "

    def test_right_align(self):
        assert _pad("hi", 5, ">") == "   hi"

    def test_center_align(self):
        result = _pad("hi", 6, "^")
        assert result == "  hi  "

    def test_truncates_when_text_too_long(self):
        assert _pad("toolong", 4) == "tool"

    def test_exact_width_unchanged(self):
        assert _pad("hello", 5) == "hello"


class TestRenderTable:
    def test_returns_string(self):
        assert isinstance(render_table([HIGH_CONF_RESULT]), str)

    def test_contains_total_row(self):
        table = render_table([HIGH_CONF_RESULT])
        assert "TOTAL" in table

    def test_contains_item_name(self):
        table = render_table([HIGH_CONF_RESULT])
        assert "banana" in table

    def test_single_item_total_equals_item_nutrition(self):
        table = render_table([HIGH_CONF_RESULT])
        # 106.8 kcal appears as both item row and TOTAL row
        assert "106.8" in table

    def test_two_items_totals_sum_correctly(self):
        second = {**HIGH_CONF_RESULT, "item_name": "egg", "nutrition": {"calories": 78.0, "protein": 6.0, "fat": 5.0, "carbs": 0.6}}
        table = render_table([HIGH_CONF_RESULT, second])
        # 106.8 + 78.0 = 184.8
        assert "184.8" in table

    def test_separator_rows_present(self):
        table = render_table([HIGH_CONF_RESULT])
        assert "+" in table and "-" in table

    def test_empty_results_renders_zero_total(self):
        table = render_table([])
        assert "0.0" in table

    def test_none_quantity_raw_renders_empty_string(self):
        result = {**HIGH_CONF_RESULT, "quantity_raw": None}
        table = render_table([result])
        assert isinstance(table, str)

    def test_none_values_in_nutrition_treated_as_zero(self):
        result = {**HIGH_CONF_RESULT, "nutrition": {"calories": None, "protein": None, "fat": None, "carbs": None}}
        table = render_table([result])
        assert "TOTAL" in table
        assert "0.0" in table


class TestRenderSummary:
    def test_returns_string(self):
        assert isinstance(render_summary([HIGH_CONF_RESULT]), str)

    def test_shows_meal_totals_header(self):
        summary = render_summary([HIGH_CONF_RESULT])
        assert "Meal Totals" in summary

    def test_shows_calories(self):
        summary = render_summary([HIGH_CONF_RESULT])
        assert "106.8" in summary

    def test_no_low_conf_section_when_all_high(self):
        summary = render_summary([HIGH_CONF_RESULT])
        assert "uncertain matching" not in summary

    def test_low_conf_item_flagged(self):
        summary = render_summary([LOW_CONF_RESULT])
        assert "uncertain matching" in summary
        assert "mystery item" in summary

    def test_medium_conf_item_also_flagged(self):
        med = {**HIGH_CONF_RESULT, "item_name": "egg", "confidence": "medium", "confidence_note": "Weak score."}
        summary = render_summary([med])
        assert "uncertain matching" in summary
        assert "egg" in summary

    def test_totals_summed_across_multiple_items(self):
        second = {**HIGH_CONF_RESULT, "nutrition": {"calories": 100.0, "protein": 5.0, "fat": 2.0, "carbs": 10.0}}
        summary = render_summary([HIGH_CONF_RESULT, second])
        # 106.8 + 100.0 = 206.8
        assert "206.8" in summary


class TestFormatOutput:
    def test_returns_string(self):
        output = format_output({"results": [HIGH_CONF_RESULT]})
        assert isinstance(output, str)

    def test_contains_sayfit_header(self):
        output = format_output({"results": [HIGH_CONF_RESULT]})
        assert "SayFit" in output

    def test_contains_table_and_summary(self):
        output = format_output({"results": [HIGH_CONF_RESULT]})
        assert "TOTAL" in output
        assert "Meal Totals" in output

    def test_empty_results_still_renders(self):
        output = format_output({"results": []})
        assert isinstance(output, str)
        assert "SayFit" in output
