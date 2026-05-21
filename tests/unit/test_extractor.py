import json
from unittest.mock import MagicMock, patch

import pytest

from step1_extraction.extractor import (
    _clean_segment,
    _parse_quantity,
    extract_items,
    extract_items_heuristic,
)


# ── Heuristic path ────────────────────────────────────────────────────────────
# These test the rule-based fallback (use_llm=False / extract_items_heuristic).
# Fast, no network. Kept for regression coverage of the fallback internals.


@pytest.mark.heuristic
class TestCleanSegment:
    def test_removes_i_had_filler(self):
        assert _clean_segment("i had a banana") == "a banana"

    def test_removes_i_ate_filler(self):
        result = _clean_segment("i ate some rice")
        assert "i ate" not in result

    def test_removes_maybe_filler(self):
        result = _clean_segment("maybe some yogurt")
        assert "maybe" not in result

    def test_plain_text_unchanged(self):
        assert _clean_segment("banana") == "banana"

    def test_normalizes_multiple_spaces(self):
        result = _clean_segment("i had  a  banana")
        assert "  " not in result

    def test_strips_trailing_comma(self):
        assert _clean_segment("banana,") == "banana"

    def test_strips_trailing_period(self):
        assert _clean_segment("banana.") == "banana"

    def test_multiword_no_filler(self):
        assert _clean_segment("chicken breast") == "chicken breast"

    def test_hesitation_fillers_pass_through(self):
        # "eh" and "i guess" are not in _FILLER_PHRASES — known gap in heuristic path.
        # On the LLM path the model corrects these automatically.
        result = _clean_segment("chicken eh breast i guess")
        assert result == "chicken eh breast i guess"


@pytest.mark.heuristic
class TestParseQuantity:
    def test_gram_weight_prefix(self):
        qty, name = _parse_quantity("150g chicken")
        assert qty == "150g"
        assert name == "chicken"

    def test_word_quantity_two(self):
        qty, name = _parse_quantity("two eggs")
        assert qty == "two"
        assert name == "eggs"

    def test_article_a(self):
        qty, name = _parse_quantity("a banana")
        assert qty == "a"
        assert name == "banana"

    def test_half_prefix(self):
        qty, name = _parse_quantity("half avocado")
        assert qty == "half"
        assert name == "avocado"

    def test_no_quantity_multiword(self):
        qty, name = _parse_quantity("chicken breast")
        assert qty is None
        assert name == "chicken breast"

    def test_empty_string(self):
        qty, name = _parse_quantity("")
        assert qty is None

    def test_single_word_no_quantity(self):
        qty, name = _parse_quantity("salmon")
        assert qty is None
        assert name == "salmon"


@pytest.mark.heuristic
class TestExtractItemsHeuristic:
    def test_single_item_returns_one_item(self):
        result = extract_items_heuristic("salmon")
        assert len(result["items"]) == 1

    def test_two_items_from_and(self):
        result = extract_items_heuristic("banana and two eggs")
        assert len(result["items"]) == 2

    def test_two_items_from_comma(self):
        result = extract_items_heuristic("banana, yogurt")
        assert len(result["items"]) == 2

    def test_also_splits_items(self):
        result = extract_items_heuristic("banana also yogurt")
        assert len(result["items"]) == 2

    def test_filler_stripped_item_name(self):
        result = extract_items_heuristic("i had a banana")
        items = list(result["items"].values())
        assert items[0]["item_name"] == "banana"

    def test_quantity_captured(self):
        result = extract_items_heuristic("two eggs")
        items = list(result["items"].values())
        assert items[0]["quantity_raw"] == "two"
        assert items[0]["item_name"] == "eggs"

    def test_items_keyed_sequentially(self):
        result = extract_items_heuristic("banana and apple")
        assert "item1" in result["items"]
        assert "item2" in result["items"]

    def test_query_count_matches_item_count(self):
        result = extract_items_heuristic("banana and apple and yogurt")
        assert len(result["queries"]) == len(result["items"])

    def test_date_time_attached_to_items(self):
        result = extract_items_heuristic("banana", date_time="2026-01-01")
        items = list(result["items"].values())
        assert items[0]["date_time"] == "2026-01-01"

    def test_uid_attached_to_items(self):
        result = extract_items_heuristic("banana", uid="user42")
        items = list(result["items"].values())
        assert items[0]["uid"] == "user42"

    def test_description_is_unspecified(self):
        result = extract_items_heuristic("banana")
        items = list(result["items"].values())
        assert items[0]["description"] == "unspecified"

    def test_output_has_items_and_queries_keys(self):
        result = extract_items_heuristic("banana")
        assert "items" in result
        assert "queries" in result


# ── LLM path ──────────────────────────────────────────────────────────────────
# These test the production path (use_llm=True / extract_items default).
# The LLM client is always mocked — no real API calls.


def _make_mock_llm(response_payload: dict | str):
    """Return a patched llm_client whose completion returns response_payload."""
    if isinstance(response_payload, dict):
        content = json.dumps(response_payload)
    else:
        content = response_payload  # raw string, e.g. markdown-fenced or garbage

    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = content

    mock_llm = MagicMock()
    mock_llm.get_client.return_value.chat.completions.create.return_value = mock_completion
    mock_llm.extraction_model.return_value = "test-model"
    return mock_llm


_SINGLE_ITEM_RESPONSE = {
    "items": {
        "item1": {
            "item_name": "banana",
            "quantity_raw": None,
            "quantity_parsed": None,
            "unit_hint": None,
            "description": "raw fruit",
            "category_ranks": ["fruits"],
        }
    },
    "queries": ["banana (raw fruit)"],
}

_TWO_ITEM_RESPONSE = {
    "items": {
        "item1": {
            "item_name": "chicken breast",
            "quantity_raw": "150g",
            "quantity_parsed": 150,
            "unit_hint": None,
            "description": "grilled",
            "category_ranks": ["poultry"],
        },
        "item2": {
            "item_name": "rice",
            "quantity_raw": "a cup",
            "quantity_parsed": 1,
            "unit_hint": "cup",
            "description": "boiled",
            "category_ranks": ["grains & pasta"],
        },
    },
    "queries": ["chicken breast (grilled)", "rice (boiled)"],
}


@pytest.mark.llm_path
class TestExtractItems:
    def test_returns_items_and_queries_keys(self):
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(_SINGLE_ITEM_RESPONSE)):
            result = extract_items("i had a banana", use_llm=True)
        assert "items" in result
        assert "queries" in result

    def test_list_response_normalised_to_keyed_dict(self):
        # Some models return items as a JSON array instead of a keyed object.
        payload = {
            "items": [_SINGLE_ITEM_RESPONSE["items"]["item1"]],
            "queries": _SINGLE_ITEM_RESPONSE["queries"],
        }
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(payload)):
            result = extract_items("banana", use_llm=True)
        assert isinstance(result["items"], dict)
        assert "item1" in result["items"]

    def test_metadata_attached_to_each_item(self):
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(_SINGLE_ITEM_RESPONSE)):
            result = extract_items("banana", date_time="2026-01-01", uid="u1", use_llm=True)
        item = result["items"]["item1"]
        assert item["date_time"] == "2026-01-01"
        assert item["uid"] == "u1"

    def test_multi_item_response_preserves_all_items(self):
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(_TWO_ITEM_RESPONSE)):
            result = extract_items("150g chicken breast and a cup of rice", use_llm=True)
        assert len(result["items"]) == 2
        assert "item1" in result["items"]
        assert "item2" in result["items"]

    def test_multi_item_query_count_matches_item_count(self):
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(_TWO_ITEM_RESPONSE)):
            result = extract_items("chicken and rice", use_llm=True)
        assert len(result["queries"]) == len(result["items"])

    def test_markdown_fenced_json_is_parsed(self):
        # Some model configs return JSON wrapped in ```json ... ``` fences.
        fenced = "```json\n" + json.dumps(_SINGLE_ITEM_RESPONSE) + "\n```"
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(fenced)):
            result = extract_items("banana", use_llm=True)
        assert result["items"]["item1"]["item_name"] == "banana"

    def test_malformed_json_raises(self):
        # No error recovery exists — a bad LLM response surfaces as JSONDecodeError.
        # This test documents current behavior so any future recovery is an explicit decision.
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm("not json at all")):
            with pytest.raises(Exception):
                extract_items("banana", use_llm=True)

    def test_voice_noise_item_name_comes_from_llm(self):
        # "chicken eh breast i guess" is realistic voice input.
        # The LLM corrects it to "chicken breast" — the pipeline should use whatever
        # item_name the LLM returns, not the raw input text.
        clean_response = {
            "items": {
                "item1": {
                    "item_name": "chicken breast",
                    "quantity_raw": None,
                    "quantity_parsed": None,
                    "unit_hint": None,
                    "description": "unspecified",
                    "category_ranks": ["poultry"],
                }
            },
            "queries": ["chicken breast (unspecified)"],
        }
        with patch("step1_extraction.extractor.llm_client", _make_mock_llm(clean_response)):
            result = extract_items("chicken eh breast i guess", use_llm=True)
        assert result["items"]["item1"]["item_name"] == "chicken breast"

    def test_use_llm_false_delegates_to_heuristic(self):
        result = extract_items("banana and eggs", use_llm=False)
        assert "items" in result
        assert "queries" in result
        assert len(result["items"]) == 2
