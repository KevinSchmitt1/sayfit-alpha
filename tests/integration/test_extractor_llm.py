"""
Integration test for step1 extraction.

Calls the real Groq API — runs in CI once GROQ_API_KEY is added as a secret.
Assertions are structural only: we check the shape, not the exact content,
because LLM output is non-deterministic.
"""

import pytest
from step1_extraction.extractor import extract_items


@pytest.mark.integration
def test_extract_items_structure():
    # A simple two-item sentence — the LLM should find at least one item
    result = extract_items("i had a banana and two eggs")

    # Top-level keys must exist
    assert "items" in result
    assert "queries" in result

    # items is a keyed dict (item1, item2, ...), not a list
    assert isinstance(result["items"], dict)
    assert len(result["items"]) > 0

    # queries is a non-empty list of strings
    assert isinstance(result["queries"], list)
    assert len(result["queries"]) > 0

    # Every item must have the required fields with the right types
    for key, item in result["items"].items():
        assert isinstance(item["item_name"], str)
        assert len(item["item_name"]) > 0
        assert isinstance(item["category_ranks"], list)
        assert len(item["category_ranks"]) > 0
        # quantity_parsed is either a number or None — never a string
        qp = item.get("quantity_parsed")
        assert qp is None or isinstance(qp, (int, float))
