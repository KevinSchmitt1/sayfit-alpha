"""
Full end-to-end pipeline integration test.

Runs all 5 steps with real LLM and real FAISS index.
Local only — requires:
  - FAISS index built at data/faiss_index/  (python main.py --build-index)
  - GROQ_API_KEY (or OPENAI_API_KEY) in environment

Do NOT run in CI: FAISS index is not available there.
"""

import pytest
from main import run_pipeline


@pytest.mark.integration
def test_pipeline_returns_valid_structure():
    result = run_pipeline(
        text="i had a banana",
        date_time="2026-05-22T10:00:00",
        uid="test_user",
    )

    # Top-level shape
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0

    # Every item must have the required fields with correct types
    for item in result["results"]:
        assert "matched_name" in item
        assert isinstance(item["matched_name"], str)

        assert "amount_grams" in item
        assert isinstance(item["amount_grams"], (int, float))
        assert item["amount_grams"] > 0

        assert "nutrition" in item
        nutr = item["nutrition"]
        for key in ["calories", "protein", "fat", "carbs"]:
            assert key in nutr, f"Missing nutrition key: {key}"
            assert isinstance(nutr[key], (int, float))
            assert nutr[key] >= 0
