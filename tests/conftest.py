from unittest.mock import patch
import pytest


FAKE_CANDIDATES = [[{"food_name": "banana", "score": 0.95, "food_id": "001"}]]


@pytest.fixture(autouse=True)
def mock_retriever():
    with patch("step2_retrieval.retriever.retrieve", return_value=FAKE_CANDIDATES):
        yield
