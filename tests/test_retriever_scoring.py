from step2_retrieval.retriever import (
    _extract_core_name,
    _compute_name_penalty,
    _build_query_variants,
    _safe_float,
)

def test_extract_core_name():
    assert _extract_core_name("banana") == "banana"
    assert _extract_core_name("banana (fruit)") == "banana"
    assert _extract_core_name("banana (raw)") == "banana"