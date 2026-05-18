from step1_extraction.extractor import extract_items


def test_extraction_returns_expected_shape():
    result = extract_items("had a banana and two eggs", use_llm=False)
    assert "items" in result
    assert "queries" in result
