import pytest
import llm_client


@pytest.fixture(autouse=True, scope="session")
def use_openai_backend():
    llm_client.configure(use_openai=True)
