"""
SayFit Alpha – LLM Client Factory
====================================
Single source of truth for LLM access.
Supports two backends — configured once at startup via configure():

  • Groq   (default) → cloud API, requires GROQ_API_KEY in .env
  • Ollama (--locllm) → local inference, no API key needed

Both backends use the OpenAI-compatible chat.completions API,
so Step 1 and Step 3 need no changes.

Usage (in main.py):
    import llm_client
    llm_client.configure(use_local=True)   # call once before pipeline

Usage (in step modules):
    from llm_client import get_client, extraction_model, reasoning_model
"""

from openai import OpenAI
import config

# ── Runtime state (set once via configure()) ─────────────────────────────────
_use_local: bool = False
_client: OpenAI | None = None


def configure(use_local: bool = False):
    """
    Select the LLM backend. Call this once at startup before any pipeline run.

    Parameters
    ----------
    use_local : bool
        True  → Ollama (http://localhost:11434)
        False → Groq cloud API
    """
    global _use_local, _client
    _use_local = use_local
    _client = None  # reset lazy singleton so next get_client() picks new backend

    backend = f"Ollama ({config.OLLAMA_MODEL})" if use_local else f"Groq ({config.EXTRACTION_MODEL})"
    print(f"   🤖 LLM backend: {backend}")


def get_client() -> OpenAI:
    """Return the active OpenAI-compatible client (lazy singleton)."""
    global _client
    if _client is None:
        if _use_local:
            _client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama",           # Ollama ignores the key, but openai lib requires one
            )
        else:
            _client = OpenAI(
                api_key=config.GROQ_API_KEY,
                base_url=config.GROQ_BASE_URL,
            )
    return _client


def extraction_model() -> str:
    """Model name to use for Step 1 (extraction)."""
    return config.OLLAMA_MODEL if _use_local else config.EXTRACTION_MODEL


def reasoning_model() -> str:
    """Model name to use for Step 3 (reranker)."""
    return config.OLLAMA_MODEL if _use_local else config.REASONING_MODEL


def is_local() -> bool:
    """True if currently using local Ollama backend."""
    return _use_local



"""

New file: llm_client.py

Single factory, configured once at startup
Both steps import get_client(), extraction_model(), reasoning_model() from it
No changes to step logic — both steps just switched from their own private _get_client() to the shared factory. The chat.completions.create() calls are identical either way since Ollama is OpenAI-compatible.

Usage:


# Groq (default, cloud)
python main.py --text "i ate a banana"

# Local Ollama (qwen2.5:7b)
python main.py --text "i ate a banana" --locllm

# Local + no LLM at all (heuristic fallback, fastest)
python main.py --text "i ate a banana" --no-llm
If you ever install a different Ollama model, you can override it without touching code:


OLLAMA_MODEL=qwen2.5:14b python main.py --text "..." --locllm


"""