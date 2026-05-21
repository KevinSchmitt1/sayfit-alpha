"""
One-time script: upload the hardcoded system prompts to Langfuse Prompt Management.

Run once after setting up your Langfuse account:
    python setup_langfuse_prompts.py

After running, the prompts appear in the Langfuse UI under "Prompt Management"
and can be edited there without touching code. The pipeline fetches the latest
production version at runtime (with fallback to the hardcoded string if Langfuse
is unreachable).
"""

from langfuse import Langfuse
from step1_extraction.extractor import SYSTEM_PROMPT as EXTRACTION_PROMPT
from step3_reranker.reranker import SYSTEM_PROMPT as RERANKER_PROMPT


def main():
    lf = Langfuse()

    lf.create_prompt(
        name="sayfit-extraction",
        prompt=EXTRACTION_PROMPT,
        labels=["production"],
        config={
            "description": "Step 1 – extract structured food items from raw user text",
        },
    )
    print("✅ Uploaded: sayfit-extraction")

    lf.create_prompt(
        name="sayfit-reranker",
        prompt=RERANKER_PROMPT,
        labels=["production"],
        config={
            "description": "Step 3 – select best database candidate and estimate portion",
        },
    )
    print("✅ Uploaded: sayfit-reranker")

    print("\nDone. Open Langfuse → Prompt Management to view and edit both prompts.")


if __name__ == "__main__":
    main()
