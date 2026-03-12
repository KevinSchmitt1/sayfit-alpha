 — 3 new boost values:

ONTOLOGY_BOOST_RANK1 = 1.30 (rank-1 category match)
ONTOLOGY_BOOST_RANK2 = 1.10 (rank-2 category match)
ONTOLOGY_BOOST_RANK3 = 1.03 (rank-3, minimal nudge)
extractor.py — Updated system prompt:

Lists all 18 allowed L1 categories explicitly so the LLM can't hallucinate category names
Adds rule 4: output "category_ranks": [...] per item — 1 entry for simple items ("banana" → ["fruits"]), up to 3 for composite dishes ("spaghetti bolognese" → ["prepared & frozen meals", "grains & pasta", "meat"])
Updated JSON schema to include the new field
ontology_filter.py — New role: passthrough + expander:

If category_ranks present on item → uses LLM's ranked list directly (source: "llm")
If absent (heuristic mode) → falls back to original rule-based classify_item_name() (source: "rules")
category_hints now includes ranked_l1: [...] alongside cat_l1/cat_l2
retriever.py — Tiered boost logic:

Reads ranked_l1 from hints; finds the best (lowest rank index) match for each candidate
Applies rank-appropriate boost: ×1.30 / ×1.10 / ×1.03
Backwards-compatible: still handles old flat cat_l1 hints if ranked_l1 is absent