"""norm_cards: subfield-aware scientific paper gathering for norm-card construction.

Pipeline: claim -> PwC subfield(s) + OpenAlex topic IDs + claim type -> query
buckets -> union of OpenAlex semantic search and topic-filtered lexical search
(plus optional extra sources) -> dedup/rank -> curated paper bundle (for
downstream norm-card extraction).

This package is additive to the existing claim-verification codebase and reuses
its conventions (litellm models, scimcp search style). The source/search/rank
stages are keyless; classification and query generation require an LLM key
(no heuristic fallbacks).
"""

__version__ = "0.2.0"
