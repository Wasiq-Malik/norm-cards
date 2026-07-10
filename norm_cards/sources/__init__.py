"""Paper search sources.

Each source exposes a class with `.name` and `.search(query, limit) -> List[Paper]`.
Add a new API by dropping a module here and registering it below.

Keyless: OpenAlex, arXiv, Papers With Code. Semantic Scholar is keyless but
rate-limited (a free key raises limits). Google Scholar requires a paid SERP key.
"""

from typing import List

from .base import BaseSource
from .openalex import OpenAlexSource
from .arxiv_source import ArxivSource
from .semantic_scholar import SemanticScholarSource
from .serp_scholar import SerpScholarSource

# Query-search sources only. Papers With Code is NOT here — its free-text search
# is broken; it's used post-classification by task id (see sources/pwc.py).
SOURCE_REGISTRY = {
    "openalex": OpenAlexSource,
    "arxiv": ArxivSource,
    "semantic_scholar": SemanticScholarSource,
    "google_scholar": SerpScholarSource,
}

# Default: OpenAlex only — its semantic search is on-topic by construction.
# arXiv / Semantic Scholar are lexical (re-introduce off-topic noise) and are
# opt-in via --sources when broader recall is wanted.
DEFAULT_SOURCES = ["openalex"]


def get_sources(names: List[str] = None) -> List[BaseSource]:
    """Instantiate sources by name (default: the keyless set)."""
    names = names or DEFAULT_SOURCES
    out = []
    for n in names:
        cls = SOURCE_REGISTRY.get(n.lower())
        if cls is None:
            raise ValueError(f"Unknown source '{n}'. Known: {list(SOURCE_REGISTRY)}")
        out.append(cls())
    return out
