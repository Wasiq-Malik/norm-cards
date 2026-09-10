"""Shared data structures for the paper-gathering pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import re


@dataclass
class Paper:
    """A normalized paper record merged across sources."""

    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    citations: int = 0
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    fields_of_study: List[str] = field(default_factory=list)
    # OpenAlex topic IDs assigned to this work (e.g. "T12026"), up to 3, returned
    # free with every result. Used to filter/score by real topic-ID overlap
    # against the claim's classified topics instead of claim-text injection.
    topic_ids: List[str] = field(default_factory=list)
    # Provenance: which sources returned this, and under which queries/buckets.
    sources: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    buckets: List[str] = field(default_factory=list)
    # Source-specific structured metadata (e.g. PwC tasks / introduced_benchmarks
    # / hf_datasets / methods) — valuable priors for downstream norm extraction.
    tags: Dict[str, Any] = field(default_factory=dict)
    # API-provided semantic relevance (OpenAlex search.semantic cosine score).
    semantic_score: float = 0.0
    # Populated downstream by the ranker.
    paper_type: str = ""  # survey | benchmark | method | application | ""
    score: float = 0.0

    def dedup_key(self) -> str:
        """Stable identity for cross-source dedup: DOI > arXiv id > norm title."""
        if self.doi:
            return "doi:" + self.doi.lower().strip()
        if self.arxiv_id:
            return "arxiv:" + self.arxiv_id.lower().strip()
        return "title:" + normalize_title(self.title)

    def version_key(self) -> str:
        """Looser identity for collapsing preprint/published *duplicate DOI*
        records dedup_key() misses. arXiv preprints get their own DOI
        (10.48550/arxiv.*) distinct from the eventual venue DOI, so two
        records of the same paper both have a DOI but different ones and
        dedup_key() never falls through to title. Normalized title + first
        author is precise enough to not collide across unrelated papers, but
        loose enough to catch this case; used as a second merge pass, never
        as the primary key (a title/author scrape can be missing or noisy)."""
        if not self.title or not self.authors:
            return ""
        return normalize_title(self.title) + "|" + self.authors[0].lower().strip()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy title matching."""
    t = (title or "").lower()
    # Some OpenAlex titles carry literal backslash-escape artifacts from a dirty
    # LaTeX/text scrape (e.g. a literal "\n" as two chars, not a real newline) —
    # strip the escape as a unit so it doesn't leave a stray letter token behind.
    t = re.sub(r"\\[nrt]", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def merge_papers(a: Paper, b: Paper) -> Paper:
    """Merge two records of the same paper, preferring richer fields."""
    a.title = a.title or b.title
    a.abstract = a.abstract if len(a.abstract) >= len(b.abstract) else b.abstract
    a.authors = a.authors or b.authors
    a.year = a.year or b.year
    a.venue = a.venue or b.venue
    a.citations = max(a.citations or 0, b.citations or 0)
    a.url = a.url or b.url
    a.pdf_url = a.pdf_url or b.pdf_url
    a.doi = a.doi or b.doi
    a.arxiv_id = a.arxiv_id or b.arxiv_id
    a.fields_of_study = sorted(set(a.fields_of_study) | set(b.fields_of_study))
    a.topic_ids = sorted(set(a.topic_ids) | set(b.topic_ids))
    a.sources = sorted(set(a.sources) | set(b.sources))
    a.queries = sorted(set(a.queries) | set(b.queries))
    a.buckets = sorted(set(a.buckets) | set(b.buckets))
    a.tags = {**b.tags, **a.tags}
    a.semantic_score = max(a.semantic_score, b.semantic_score)  # best query match
    return a


@dataclass
class ClaimAnalysis:
    """Output of the subfield classifier for one claim."""

    subfields: List[str] = field(default_factory=list)  # canonical task names, ranked
    new_subfields: List[str] = field(default_factory=list)  # proposed, not in taxonomy
    # OpenAlex topic IDs (e.g. "T12026") the claim maps to, ranked. A second,
    # finer-grained vocabulary (literature clusters) used to filter/score search
    # results by real topic overlap; see topics.py.
    openalex_topic_ids: List[str] = field(default_factory=list)
    openalex_topic_names: List[str] = field(default_factory=list)
    temporal_mode: str = "static"  # static | forecast
    entities: Dict[str, List[str]] = field(default_factory=dict)  # models/datasets/metrics/thresholds
    confidence: float = 0.0
    method: str = ""  # "llm" | "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
