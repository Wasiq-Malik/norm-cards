"""Stage 1: generate query strings aimed at norm-DEFINING literature (LLM).

Unlike claim-targeted retrieval, we want the papers that establish a subfield's
conventions: surveys, benchmark/dataset papers, canonical methods, plus the
entities the claim itself names. For forecast claims we add trend/progress
queries; for multi-subfield claims we add intersection queries.
"""

from typing import Dict, List

from .models import ClaimAnalysis
from .taxonomy import Taxonomy
from . import llm


def _dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = s.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


_PROMPT = """You generate search queries to find the papers that DEFINE THE
CONVENTIONS of an AI subfield (so we can later extract its "scientific norm":
standard models, datasets, benchmarks, metrics, thresholds, and experimental
protocol). We are NOT trying to verify the specific claim; we want the papers
that establish what experiments researchers in this subfield always run.

CLAIM:
{claim}

SUBFIELDS: {subfields}
TEMPORAL MODE: {temporal_mode}
ENTITIES: {entities}

Produce JSON mapping each bucket to a list of 2-5 short search queries (each 2-8
words). Include these buckets when relevant:
- "survey": surveys/reviews of the subfield
- "benchmark_dataset": standard datasets, benchmarks, evaluation metrics
- "canonical_method": seminal / state-of-the-art methods
- "sota_leaderboard": current SOTA / leaderboards
- "entity_anchored": queries built from the claim's named models/datasets/metrics
- "intersection": ONLY if multiple subfields - their intersection
- "trend": ONLY if temporal_mode is forecast - progress/scaling/trajectory over time
- "theory": ONLY if the claim turns on provable guarantees, bounds or formal
  constraints - standard assumptions, provable bounds

Return ONLY the JSON object {{bucket: [queries...]}}.
"""


def generate(claim: str, analysis: ClaimAnalysis, taxonomy: Taxonomy,
             model: str = None) -> Dict[str, List[str]]:
    """Generate norm-defining query buckets for a claim (LLM-driven)."""
    data = llm.complete_json(_PROMPT.format(
        claim=claim,
        subfields=", ".join(analysis.subfields) or "(unknown)",
        temporal_mode=analysis.temporal_mode,
        entities=analysis.entities,
    ), model=model)
    out = {}
    for bucket, qs in (data or {}).items():
        if isinstance(qs, list):
            cleaned = _dedup([str(q) for q in qs])
            if cleaned:
                out[bucket] = cleaned
    return out


def flatten(queries: Dict[str, List[str]]) -> List[tuple]:
    """[(query, bucket), ...] de-duplicated across buckets."""
    seen, out = set(), []
    for bucket, qs in queries.items():
        for q in qs:
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append((q, bucket))
    return out
