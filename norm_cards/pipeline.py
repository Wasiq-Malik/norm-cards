"""Orchestrate the per-claim paper-gathering pipeline and emit a bundle.

claim -> classify -> queries -> multi-source search -> consolidate -> rank
      -> bundle {subfields, queries, ranked papers, pwc scaffold}
"""

import time
from typing import Dict

from . import classifier, query_gen, collect
from .models import ClaimAnalysis
from .taxonomy import Taxonomy


def gather_from_analysis(analysis: ClaimAnalysis, queries: Dict, taxonomy: Taxonomy,
                         sources, per_query: int = 8, top_k: int = 30) -> Dict:
    """Run the source/search/rank stages from a precomputed analysis + queries.

    For sources that support it (currently OpenAlex), unions each query's
    primary search with a topics.id-filtered lexical pass keyed on
    `analysis.openalex_topic_ids` — this recovers survey/benchmark papers
    semantic search under-ranks, without reintroducing the off-topic noise
    plain lexical search had (the topic filter, not claim-text injection,
    is what keeps results on-topic; validated experiment beat both
    semantic-only and semantic+claim-injected on survey/benchmark coverage).

    Used by the live pipeline (after the LLM stages) and by the keyless test
    harness (where the analysis + queries are supplied by hand).
    """
    t0 = time.time()
    pairs = query_gen.flatten(queries)
    print(f"  {len(pairs)} queries across {len(queries)} buckets, "
          f"{len(sources)} sources, topics={analysis.openalex_topic_ids}")

    papers = collect.run_queries(sources, pairs, per_query=per_query,
                                 topic_ids=analysis.openalex_topic_ids)
    # The taxonomy stores a keyword list per subfield; ranking needs it, because the
    # subfield name alone almost never appears verbatim in a paper's title or abstract.
    kws = sorted({k for sf in analysis.subfields
                  for k in ((taxonomy.get(sf) or {}).get("keywords") or [])})
    papers = collect.rank(papers, analysis.subfields,
                          topic_ids=analysis.openalex_topic_ids, keywords=kws)
    top = collect.select_balanced(papers, top_k)  # guarantee recent + foundational mix

    return {
        "analysis": analysis.to_dict(),
        "queries": queries,
        "stats": {
            "n_queries": len(pairs),
            "n_candidates": len(papers),
            "n_selected": len(top),
            "sources_used": [s.name for s in sources],
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "papers": [p.to_dict() for p in top],
    }


def gather_for_claim(claim: str, taxonomy: Taxonomy, sources,
                     model: str = None, per_query: int = 8, top_k: int = 30) -> Dict:
    analysis = classifier.classify(claim, taxonomy, model=model)
    print(f"  subfields={analysis.subfields} "
          f"mode={analysis.temporal_mode} openalex_topics={analysis.openalex_topic_names}")
    queries = query_gen.generate(claim, analysis, taxonomy, model=model)
    return gather_from_analysis(analysis, queries, taxonomy, sources,
                                per_query=per_query, top_k=top_k)


def build_bundle(input_record: Dict, taxonomy: Taxonomy, sources,
                 model: str = None, per_query: int = 8, top_k: int = 30) -> Dict:
    """Wrap a single dataset record (with problem_id/claim) into a bundle."""
    claim = input_record["claim"]
    result = gather_for_claim(claim, taxonomy, sources, model=model,
                              per_query=per_query, top_k=top_k)
    return {
        "type": "norm_card_paper_bundle",
        "format_version": "0.2",
        "problem_id": input_record.get("problem_id", ""),
        "problem_version": input_record.get("problem_version", "1.0"),
        "domain": input_record.get("domain", ""),
        "claim": claim,
        **result,
    }
