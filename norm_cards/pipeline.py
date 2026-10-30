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
                         sources, per_query: int = 8, top_k: int = 30,
                         require_fulltext: bool = False,
                         cache_dir: str = ".pdfcache", screen=None) -> Dict:
    """Run the source/search/rank stages from a precomputed analysis + queries.

    `screen`, if given, takes the ranked candidates and returns the ones to keep (rank
    order kept); subfield_gather passes an LLM relevance screen. It runs before the
    full-text filter, so off-topic papers are neither fetched nor given a slot.

    `require_fulltext` drops every candidate whose full text cannot be fetched
    (paywalled, no open-access copy) BEFORE the top_k are chosen. Without it, a third
    of a bundle's slots went to papers the card could read only the abstract of —
    12/30 for Robotics — and abstracts carry almost none of the experimental detail a
    card is made of.

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
    n_ranked = len(papers)
    n_screened = None
    if screen is not None:
        papers = screen(papers)
        n_screened = len(papers)
        print(f"  relevance screen kept {n_screened}/{n_ranked} ranked candidates")
    if require_fulltext:
        papers = _with_fulltext(papers, cache_dir)
        print(f"  full text for {len(papers)}/{n_ranked} ranked candidates")
    top = collect.select_balanced(papers, top_k)  # guarantee recent + foundational mix

    return {
        "analysis": analysis.to_dict(),
        "queries": queries,
        "stats": {
            "n_queries": len(pairs),
            "n_candidates": n_ranked,
            "n_on_topic": n_screened,
            "n_with_fulltext": len(papers) if require_fulltext else None,
            "n_selected": len(top),
            "sources_used": [s.name for s in sources],
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "papers": [p.to_dict() for p in top],
    }


def _with_fulltext(papers, cache_dir: str, workers: int = 8):
    """The ranked candidates whose full text can be fetched, rank order kept. Fetched
    texts land in the same cache the card build and the rag arm read from."""
    from concurrent.futures import ThreadPoolExecutor
    from . import fulltext

    def ok(p):
        try:
            # what the card build reads (arXiv HTML first), so a paper whose PDF is
            # unreachable but whose HTML exists still counts as full text
            return bool(fulltext.fetch_body(p.to_dict(), cache_dir)[0])
        except Exception:
            return False
    with ThreadPoolExecutor(workers) as ex:
        keep = list(ex.map(ok, papers))
    return [p for p, k in zip(papers, keep) if k]


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
