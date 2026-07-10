"""Stages 2-4: run queries across sources, consolidate/dedup, and rank for
norm-representativeness.

Ranking here is intentionally simple and transparent for the scaffold:
  score = w_rel * subfield_relevance      (query/bucket coverage + fields match)
        + w_cite * citation_authority       (log, age-normalized)
        + w_type * paper_type_boost         (survey/benchmark favored)
        + w_rec  * recency
MMR-style diversity selection is left as a clearly marked TODO for P2.
"""

import math
import re
import time
from typing import Dict, List

from .models import Paper, merge_papers
from .sources.base import BaseSource

# Buckets whose hits indicate convention-defining papers get extra weight.
_NORM_BUCKETS = {"survey": 1.0, "benchmark_dataset": 0.8, "sota_leaderboard": 0.5,
                 "canonical_method": 0.4, "intersection": 0.4, "trend": 0.6,
                 "theory": 0.6, "entity_anchored": 0.3, "pwc_task": 0.5}

_SURVEY_RE = re.compile(r"\b(survey|a review|overview of|systematic review)\b", re.I)
_BENCH_RE = re.compile(r"\b(benchmark|dataset|evaluation|toolkit|leaderboard)\b", re.I)


def _dedup_by_version(papers: List[Paper]) -> List[Paper]:
    """Second merge pass: collapse preprint/published pairs that dedup_key()
    missed because both had a DOI but different ones (arXiv assigns its own
    10.48550/arxiv.* DOI distinct from the eventual venue DOI). Keyed on
    normalized title + first author — see Paper.version_key()."""
    pool: Dict[str, Paper] = {}
    passthrough: List[Paper] = []
    for p in papers:
        key = p.version_key()
        if not key:
            passthrough.append(p)
        elif key in pool:
            merge_papers(pool[key], p)
        else:
            pool[key] = p
    return list(pool.values()) + passthrough


def run_queries(sources: List[BaseSource], query_bucket_pairs, per_query: int,
                progress: bool = True, delay: float = 1.05,
                topic_ids: List[str] = None) -> List[Paper]:
    """Execute every (query, bucket) across every source; tag provenance; union.

    For each source, always runs its primary (semantic/lexical) search. If the
    source also exposes `search_lexical` (currently just OpenAlex) and
    `topic_ids` is given, additionally runs a topics.id-filtered lexical pass
    and unions the results in — this recovers survey/benchmark papers semantic
    search under-ranks, without the off-topic noise plain lexical search had
    (the topic filter, not query phrasing, keeps it on-topic). Validated
    experiment: this union beats semantic-only, lexical-only, and their
    intersection (near-empty) on survey/benchmark/recency coverage.

    `delay` paces requests under OpenAlex semantic search's ~1 req/sec limit
    (the shared http layer also retries on 429/503); the topic-filtered lexical
    pass isn't semantic-rate-limited so it's paced lighter.
    """
    pool: Dict[str, Paper] = {}
    calls = []
    for query, bucket in query_bucket_pairs:
        for src in sources:
            calls.append((query, bucket, src, "primary"))
            if topic_ids and hasattr(src, "search_lexical"):
                calls.append((query, bucket, src, "lexical_topic"))
    total = len(calls)
    for done, (query, bucket, src, mode) in enumerate(calls, 1):
        if progress:
            print(f"  [{done}/{total}] {src.name}:{mode} <- {bucket}: {query!r}")
        if delay:
            time.sleep(delay if mode == "primary" else min(delay, 0.3))
        if mode == "primary":
            papers = src.search_tagged(query, per_query)
        else:
            papers = src.search_lexical(query, per_query, topic_ids=topic_ids)
            for p in papers:
                if src.name not in p.sources:
                    p.sources.append(src.name)
                if query not in p.queries:
                    p.queries.append(query)
        for paper in papers:
            if bucket not in paper.buckets:
                paper.buckets.append(bucket)
            key = paper.dedup_key()
            if key in pool:
                merge_papers(pool[key], paper)
            else:
                pool[key] = paper
    return _dedup_by_version(list(pool.values()))


def select_balanced(papers: List[Paper], top_k: int, current_year: int = 2026,
                    recent_years: int = 4, recent_frac: float = 0.34) -> List[Paper]:
    """Pick top_k while guaranteeing recent papers aren't crowded out by highly
    cited classics. Assumes `papers` is already ranked (score desc)."""
    if len(papers) <= top_k:
        return papers
    n_recent = max(1, round(top_k * recent_frac))
    recent = [p for p in papers if p.year and p.year >= current_year - recent_years]
    chosen, seen = [], set()
    for p in recent[:n_recent]:
        chosen.append(p)
        seen.add(id(p))
    for p in papers:  # fill remainder by overall rank
        if len(chosen) >= top_k:
            break
        if id(p) not in seen:
            chosen.append(p)
            seen.add(id(p))
    chosen.sort(key=lambda x: x.score, reverse=True)
    return chosen


def detect_paper_type(p: Paper) -> str:
    text = f"{p.title} {p.abstract[:400]}"
    if _SURVEY_RE.search(text):
        return "survey"
    if _BENCH_RE.search(text):
        return "benchmark"
    return "method"


def _relevance(p: Paper, subfields: List[str]) -> float:
    # Coverage across distinct norm buckets + subfield name hits in title/abstract.
    bucket_score = sum(_NORM_BUCKETS.get(b, 0.2) for b in set(p.buckets))
    text = f"{p.title} {p.abstract}".lower()
    name_hits = sum(1 for sf in subfields if sf.lower() in text)
    fos_hits = sum(1 for sf in subfields
                   for f in p.fields_of_study if sf.lower() in f.lower())
    src_div = 0.3 * (len(set(p.sources)) - 1)  # corroborated across sources
    return bucket_score + 0.5 * name_hits + 0.3 * fos_hits + src_div


def _authority(p: Paper, current_year: int) -> float:
    cites = p.citations or 0
    base = math.log1p(cites)
    if p.year and p.year <= current_year:
        age = max(1, current_year - p.year + 1)
        # Reward citations accrued quickly; don't over-penalize seminal old work.
        base += math.log1p(cites / age)
    return base


def _recency(p: Paper, current_year: int) -> float:
    if not p.year:
        return 0.0
    return max(0.0, 1.0 - (current_year - p.year) / 15.0)


def rank(papers: List[Paper], subfields: List[str], current_year: int = 2026,
         weights=(1.0, 0.8, 0.8, 0.15, 0.4), topic_ids: List[str] = None) -> List[Paper]:
    w_rel, w_cite, w_type, w_rec, w_topic = weights
    topic_ids = set(topic_ids or [])
    # Normalize authority to 0..1 across the pool so weights are comparable.
    auths = [_authority(p, current_year) for p in papers]
    amax = max(auths) if auths else 1.0
    amax = amax or 1.0
    for p, a in zip(papers, auths):
        p.paper_type = detect_paper_type(p)
        type_boost = {"survey": 1.0, "benchmark": 0.8}.get(p.paper_type, 0.2)
        topic_overlap = len(set(p.topic_ids) & topic_ids)
        p.score = round(
            w_rel * _relevance(p, subfields)
            + w_cite * (a / amax)
            + w_type * type_boost
            + w_rec * _recency(p, current_year)
            + w_topic * topic_overlap,
            4,
        )
    papers.sort(key=lambda x: x.score, reverse=True)
    return papers
