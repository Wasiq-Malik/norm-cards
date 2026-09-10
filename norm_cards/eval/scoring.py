"""Turning the judge's coverage calls into a number, and combining repeated runs.

All arithmetic, no model calls: the judge returns one status per reference experiment
and everything here is computed from those. That split is the main defence against the
failure that made an earlier gpt-5.5 judge unusable — asked for holistic scores, it
quietly under-rated correct work.

One metric:

    recall — of the experiments the source team actually ran, how many would a team
             running the proposed set have effectively performed?

Earlier versions also scored utility, soundness, norm alignment and resource
grounding. They were dropped, not hidden: the question this harness exists to answer
is whether a proposed set establishes what the reference establishes, and the extra
columns made that harder to see rather than easier.
"""

from collections import Counter
from typing import Dict, List, Optional

COVER_SCORE = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
SUFFICIENCY_SCORE = {"sufficient": 1.0, "sufficient_with_gaps": 0.5, "insufficient": 0.0}


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def recall(coverage: List[Dict]) -> Optional[float]:
    """Fraction of the reference the proposed set covers; partial counts half."""
    return _mean([c.get("mean_status_score") if c.get("status") == "CONTESTED"
                  else COVER_SCORE.get(c.get("status")) for c in coverage])


def score_run(ev: Dict, n_proposed: int = 0) -> Dict:
    """Scores for one evaluator pass over one arm's experiment set."""
    cov = ev.get("reference_coverage") or []
    statuses = [c.get("status") for c in cov]
    contributing = {i for c in cov for i in (c.get("covered_by") or [])}
    return {
        "recall": recall(cov),
        "n_reference": len(cov),
        "n_proposed": n_proposed,
        "coverage_statuses": statuses,
        "n_covered": sum(1 for s in statuses if s == "covered"),
        "n_partial": sum(1 for s in statuses if s == "partial"),
        "n_missing": sum(1 for s in statuses if s == "missing"),
        # Derived rather than asked for: a proposed experiment that never appears in
        # any `covered_by` contributed to nothing the reference asked for. That is not
        # automatically a fault — it may be sound work the reference does not
        # contain — but it is worth surfacing.
        "unused_proposed": sorted(set(range(n_proposed)) - contributing),
        "decision_sufficiency": ev.get("decision_sufficiency"),
        "sufficiency_score": SUFFICIENCY_SCORE.get(ev.get("decision_sufficiency")),
    }


def _majority(values: List, tie: str = None):
    """Modal value if it has a strict majority of the runs; otherwise `tie`."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    top, n = Counter(values).most_common(1)[0]
    return top if n * 2 > len(values) else (tie if tie is not None else top)


def aggregate_runs(runs: List[Dict], n_proposed: int = 0) -> Dict:
    """Majority-vote k independent evaluator runs over the same experiment set.

    Calls without a majority become CONTESTED and are flagged for hand review rather
    than averaged into a value no run supports — though the mean is carried, so a
    contested item does not vanish from the aggregate.
    """
    per_ref: Dict[int, List[Dict]] = {}
    for r in runs:
        for c in r.get("reference_coverage") or []:
            per_ref.setdefault(c.get("ref_index"), []).append(c)

    coverage, contested = [], []
    for idx in sorted(x for x in per_ref if x is not None):
        entries = per_ref[idx]
        sts = [c.get("status") for c in entries]
        agreed = _majority(sts, tie="CONTESTED")
        if agreed == "CONTESTED":
            contested.append({"ref_index": idx, "votes": sts})
        # Union `covered_by` only over runs that voted with the majority; taking it
        # over every run produces records like "status: missing, covered_by: [0,1]",
        # contradicting the rule the validator enforces on each individual run.
        agreeing = [c for c in entries if c.get("status") == agreed] or (
            entries if agreed == "CONTESTED" else [])
        rep = max(agreeing or entries,
                  key=lambda c: len(c.get("reasoning_chain") or []))
        coverage.append({**rep, "status": agreed, "run_statuses": sts,
                         "mean_status_score": _mean([COVER_SCORE.get(s, 0.0)
                                                     for s in sts]),
                         "covered_by": sorted({i for c in agreeing
                                               for i in (c.get("covered_by") or [])})})

    suff = _majority([r.get("decision_sufficiency") for r in runs])
    # `missing` comes from one representative run rather than the union of all k: runs
    # phrase the same gap differently, so a union reads as near-duplicates that no code
    # can dedupe, while one run's list is coherent and in a single voice.
    rep_run = next((r for r in runs if r.get("decision_sufficiency") == suff), runs[0])
    agg = {
        "reference_coverage": coverage,
        "decision_sufficiency": suff,
        "sufficiency_reasoning": rep_run.get("sufficiency_reasoning", ""),
        "missing": rep_run.get("missing") or [],
        "missing_all_runs": sorted({m for r in runs for m in (r.get("missing") or [])}),
        "contested": contested,
    }
    scores = score_run(agg, n_proposed)
    scores["n_contested"] = len(contested)
    scores["judge_runs"] = len(runs)
    scores["run_agreement"] = _mean([1.0 if c.get("status") != "CONTESTED" else 0.0
                                     for c in coverage])
    agg["scores"] = scores
    return agg
