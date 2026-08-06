"""Turning an evaluator's judgments into numbers, and combining repeated runs.

Deliberately all arithmetic, no model calls. The judge emits small per-experiment
and per-role decisions; every score in the report is computed here from those. That
split is the main defence against the failure that made an earlier gpt-5.5 judge
unusable — it was asked for holistic scores and quietly under-rated correct work.
"""

from collections import Counter
from typing import Dict, List, Optional

VERDICT_SCORE = {"ACCURATE": 1.0, "MIXED": 0.5, "IRRELEVANT": 0.0}

# Coverage weights: the gate (cheapest refutation) and the headline (the actual
# test) are what decide a claim; apparatus and controls protect the result.
ROLE_WEIGHTS = {"gate": 0.30, "headline": 0.40, "apparatus": 0.15, "control": 0.15}
STATUS_CREDIT = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
SUFFICIENCY_SCORE = {"sufficient": 1.0, "sufficient_with_gaps": 0.5, "insufficient": 0.0}


def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 3) if xs else None


def role_coverage(rc: Dict) -> Optional[float]:
    """Weighted coverage of the decision structure. Roles the judge marked
    not_required drop out of the denominator rather than scoring zero — some
    claims genuinely need no apparatus."""
    num = den = 0.0
    for role, w in ROLE_WEIGHTS.items():
        status = (rc.get(role) or {}).get("status")
        if status in (None, "not_required"):
            continue
        num += w * STATUS_CREDIT.get(status, 0.0)
        den += w
    return round(num / den, 3) if den else None


def groundedness(experiments: List[Dict]) -> Optional[float]:
    """Fraction of named resources that survive the audit: exists, obtainable, and
    appropriate for the use the experiment makes of it."""
    ok = total = 0
    for e in experiments:
        for r in e.get("resource_audit") or []:
            total += 1
            if (r.get("exists") and r.get("publicly_available") is not False
                    and r.get("appropriate") is not False):
                ok += 1
    return round(ok / total, 3) if total else None


def score_run(ev: Dict) -> Dict:
    """Scores for a single evaluator run over one arm's experiment set."""
    exps = ev.get("experiments") or []
    scores = [VERDICT_SCORE.get(e.get("verdict"), 0.0) for e in exps]
    redundant = sum(1 for e in exps if e.get("redundant_with"))
    unsupported = [e.get("index") for e in exps if e.get("evidence_tier") == "T0"]
    return {
        "n_experiments": len(exps),
        "correctness": _mean(scores),
        "coverage": role_coverage(ev.get("role_coverage") or {}),
        "groundedness": groundedness(exps),
        "redundancy_penalty": round(redundant / len(exps), 3) if exps else None,
        "decision_sufficiency": ev.get("decision_sufficiency"),
        "sufficiency_score": SUFFICIENCY_SCORE.get(ev.get("decision_sufficiency")),
        "verdicts": [e.get("verdict") for e in sorted(exps, key=lambda x: x.get("index", 0))],
        "n_novel": sum(1 for e in exps if e.get("novel")),
        "unsupported_indices": unsupported,
    }


def _majority(values: List, tie: str = None):
    """Modal value if it has a strict majority of the runs; otherwise `tie`."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    top, n = Counter(values).most_common(1)[0]
    return top if n * 2 > len(values) else (tie if tie is not None else top)


def aggregate_runs(runs: List[Dict]) -> Dict:
    """Majority-vote k independent evaluator runs over the same experiment set.

    Verdicts that do not reach a majority become CONTESTED and are flagged for
    hand review rather than silently averaged into a number that no run supports —
    though the mean score is still carried so a contested item does not vanish
    from the aggregate.
    """
    per_index: Dict[int, List[Dict]] = {}
    for r in runs:
        for e in r.get("experiments") or []:
            per_index.setdefault(e.get("index"), []).append(e)

    experiments, contested = [], []
    for idx in sorted(per_index):
        entries = per_index[idx]
        verdicts = [e.get("verdict") for e in entries]
        agreed = _majority(verdicts, tie="CONTESTED")
        mean_score = _mean([VERDICT_SCORE.get(v, 0.0) for v in verdicts])
        if agreed == "CONTESTED":
            contested.append({"index": idx, "verdicts": verdicts})
        # Keep the fullest run's detail as the representative record; prefer one
        # that voted with the majority so the reasoning matches the verdict.
        pool = [e for e in entries if e.get("verdict") == agreed] or entries
        rep = max(pool, key=lambda e: len(e.get("evidence") or [])
                  + len(e.get("reasoning_chain") or []))
        experiments.append({**rep, "verdict": agreed, "run_verdicts": verdicts,
                            "mean_verdict_score": mean_score})

    rc: Dict[str, Dict] = {}
    for role in ROLE_WEIGHTS:
        statuses = [(r.get("role_coverage") or {}).get(role, {}).get("status")
                    for r in runs]
        notes = [(r.get("role_coverage") or {}).get(role, {}).get("note", "")
                 for r in runs]
        rc[role] = {"status": _majority(statuses), "run_statuses": statuses,
                    "note": notes[0] if notes else ""}

    suff = _majority([r.get("decision_sufficiency") for r in runs])
    # `missing` comes from one representative run rather than the union of all k.
    # Runs phrase the same gap differently, so a union reads as a wall of near
    # duplicates that no code can dedupe; one run's list is coherent and in a
    # single voice. The union is kept alongside for completeness.
    rep_run = next((r for r in runs if r.get("decision_sufficiency") == suff), runs[0])
    agg = {"experiments": experiments, "role_coverage": rc,
           "decision_sufficiency": suff,
           "sufficiency_reasoning": rep_run.get("sufficiency_reasoning", ""),
           "missing": rep_run.get("missing") or [],
           "missing_all_runs": sorted({m for r in runs for m in (r.get("missing") or [])}),
           "contested": contested}

    scores = score_run(agg)
    # Contested items have no agreed verdict, so score them by their run mean.
    if contested:
        pool = [e.get("mean_verdict_score") if e.get("verdict") == "CONTESTED"
                else VERDICT_SCORE.get(e.get("verdict"), 0.0) for e in experiments]
        scores["correctness"] = _mean([p for p in pool if p is not None])
    scores["n_contested"] = len(contested)
    scores["judge_runs"] = len(runs)
    scores["run_agreement"] = _mean(
        [1.0 if e.get("verdict") != "CONTESTED" else 0.0 for e in experiments])
    agg["scores"] = scores
    return agg


def gather_gt_defects(runs: List[Dict]) -> List[Dict]:
    """All reference-recipe defects the judge filed, deduped by (step, defect)."""
    out, seen = [], set()
    for r in runs:
        items = list(r.get("set_gt_defects") or [])
        for e in r.get("experiments") or []:
            items += list(e.get("gt_defects") or [])
        for d in items:
            key = (d.get("gt_step"), (d.get("defect") or "")[:80])
            if key not in seen:
                seen.add(key)
                out.append(d)
    return out
