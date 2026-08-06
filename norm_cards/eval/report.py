"""Aggregate judgments into results/eval_v2/REPORT.md.

    python -m norm_cards.eval.report

The point of this file is comparability across pipeline iterations: same reference
recipes, same judge, same scoring, so a change in the numbers is a change in the
pipeline. Run metadata is stamped into the report for exactly that reason — a
report produced with a different judge model or a different k is not comparable to
one that wasn't, and shouldn't look like it is.
"""

import argparse
import json
import os
from typing import Dict, List

from . import EVAL_ROOT, load_ground_truth
from . import reference

ARMS = ("baseline", "method")
METRICS = ("correctness", "coverage", "groundedness", "redundancy_penalty")


def load_judgments(root: str = EVAL_ROOT) -> Dict[str, Dict[str, Dict]]:
    """problem_id -> arm -> judgment."""
    out: Dict[str, Dict[str, Dict]] = {}
    jroot = os.path.join(root, "judgments")
    if not os.path.isdir(jroot):
        return out
    for d in sorted(os.listdir(jroot)):
        if not d.startswith("problem_"):
            continue
        pid = d.replace("problem_", "")
        for arm in ARMS:
            path = os.path.join(jroot, d, f"{arm}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    out.setdefault(pid, {})[arm] = json.load(f)
    return out


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _delta(a, b):
    """method − baseline, when both exist."""
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{d:+.2f}" if d else "0.00"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


SUFF_SHORT = {"sufficient": "suff", "sufficient_with_gaps": "gaps",
              "insufficient": "insuff", None: "—"}


def _provenance_section(pids: List[str]) -> List[str]:
    """Where each reference came from, and a warning if any is weakly sourced.

    A score is only as good as the reference it was measured against, so the report
    states that up front rather than leaving it to whoever remembers. This is also
    what keeps the caveat from being a hand-edited banner that the next
    `python -m norm_cards.eval.report` silently overwrites.
    """
    rows, suspect = [], []
    for pid in pids:
        try:
            line = reference.provenance_line(load_ground_truth(pid))
        except FileNotFoundError:
            line = "reference missing"
        rows.append(f"| {pid} | {line} |")
        if line.startswith("llm_generated") or line.startswith("provenance not stated") \
                or line == "reference missing":
            suspect.append(pid)

    L = ["## Reference provenance", "",
         "| Claim | Reference recipe came from |", "|---|---|"] + rows + [""]
    if suspect:
        L += [f"> **⚠️ These numbers are not a measurement of the pipeline.** The "
              f"reference recipes for claim(s) {', '.join(suspect)} are "
              f"machine-generated or unattributed. A reference written by asking a "
              f"model to design experiments for the claim is not an independent "
              f"standard — it is another system's output. Replace them with "
              f"hand-authored or paper-derived references (see "
              f"`docs/reference_format.md`) and re-run with `--force` before quoting "
              f"anything below.", ""]
    return L


def build(judgments: Dict[str, Dict[str, Dict]]) -> str:
    pids = sorted(judgments, key=lambda p: int(p) if p.isdigit() else 0)
    L = ["# Norm-card evaluation — baseline vs method", ""]

    any_j = next((j for p in judgments.values() for j in p.values()), None)
    if any_j:
        jd = any_j.get("judge", {})
        L += [f"Judge: `{jd.get('model')}` @ {jd.get('reasoning_effort')} effort, "
              f"k={jd.get('runs')} runs, majority vote. "
              f"Proposer: `{any_j.get('proposer_model', '?')}`. "
              f"Claims scored: {len(pids)}.", "",
              "Verdicts per experiment are ACCURATE=1.0 / MIXED=0.5 / IRRELEVANT=0.0; "
              "*correctness* is their mean. *Coverage* is weighted fill of the decision "
              "structure (gate .30, headline .40, apparatus .15, control .15), with "
              "roles the judge marked not-required dropped from the denominator. "
              "*Grounded* is the fraction of named resources that exist, are obtainable, "
              "and fit their use. *Redund.* is the fraction of experiments duplicating "
              "another in the same set (lower is better).", ""]

    L += _provenance_section(pids)

    # ---- per-claim table ---- #
    L += ["## Per-claim scores", "",
          "| Claim | Arm | n | Verdicts | Correct | Coverage | Grounded | Redund. | "
          "Sufficiency | Contested |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for pid in pids:
        for arm in ARMS:
            j = judgments[pid].get(arm)
            if not j:
                continue
            s = j["scores"]
            L.append(
                f"| {pid} | {arm} | {s.get('n_experiments')} | "
                f"{' '.join(v[:4] for v in s.get('verdicts') or [])} | "
                f"{_fmt(s.get('correctness'))} | {_fmt(s.get('coverage'))} | "
                f"{_fmt(s.get('groundedness'))} | {_fmt(s.get('redundancy_penalty'))} | "
                f"{SUFF_SHORT.get(s.get('decision_sufficiency'), '—')} | "
                f"{s.get('n_contested', 0)} |")
    L.append("")

    # ---- deltas ---- #
    both = [p for p in pids if "baseline" in judgments[p] and "method" in judgments[p]]
    if both:
        L += ["## Method − baseline", "",
              "The pipeline-improvement signal: positive means the norm card helped on "
              "that metric for that claim.", "",
              "| Claim | Δ correctness | Δ coverage | Δ grounded | Δ sufficiency |",
              "|---|---|---|---|---|"]
        for pid in both:
            b, m = judgments[pid]["baseline"]["scores"], judgments[pid]["method"]["scores"]
            L.append(f"| {pid} | {_delta(b.get('correctness'), m.get('correctness'))} | "
                     f"{_delta(b.get('coverage'), m.get('coverage'))} | "
                     f"{_delta(b.get('groundedness'), m.get('groundedness'))} | "
                     f"{_delta(b.get('sufficiency_score'), m.get('sufficiency_score'))} |")
        L += ["", "**Means across claims**", "",
              "| Metric | Baseline | Method | Δ |", "|---|---|---|---|"]
        for metric in METRICS + ("sufficiency_score",):
            bs = _mean([judgments[p]["baseline"]["scores"].get(metric) for p in both])
            ms = _mean([judgments[p]["method"]["scores"].get(metric) for p in both])
            L.append(f"| {metric} | {_fmt(bs)} | {_fmt(ms)} | {_delta(bs, ms)} |")
        L.append("")

    # ---- things a human has to look at ---- #
    contested: List[str] = []
    unsupported: List[str] = []
    for pid in pids:
        for arm, j in judgments[pid].items():
            for c in j.get("contested") or []:
                contested.append(f"- claim {pid} / {arm} / experiment {c['index']}: "
                                 f"runs voted {c['verdicts']}")
            for idx in j["scores"].get("unsupported_indices") or []:
                unsupported.append(f"- claim {pid} / {arm} / experiment {idx}: verdict "
                                   f"carries no evidence (tier T0)")
    if contested or unsupported:
        L += ["## Needs hand review", ""]
        if contested:
            L += ["**Contested** — judge runs did not reach a majority:", ""] + contested + [""]
        if unsupported:
            L += ["**Unsupported** — verdict not backed by evidence:", ""] + unsupported + [""]

    # ---- reference-recipe defects ---- #
    defects = [(pid, arm, d) for pid in pids for arm, j in judgments[pid].items()
               for d in (j.get("gt_defects") or [])]
    if defects:
        L += ["## Reference-recipe defects filed by the judge", "",
              "Input for the *manual* decision to regenerate a ground truth. The "
              "reference is a fixture; nothing here changes it automatically.", ""]
        by_pid: Dict[str, list] = {}
        for pid, arm, d in defects:
            by_pid.setdefault(pid, []).append((arm, d))
        for pid, items in by_pid.items():
            L.append(f"**Claim {pid}**")
            for arm, d in items:
                L.append(f"- `{d.get('gt_step', '?')}` ({arm}): {d.get('defect', '')}")
            L.append("")

    # ---- what each set was judged to be missing ---- #
    L += ["## Missing pieces, by claim", "",
          "What the judge said each set would still need to decide the claim.", ""]
    for pid in pids:
        for arm, j in judgments[pid].items():
            miss = j.get("missing") or []
            if miss:
                L.append(f"**{pid} / {arm}**")
                L += [f"- {m}" for m in miss]
                L.append("")
    return "\n".join(L)


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=EVAL_ROOT)
    ap.add_argument("--out", default=os.path.join(EVAL_ROOT, "REPORT.md"))
    args = ap.parse_args()

    judgments = load_judgments(args.root)
    if not judgments:
        print(f"no judgments under {args.root}/judgments — run "
              f"`python -m norm_cards.eval.evaluate` first")
        return
    md = build(judgments)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    n = sum(len(v) for v in judgments.values())
    print(f"wrote {args.out} ({len(judgments)} claims, {n} arm judgments)")


if __name__ == "__main__":
    _main()
