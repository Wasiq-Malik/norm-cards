"""Check the judge against hand labels before trusting a full evaluation run.

    python -m norm_cards.eval.calibrate --template   # writes a fill-in-the-blank file
    python -m norm_cards.eval.calibrate              # scores the judge against it

The reason this exists: the previous gpt-5.5 auto-judge under-scored correct recipes,
and nothing in the harness would have caught that on its own. So before a set of
numbers goes anywhere, a human labels a handful of experiments by hand and this
reports agreement. Judgments must already exist (run evaluate.py first) — calibration
scores the judge you actually ran, not a fresh one.

Gate suggested in the design: don't trust a full run until agreement is >= 8/10 with
every miss explainable.

Two things get labelled, because the judge makes two different kinds of call. A
`coverage` item asks whether the proposed set covers one reference experiment — that
is what `recall` is built from, so it matters most. An `experiment` item asks whether
one proposed experiment is sound on its own terms. They fail differently: a judge can
be well calibrated on soundness and still systematically mark every reference entry
`partial`, which would leave recall unable to separate a good set from a mediocre one
while every per-experiment call looked reasonable.

Run this *after* `evaluate.py --self-test`, not instead of it. The self-test is free
and catches gross miscalibration — a judge that cannot see the reference as covering
itself will not be fixed by hand labels. This is for the subtler question of whether
its middle judgments match yours.
"""

import argparse
import json
import os

from . import EVAL_ROOT
from .report import load_judgments

CAL_PATH = os.path.join(EVAL_ROOT, "calibration.json")
COVER_STATUS = ("covered", "partial", "missing")



def write_template(judgments, path: str = CAL_PATH):
    """Emit every coverage call and proposed experiment with a blank label to fill in.

    The judge's own call is deliberately NOT written into this file. Showing it would
    anchor the very judgment the file exists to measure — you would be agreeing or
    disagreeing with the judge rather than labelling the thing itself. calibrate.py
    reads the judge's calls from the judgments at scoring time.
    """
    items = []
    for pid in sorted(judgments, key=lambda p: (0, int(p)) if p.isdigit() else (1, p)):
        for arm, j in judgments[pid].items():
            proposed = j.get("experiments_evaluated") or []
            ref = j.get("reference_experiments") or []
            claim = j.get("claim", "")
            whole_set = "\n\n".join(f"--- PROPOSED {i} ---\n{t}"
                                     for i, t in enumerate(proposed))
            for c in j.get("reference_coverage") or []:
                i = c.get("ref_index", 0)
                items.append({
                    "kind": "coverage", "problem_id": pid, "arm": arm, "ref_index": i,
                    "claim": claim,
                    "reference_experiment": (ref[i] if i < len(ref) else ""),
                    "the_whole_proposed_set": whole_set,
                    "expected_status": "",            # covered|partial|missing
                    "note": ""})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"_instructions":
                   "Two kinds of item. For kind=coverage, read the claim, the ONE "
                   "reference experiment, and the WHOLE proposed set, then ask: would a "
                   "team running the proposed set learn what this reference experiment "
                   "would have told them? Fill expected_status: covered / partial "
                   "(they would get at it but the outcome stays ambiguous) / missing "
                   "(nothing bears on it). A different method that reaches the same "
                   "answer is covered. For kind=experiment, fill expected_soundness: "
                   "decisive (the claim's verdict depends on what it establishes) / "
                   "supporting (real but the claim could be decided without it) / "
                   "redundant (a sibling already establishes it) / tangential (sound "
                   "but not bearing on this claim). Leave items blank to skip them — label "
                   "maybe 10 of each, mixing obvious and subtle cases. The judge's "
                   "calls are intentionally not shown; run "
                   "`python -m norm_cards.eval.calibrate` to compare.",
                   "items": items}, f, indent=2)
    n_cov = sum(1 for i in items if i["kind"] == "coverage")
    print(f"wrote {path} ({n_cov} coverage calls, {len(items) - n_cov} experiments, "
          f"labels blank)\nFill in ~10 of each, then run: "
          f"python -m norm_cards.eval.calibrate")


def _judge_call(item, judgments):
    """What the judge actually said about this item."""
    j = judgments.get(str(item["problem_id"]), {}).get(item["arm"])
    if not j:
        return None
    return next((c.get("status") for c in j.get("reference_coverage") or []
                 if c.get("ref_index") == item.get("ref_index")), None)


def score(labels, judgments):
    """Agreement per kind, so a good soundness score cannot mask bad recall."""
    rows = {"coverage": []}
    for lab in labels:
        kind = lab.get("kind", "coverage")
        if kind != "coverage":
            continue
        field, allowed = "expected_status", COVER_STATUS
        want = (lab.get(field) or "").strip().lower()
        if want not in allowed:
            continue
        got = _judge_call(lab, judgments)
        rows[kind].append({**lab, "want": want, "judge": got, "agrees": got == want})
    return rows


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=EVAL_ROOT)
    ap.add_argument("--file", default=CAL_PATH)
    ap.add_argument("--template", action="store_true",
                    help="write a blank calibration file from existing judgments")
    args = ap.parse_args()

    judgments = load_judgments(args.root)
    if not judgments:
        print(f"no judgments under {args.root}/judgments — run evaluate.py first")
        return
    if args.template:
        write_template(judgments, args.file)
        return
    if not os.path.exists(args.file):
        print(f"no {args.file}; create one with --template, then fill in the "
              f"expected_status / expected_soundness fields")
        return

    with open(args.file, encoding="utf-8") as f:
        labels = json.load(f).get("items") or []
    rows = score(labels, judgments)
    if not any(rows.values()):
        print(f"{args.file} has no filled-in labels")
        return

    for kind in ("coverage",):
        rs = rows[kind]
        if not rs:
            continue
        agree = sum(r["agrees"] for r in rs)
        print(f"\n{kind.capitalize()} agreement: {agree}/{len(rs)} "
              f"({agree / len(rs):.0%})\n")
        for r in rs:
            mark = "ok  " if r["agrees"] else "MISS"
            what = f"ref {r.get('ref_index')}"
            print(f"  [{mark}] claim {r['problem_id']}/{r['arm']} {what}: "
                  f"hand={r['want']:12s} judge={r['judge']}"
                  + (f"   — {r['note']}" if r.get("note") else ""))
        if agree / len(rs) < 0.8:
            print(f"\nBelow the 0.8 gate on {kind}s: inspect the misses in the run "
                  f"traces before reporting any numbers from this judge.")

    for kind, metric in (("coverage", "recall"),):
        rs = rows[kind]
        if rs and len({r["judge"] for r in rs}) == 1 and len(rs) >= 5:
            print(f"\nNOTE: the judge gave every labelled {kind} call the same value "
                  f"({rs[0]['judge']!r}). Even at perfect agreement that makes "
                  f"{metric} a constant, so check the spread before comparing arms on "
                  f"it. This is exactly how the previous binary rubric failed — 27 of "
                  f"27 experiments landed in one category and the metric carried no "
                  f"information.")


if __name__ == "__main__":
    _main()
