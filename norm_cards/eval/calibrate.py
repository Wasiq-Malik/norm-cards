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
"""

import argparse
import json
import os

from . import EVAL_ROOT
from .report import load_judgments

CAL_PATH = os.path.join(EVAL_ROOT, "calibration.json")
VERDICTS = ("ACCURATE", "MIXED", "IRRELEVANT")


def write_template(judgments, path: str = CAL_PATH):
    """Emit every judged experiment with a blank expected verdict to fill in.

    The judge's own verdict is deliberately NOT written into this file. Showing it
    would anchor the very human judgment the file exists to measure — you would be
    agreeing or disagreeing with the judge rather than labelling the experiment.
    calibrate.py reads the judge's verdicts from the judgments at scoring time.
    """
    items = []
    for pid in sorted(judgments, key=lambda p: int(p) if p.isdigit() else 0):
        for arm, j in judgments[pid].items():
            texts = j.get("experiments_evaluated") or []
            claim = j.get("claim", "")
            for e in sorted(j.get("experiments") or [], key=lambda x: x.get("index", 0)):
                i = e.get("index", 0)
                items.append({
                    "problem_id": pid, "arm": arm, "index": i,
                    "claim": claim,
                    "experiment": (texts[i] if i < len(texts) else ""),
                    "expected_verdict": "",      # <- fill in: ACCURATE|MIXED|IRRELEVANT
                    "note": ""})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"_instructions":
                   "Read the claim and the experiment, then fill expected_verdict with "
                   "your own judgment: ACCURATE (conveys the correct thing) / MIXED "
                   "(right idea, under-specified or missing a minor element) / "
                   "IRRELEVANT (would not yield useful information about the claim). "
                   "Leave items blank to skip them — label maybe 10, mixing obvious and "
                   "subtle cases. The judge's verdicts are intentionally not shown here; "
                   "run `python -m norm_cards.eval.calibrate` to compare.",
                   "items": items}, f, indent=2)
    print(f"wrote {path} ({len(items)} items, expected_verdict blank)\n"
          f"Fill in ~10 expected_verdict values, then run: "
          f"python -m norm_cards.eval.calibrate")


def score(labels, judgments):
    rows, agree = [], 0
    for lab in labels:
        want = (lab.get("expected_verdict") or "").strip().upper()
        if want not in VERDICTS:
            continue
        j = judgments.get(str(lab["problem_id"]), {}).get(lab["arm"])
        got = None
        if j:
            got = next((e.get("verdict") for e in j.get("experiments") or []
                        if e.get("index") == lab["index"]), None)
        ok = got == want
        agree += ok
        rows.append({**lab, "judge_verdict": got, "agrees": ok})
    return rows, agree


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
        print(f"no {args.file}; create one with --template, then fill in "
              f"expected_verdict")
        return

    with open(args.file, encoding="utf-8") as f:
        labels = json.load(f).get("items") or []
    rows, agree = score(labels, judgments)
    if not rows:
        print(f"{args.file} has no filled-in expected_verdict values")
        return

    print(f"\nJudge agreement: {agree}/{len(rows)} ({agree / len(rows):.0%})\n")
    for r in rows:
        mark = "ok  " if r["agrees"] else "MISS"
        print(f"  [{mark}] claim {r['problem_id']}/{r['arm']} exp {r['index']}: "
              f"hand={r['expected_verdict']:10s} judge={r['judge_verdict']}"
              + (f"   — {r['note']}" if r.get("note") else ""))
    if agree / len(rows) < 0.8:
        print("\nBelow the 0.8 gate: inspect the misses in the run traces before "
              "reporting any numbers from this judge.")


if __name__ == "__main__":
    _main()
