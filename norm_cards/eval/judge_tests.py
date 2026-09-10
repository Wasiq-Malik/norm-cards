"""A battery of adversarial tests for the JUDGE, not for the pipeline.

    python -m norm_cards.eval.judge_tests --source <proposals dir> --problems all

Every number this harness reports is the judge's opinion. Before scaling a claim
set it is worth knowing which of these the judge would notice:

  * a proposal that answers a DIFFERENT claim,
  * a proposal padded with fluent prose that establishes nothing,
  * the same experiment repeated three times,
  * an experiment removed,
  * every dataset and model renamed to an equivalent.

The self-test (`evaluate --self-test`) only shows the judge recognises a reference
as covering itself. That is necessary and very weak: a judge that returned
"covered" unconditionally would pass it. These tests are the ones it can fail.

TWO FAMILIES, OPPOSITE EXPECTATIONS
-----------------------------------
Invariance — the rewrite preserves what the set would establish, so the score must
NOT move. A drop means the judge is reading presentation.

Sensitivity — the rewrite destroys or reduces what the set establishes, so the
score MUST move, and downwards. No drop means the judge is rewarding the shape of
experimental writing rather than its content. This family is the one that catches
a judge inflating every arm, which is the failure that would quietly invalidate a
20-claim run.

`order` is the control for both. Reversing a list cannot change what a set
establishes — set-level judging is order-free by construction — so whatever drift
it produces is the judge's own run-to-run noise, and no other test can be read
below that floor.

`half_reference` is the only test with an exactly known answer: feed back half the
reference verbatim and recall must come out at that fraction. It calibrates the
scale itself rather than a difference on it.
"""

import argparse
import copy
import json
import os

from . import EVAL_ROOT, JUDGE_EFFORT, JUDGE_MODEL, load_claims
from . import evaluate as ev
from . import scoring
from .invariance import perturb_resources

# Fluent, on-topic-sounding, and establishes nothing about any claim. Appended to
# every proposed experiment by the `pad` test: if recall rises, length is being
# rewarded.
FILLER = (
    " As part of this experiment we will maintain a detailed experimental log, "
    "record software versions and hardware configuration, and archive all "
    "intermediate artifacts for reproducibility. Results will be presented in "
    "tables and figures with clear captions, and we will discuss the findings in "
    "the context of the broader literature, noting limitations and outlining "
    "promising directions for future work."
)


def _half(exps: list) -> list:
    """The first half of the reference, verbatim."""
    return list(exps[:max(1, len(exps) // 2)])


# name -> (family, builder(ref_exps, prop_exps, other_prop_exps) -> (ref, prop), what)
TESTS = {
    # ---- invariance: score must not move -------------------------------- #
    "order": ("invariance", lambda r, p, o: (r, list(reversed(p))),
              "reverse the proposal list; set-level judging is order-free, so this "
              "is pure judge noise and the floor for every other test"),
    "resource": ("invariance",
                 lambda r, p, o: (r, [perturb_resources(x) for x in p]),
                 "rename every dataset/model to an equivalent; where the claim does "
                 "not name a resource, a substitute must score the same"),
    "duplicate": ("invariance", lambda r, p, o: (r, p + p),
                  "repeat the whole proposal set; saying a thing twice establishes "
                  "nothing further, so recall must be unchanged"),

    # ---- sensitivity: score must fall ----------------------------------- #
    "cross_claim": ("sensitivity", lambda r, p, o: (r, o),
                    "swap in another claim's proposals — real, fluent, competent "
                    "experiments that answer the wrong question. Recall must "
                    "collapse; this is the false-positive test"),
    "pad": ("sensitivity", lambda r, p, o: (r, [x + FILLER for x in p]),
            "append fluent filler that establishes nothing; recall must not rise"),
    "drop_one": ("sensitivity", lambda r, p, o: (r, p[:-1] if len(p) > 1 else p),
                 "delete the last proposed experiment; recall must not rise"),
    "empty": ("sensitivity", lambda r, p, o: (r, ["We will investigate the claim "
                                                  "using appropriate methods."]),
              "replace the proposal with one contentless sentence; recall must be "
              "near zero"),

    # ---- calibration and direction -------------------------------------- #
    "half_reference": ("calibration", lambda r, p, o: (r, _half(r)),
                       "feed back the first half of the reference verbatim; recall "
                       "must land on that exact fraction"),
    "swap": ("direction", lambda r, p, o: (p, r),
             "score the proposal AS the reference against the reference AS the "
             "proposal. The authors' own experiments must substantially address what "
             "the proposal asks for — far above the cross-claim floor. NOT compared "
             "to the forward score: see the granularity note below"),
}


def _score(problem, gt, ref_exps, prop_exps, subclaims, out_dir, tag,
           model, effort, use_tools):
    g = copy.deepcopy(gt)
    g["experiments"] = list(ref_exps)
    g.pop("roles", None)                       # roles no longer line up after a rewrite
    arm = {"experiments": list(prop_exps), "subclaims": subclaims,
           "claim": problem["claim"], "proposer_model": tag}
    res = ev.evaluate_arm(problem, g, arm, 0, out_dir, tag, model=model, effort=effort,
                          progress=False, use_tools=use_tools)
    return scoring.aggregate_runs([res], n_proposed=len(prop_exps))["scores"]["recall"]


def verdicts(rows: list) -> list:
    """Turn raw recalls into PASS/FAIL against each test's expectation.

    The floor is the worst |Δ| the order control produced. Reading any other test
    below its own measurement noise is how a stress suite ends up certifying a
    judge it never actually challenged."""
    floor = max([abs(r["order"] - r["baseline"]) for r in rows
                 if r.get("order") is not None and r.get("baseline") is not None]
                or [0.0])
    floor = max(floor, 0.02)
    out = []
    for r in rows:
        b = r["baseline"]
        for name, (family, _, _) in TESTS.items():
            v = r.get(name)
            if v is None:
                continue
            d = v - b
            if name == "order":
                ok, why = True, f"defines the floor (|Δ|={abs(d):.3f})"
            elif family == "invariance":
                ok = abs(d) <= floor
                why = f"|Δ|={abs(d):.3f} vs floor {floor:.3f}"
            elif name == "empty":
                ok = v <= 0.15
                why = f"recall={v:.3f}, must be <= 0.15"
            elif name == "cross_claim":
                ok = v <= 0.30
                why = f"recall={v:.3f}, must be <= 0.30"
            elif family == "sensitivity":
                ok = d <= floor
                why = f"Δ={d:+.3f}, must not rise above floor {floor:.3f}"
            elif name == "half_reference":
                exp = r.get("half_reference_expected")
                ok = exp is not None and abs(v - exp) <= 0.17
                why = f"recall={v:.3f}, expected ~{exp:.3f}"
            elif name == "swap":
                # NOT `v >= b`. Coverage is not symmetric under granularity, and
                # these two sides are not at the same granularity: a reference item
                # runs ~650 chars, a proposed one ~2100, so each proposal bundles
                # several requirements. Asking whether 8 fine items cover 1 coarse
                # bundle is a harder bar than the reverse, and missing any part of a
                # bundle makes it `partial`. An earlier version of this gate assumed
                # the 8-item side was the richer one; it is richer in COUNT, not in
                # content, and the test duly "failed" on an artifact. What the swap
                # does establish is that the authors' own experiments address the
                # proposal far better than an unrelated claim's do — which is the
                # cross_claim contrast, and is the comparison kept here.
                ok = v >= 0.45
                why = (f"swapped={v:.3f} (forward={b:.3f}; not comparable — "
                       f"reference {r['n_reference']} items vs proposed "
                       f"{r['n_proposed']}, ~3x coarser)")
            else:
                ok, why = True, ""
            out.append({"problem_id": r["problem_id"], "arm": r["arm"], "test": name,
                        "family": family, "baseline": b, "value": v,
                        "delta": round(d, 3), "pass": bool(ok), "why": why})
    return out


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True,
                    help="dir of problem_<id>/scify_proposer.json")
    ap.add_argument("--problems", default="all")
    ap.add_argument("--arm", default="",
                    help="arm to stress; default = the first arm found per problem")
    ap.add_argument("--tests", default="all",
                    help="comma-separated subset of " + ",".join(TESTS))
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default=JUDGE_EFFORT)
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--out", default=os.path.join(EVAL_ROOT, "judge_tests.json"))
    args = ap.parse_args()

    names = (list(TESTS) if args.tests == "all"
             else [t.strip() for t in args.tests.split(",") if t.strip()])
    bad = [t for t in names if t not in TESTS]
    if bad:
        raise SystemExit(f"unknown test(s) {bad}; choose from {list(TESTS)}")

    claims = load_claims()
    ids = (sorted(claims) if args.problems == "all"
           else [p.strip() for p in args.problems.split(",") if p.strip()])
    ids = [p for p in ids if p in claims]

    # cross_claim needs a donor: another claim's proposals for the same arm.
    donor = {}
    for i, pid in enumerate(ids):
        other = ids[(i + 1) % len(ids)] if len(ids) > 1 else None
        donor[pid] = other

    rows = []
    for pid in ids:
        try:
            gt = ev.load_ground_truth(pid)
        except FileNotFoundError:
            print(f"[{pid}] no reference — skipping")
            continue
        available = ev.arms_available(args.source, pid)
        arm = args.arm or (available[0] if available else "")
        if arm not in available:
            print(f"[{pid}] no arm {arm!r} — have {available}")
            continue
        base = ev.load_arm(args.source, pid, arm)
        ref, prop, sub = list(gt["experiments"]), base["experiments"], base["subclaims"]
        other = ([] if not donor[pid]
                 else ev.load_arm(args.source, donor[pid], arm)["experiments"])
        out_dir = ev.judgment_dir(pid)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== {pid} / {arm} · {len(ref)} reference, {len(prop)} proposed ===")
        b = _score(claims[pid], gt, ref, prop, sub, out_dir, "_jt_baseline",
                   args.model, args.effort, args.tools)
        print(f"    {'baseline':<16} recall={b:.3f}")
        row = {"problem_id": pid, "arm": arm, "baseline": b,
               "n_reference": len(ref), "n_proposed": len(prop),
               "half_reference_expected": len(_half(ref)) / len(ref)}
        for name in names:
            family, build, _ = TESTS[name]
            if name == "cross_claim" and not other:
                print(f"    {name:<16} skipped — needs a second claim")
                continue
            r2, p2 = build(ref, prop, other)
            v = _score(claims[pid], gt, r2, p2, sub, out_dir, f"_jt_{name}",
                       args.model, args.effort, args.tools)
            row[name] = v
            print(f"    {name:<16} recall={v:.3f}  Δ={v - b:+.3f}   [{family}]")
        rows.append(row)

    if not rows:
        raise SystemExit("no rows produced")
    vs = verdicts(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"judge": {"model": args.model, "effort": args.effort,
                         "tools": bool(args.tools)},
               "rows": rows, "verdicts": vs}, open(args.out, "w"), indent=2)

    print(f"\n-> {args.out}\n")
    print(f"{'test':<16}{'family':<13}{'pass':<6}{'n':<4}detail")
    for name in names:
        got = [v for v in vs if v["test"] == name]
        if not got:
            continue
        npass = sum(1 for v in got if v["pass"])
        mark = "PASS" if npass == len(got) else f"FAIL"
        print(f"{name:<16}{got[0]['family']:<13}{mark:<6}{npass}/{len(got):<2}"
              + "; ".join(v["why"] for v in got[:3]))
    failed = [v for v in vs if not v["pass"]]
    print(f"\n{len(vs) - len(failed)}/{len(vs)} checks passed."
          + ("" if not failed else "  FAILURES: "
             + ", ".join(f"{v['problem_id']}/{v['test']}" for v in failed)))


if __name__ == "__main__":
    _main()
