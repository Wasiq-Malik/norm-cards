"""Known-answer tests for the audit stage.

    python -m norm_cards.eval.audit_suite --runs 3

The audit says what share of a proposed set is worth running. That number means nothing
until the auditor is shown to move when the set's quality moves, so each case here is a
set whose quality is known in advance, built from the same three domains judge_suite.py
uses. The expectation is stated as a BAND on the resulting precision, not a point: the
claim is "the auditor can tell these apart", not "it returns 0.81".

Cases, in the order they should separate:

  clean       the reference's own experiments, proposed verbatim. Everything bears on the
              claim by construction, so precision should be near 1.
  padded      clean plus three experiments on the same subject whose outcome cannot change
              what you believe about the claim. Precision should fall by roughly the share
              padded in, and the padding should be marked tangential rather than flawed.
  cross       another claim's competent proposals. Almost nothing should bear on this claim.
  vague       clean plus three experiments with no comparison and no decision rule. These
              should be marked flawed, which is a different failure from tangential.
  duplicated  clean with its first experiment repeated twice more, lightly reworded. The
              copies should be caught as duplicates, not counted as useful work.

A suite that only checked `clean` would certify an auditor that says yes to everything.
"""

import argparse
import collections
import json
import os
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from . import AUDIT_MODEL, audit, judge_suite, prompts, schemas  # noqa: F401

# Same three domains as the coverage suite, so a failure here is not a domain artefact.
PAD = {
    "adversarial": [
        "Survey which published adversarial-training methods report results on each dataset, "
        "and tabulate how often each is used in the literature.",
        "Measure wall-clock training time per epoch for each objective on one GPU and report "
        "the cost difference.",
        "Visualise the learned feature maps of the trained models for qualitative inspection.",
    ],
    "steering": [
        "Catalogue which open-weight models expose residual-stream hooks in their public "
        "implementations, and note the API differences.",
        "Measure the memory footprint of holding one steering vector per layer at each model "
        "size, and report the overhead.",
        "Produce t-SNE plots of the activation space for qualitative inspection.",
    ],
    "robot": [
        "Survey which manipulation benchmarks ship with demonstration data, and tabulate their "
        "licence terms.",
        "Measure the simulator's frame rate on the evaluation hardware and report throughput.",
        "Render example rollouts as videos for qualitative inspection.",
    ],
}
VAGUE = {
    "adversarial": [
        "Check that the method performs well on standard benchmarks.",
        "Confirm that robustness is reasonable under attack.",
        "Verify that the approach generalises appropriately.",
    ],
    "steering": [
        "Check that steering behaves sensibly across contexts.",
        "Confirm the intervention is reliable at an appropriate strength.",
        "Verify that model quality remains acceptable.",
    ],
    "robot": [
        "Check that the policy performs adequately on held-out tasks.",
        "Confirm that adaptation works reasonably well.",
        "Verify that the behaviour is robust enough in practice.",
    ],
}
# (name, how to build the proposed set, expected precision band, what it tests)
CASES = {
    "clean":      (lambda K, X: list(K["ref"]),            (0.85, 1.01),
                   "the reference's own experiments -> nearly all worth running"),
    "padded":     (lambda K, X: list(K["ref"]) + PAD[X],   (0.45, 0.80),
                   "three same-subject irrelevancies -> precision falls, marked tangential"),
    "cross":      (lambda K, X: list(judge_suite.KITS[judge_suite.CROSS[X]]["ref"]),
                   (0.00, 0.35), "another claim's competent proposals -> almost nothing bears"),
    "vague":      (lambda K, X: list(K["ref"]) + VAGUE[X], (0.45, 0.80),
                   "three undecidable experiments -> precision falls, marked flawed"),
    "duplicated": (lambda K, X: list(K["ref"]) + [
                       "Repeat the comparison described in experiment 0, with the same setup.",
                       "Run experiment 0 again under identical conditions and confirm the result."],
                   (0.45, 0.85), "the same work twice more -> caught as duplicates"),
}


def cases():
    out = []
    for kit_name, K in judge_suite.KITS.items():
        for case, (build, band, what) in CASES.items():
            out.append({"id": f"{kit_name}/{case}", "kit": kit_name, "case": case,
                        "claim": K["claims"]["can"], "proposed": build(K, kit_name),
                        "band": band, "what": what})
    return out


def run_case(c, model, run, out_dir):
    problem = {"problem_id": c["id"].replace("/", "__"), "claim": c["claim"], "domain": "ai"}
    try:
        res = audit.audit_arm(problem, {"experiments": c["proposed"]}, run, out_dir,
                              f"{model}-{problem['problem_id']}", model=model, progress=False)
        s = audit.score_audit(res["rows"], len(c["proposed"]))
        return {"scores": s, "error": None}
    except Exception as e:
        return {"scores": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}


def report(results):
    by = collections.defaultdict(list)
    for r in results:
        if r["scores"]:
            by[(r["case"], r["model"])].append(r)
    models = sorted({r["model"] for r in results})
    lines = ["", "  AUDIT PRECISION BY CASE (mean over kits x runs; band is what it should fall in)",
             f"  {'case':12s} {'band':>12s} " + " ".join(f"{m.replace('gpt-',''):>16s}" for m in models)]
    for case, (_, band, what) in CASES.items():
        row = f"  {case:12s} {f'{band[0]:.2f}-{band[1]:.2f}':>12s} "
        for m in models:
            rs = by.get((case, m), [])
            if not rs:
                row += f"{'—':>16s} "; continue
            v = st.fmean(r["scores"]["precision"] for r in rs)
            inside = band[0] <= v <= band[1]
            row += f"{f'{v:.2f}' + ('' if inside else ' OUT'):>16s} "
        lines.append(row)
    lines.append("")
    for m in models:
        ok = sum(1 for case, (_, band, _w) in CASES.items()
                 for r in by.get((case, m), []) if band[0] <= r["scores"]["precision"] <= band[1])
        tot = sum(len(by.get((case, m), [])) for case in CASES)
        lines.append(f"  {m}: {ok}/{tot} case-runs inside their band")
    # the two failure modes must not be confused with each other
    for m in models:
        pad = [r for r in by.get(("padded", m), [])]; vag = [r for r in by.get(("vague", m), [])]
        if pad and vag:
            pt = st.fmean(r["scores"]["tangential"] for r in pad)
            pf = st.fmean(r["scores"]["flawed"] for r in pad)
            vt = st.fmean(r["scores"]["tangential"] for r in vag)
            vf = st.fmean(r["scores"]["flawed"] for r in vag)
            lines.append(f"  {m}: padded -> tangential {pt:.2f} / flawed {pf:.2f};  "
                         f"vague -> tangential {vt:.2f} / flawed {vf:.2f}")
    dup = [r for r in results if r["case"] == "duplicated" and r["scores"]]
    if dup:
        lines.append(f"  duplicates caught: {st.fmean(r['scores']['duplicate'] for r in dup):.2f} "
                     f"of the set marked duplicate (2 of 5-7 experiments were copies)")
    errs = sum(1 for r in results if r["error"])
    if errs:
        lines.append(f"  errors: {errs}")
    return "\n".join(lines)


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=AUDIT_MODEL)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join("results", "audit_suite", "suite.json"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    cs = cases()
    if args.list:
        for case, (_, band, what) in CASES.items():
            print(f"  {case:12s} [{band[0]:.2f}-{band[1]:.2f}] {what}")
        print(f"\n  {len(cs)} cases = {len(judge_suite.KITS)} kits x {len(CASES)} behaviours")
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = os.path.join(os.path.dirname(args.out), "traces")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(c, m, r) for c in cs for m in models for r in range(args.runs)]
    print(f"  {len(cs)} cases x {args.runs} runs x {len(models)} model(s) = {len(jobs)} audit calls")
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda j: {"case": j[0]["case"], "kit": j[0]["kit"],
                                         "id": j[0]["id"], "model": j[1], "run": j[2],
                                         "band": j[0]["band"], **run_case(*j, out_dir)}, jobs))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(report(results))
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    _main()
