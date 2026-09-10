"""Stress-test the judge by changing things that must not change the score.

    python -m norm_cards.eval.invariance --source <dir> --problems all

The judge is supposed to score substance, not presentation. That is easy to assert in
a prompt and hard to believe without a measurement, so this rewrites each arm's
experiments in ways that preserve what they would establish, re-judges, and reports
the drift. A judge that scores substance returns the same recall.

Two perturbations, each targeting a failure this harness has actually seen:

  resource   swap named datasets/models/libraries for equivalents (Llama-3-8B ->
             Qwen2.5-7B-Instruct, ZsRE -> CounterFact). The claim is the contract:
             where the claim does not name a resource, any adequate substitute must
             score the same. A drop means the judge is matching the reference's
             choices rather than the claim's requirements.

  order      reverse the experiment list. Set-level judging is order-invariant by
             construction, so any drift here is noise in the judge, and its size is
             the floor below which the resource result cannot be read.

`order` is the control. Without it a resource drop is unattributable — you cannot tell
a real sensitivity from run-to-run variance.
"""

import argparse
import copy
import json
import os
import re

from . import EVAL_ROOT, JUDGE_EFFORT, JUDGE_MODEL, load_claims
from . import evaluate as ev

# Equivalent-for-purpose substitutions. Each pair must preserve the property that made
# the original adequate: an instruction-tuned model for an instruction-tuned model, a
# counterfactual editing benchmark for a counterfactual editing benchmark.
SUBSTITUTIONS = [
    (r"\bLlama[- ]?3(?:\.1)?[- ]?8B[- ]?Instruct\b", "Qwen2.5-7B-Instruct"),
    (r"\bLlama[- ]?2[- ]?7B[- ]?Chat\b", "Vicuna-7B-v1.5"),
    (r"\bLlama[- ]?3\b", "Qwen2.5"),
    (r"\bLlama[- ]?2\b", "Vicuna"),
    (r"\bMistral[- ]?7B[- ]?Instruct\b", "Llama-3.1-8B-Instruct"),
    (r"\bGPT-?J(?:-6B)?\b", "Pythia-6.9B"),
    (r"\bZsRE\b", "CounterFact"),
    (r"\bCounterFact\b", "ZsRE"),
    (r"\bMMLU\b", "ARC-Challenge"),
    (r"\bWikiText-?103\b", "The Pile (validation split)"),
    (r"\bBigCodeBench\b", "HumanEval+"),
    (r"\bEasyEdit\b", "a comparable open editing framework"),
    (r"\bHarmBench\b", "AdvBench"),
    (r"\bAdvBench\b", "HarmBench"),
    (r"\bAlpaca\b", "Dolly-15k"),
    (r"\bGCG\b", "AutoDAN"),
    (r"\bBM25\b", "Contriever"),
]


_COMBINED = re.compile("|".join(f"(?P<g{i}>{pat})"
                                for i, (pat, _) in enumerate(SUBSTITUTIONS)), re.I)


def perturb_resources(text: str) -> str:
    """Swap named resources for equivalents in ONE simultaneous pass.

    Applying the rules in sequence lets swaps chain and undo each other — with
    ZsRE->CounterFact followed by CounterFact->ZsRE, the second rule reverts the first
    and the text comes back unchanged, silently turning the test into a no-op. A single
    alternation over all patterns means every span is rewritten exactly once.
    """
    def pick(m):
        for i, (_, repl) in enumerate(SUBSTITUTIONS):
            if m.group(f"g{i}") is not None:
                return repl
        return m.group(0)
    return _COMBINED.sub(pick, text)


PERTURBATIONS = {
    "resource": lambda exps: [perturb_resources(e) for e in exps],
    "order": lambda exps: list(reversed(exps)),
}


def _score(problem, gt, exps, subclaims, out_dir, tag, model, effort, use_tools):
    arm_data = {"experiments": exps, "subclaims": subclaims, "claim": problem["claim"],
                "proposer_model": tag}
    res = ev.evaluate_arm(problem, gt, arm_data, 0, out_dir, tag, model=model,
                          effort=effort, progress=False, use_tools=use_tools)
    from . import scoring
    return scoring.aggregate_runs([res], n_proposed=len(exps))["scores"]


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True,
                    help="dir of problem_<id>/scify_proposer.json")
    ap.add_argument("--problems", default="all")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names; default = the first arm found")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default=JUDGE_EFFORT)
    ap.add_argument("--tools", action="store_true", help="give the judge its tool belt")
    ap.add_argument("--out", default=os.path.join(EVAL_ROOT, "invariance.json"))
    args = ap.parse_args()

    claims = load_claims()
    ids = (sorted(claims) if args.problems == "all"
           else [p.strip() for p in args.problems.split(",") if p.strip()])

    rows = []
    for pid in ids:
        if pid not in claims:
            continue
        try:
            gt = ev.load_ground_truth(pid)
        except FileNotFoundError:
            print(f"[{pid}] no reference — skipping")
            continue
        available = ev.arms_available(args.source, pid)
        arms = [a.strip() for a in args.arms.split(",") if a.strip()] or available[:1]
        out_dir = ev.judgment_dir(pid)
        os.makedirs(out_dir, exist_ok=True)

        for arm in arms:
            if arm not in available:
                print(f"[{pid}] no arm {arm!r} — have {available}")
                continue
            base = ev.load_arm(args.source, pid, arm)
            exps, sub = base["experiments"], base["subclaims"]
            print(f"\n=== {pid} / {arm} ===")
            base_s = _score(claims[pid], gt, exps, sub, out_dir,
                            "_inv_base", args.model, args.effort, args.tools)
            print(f"    baseline           recall={base_s['recall']}")
            row = {"problem_id": pid, "arm": arm, "baseline": base_s["recall"]}
            for name, fn in PERTURBATIONS.items():
                s = _score(claims[pid], gt, fn(copy.deepcopy(exps)), sub,
                           out_dir, f"_inv_{name}", args.model, args.effort, args.tools)
                d = (s["recall"] or 0) - (base_s["recall"] or 0)
                row[name] = s["recall"]; row[f"{name}_delta"] = round(d, 3)
                print(f"    {name:<18} recall={s['recall']}  Δ={d:+.3f}")
            rows.append(row)

    if rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"judge": {"model": args.model, "effort": args.effort,
                             "tools": bool(args.tools)}, "rows": rows},
                  open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}\n")
        for name in PERTURBATIONS:
            ds = [r[f"{name}_delta"] for r in rows if f"{name}_delta" in r]
            if ds:
                print(f"{name:<10} mean Δ {sum(ds)/len(ds):+.3f}   "
                      f"worst {min(ds):+.3f}   n={len(ds)}")
        od = [abs(r.get("order_delta", 0)) for r in rows]
        rd = [abs(r.get("resource_delta", 0)) for r in rows]
        if od and rd:
            floor = max(od)
            verdict = ("within the order-control floor — no resource sensitivity "
                       "detected" if max(rd) <= floor else
                       "ABOVE the order-control floor — the judge is scoring resource "
                       "identity, not substance")
            print(f"\norder control floor |Δ| = {floor:.3f}; "
                  f"worst resource |Δ| = {max(rd):.3f}\n=> {verdict}")


if __name__ == "__main__":
    _main()
