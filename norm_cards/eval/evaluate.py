"""Evaluate a pipeline's proposed experiments against the reference recipe.

    python -m norm_cards.eval.evaluate --problems 6,7,9,10,11,12,21,33,36,37

Scores one arm at a time, k times each (default 3 — luna is cheap, and majority
voting is what turns a single opinionated run into something reportable). Verdicts
that do not reach a majority are marked CONTESTED for hand review instead of being
averaged into a number no run actually supports.

The judge is BLIND to which arm it is scoring. It is never told whether a set came
from the baseline or the norm-card arm, or that another arm exists — otherwise it
has an obvious thumb to put on the scale.

Reference recipes are fixtures supplied to this harness — hand-written, or taken from
the source paper's own experiment section. Nothing here generates or edits one; a
missing reference is an error telling you to author it first (see reference.py). The
judge is shown each reference's provenance and is instructed to treat it as fallible
notes rather than an oracle, because a reference is only ever as good as its source.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import (EVAL_ROOT, JUDGE_EFFORT, JUDGE_MODEL, judgment_dir, load_claims,
               load_ground_truth, tool_cache_dir)
from . import agent_loop, prompts, reference, schemas, scoring, tools

DEFAULT_SOURCE = os.path.join("results", "norm_cards_hybrid")
ARM_KEYS = {"baseline": "baseline_experiments", "method": "method_experiments"}


def load_arm(source: str, problem_id: str, arm: str) -> dict:
    """Read one arm's experiment set out of a proposer run."""
    path = os.path.join(source, f"problem_{problem_id}", "scify_proposer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no proposer output at {path}")
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    key = ARM_KEYS.get(arm, arm)
    if key not in run:
        raise KeyError(f"{path} has no {key!r} (keys: {sorted(run)})")
    return {"experiments": run[key], "subclaims": run.get("subclaims") or [],
            "claim": run.get("claim", ""), "proposer_model": run.get("model", "")}


def _experiments_text(experiments: list) -> str:
    return "\n\n".join(f"--- EXPERIMENT {i} ---\n{e}" for i, e in enumerate(experiments))


def evaluate_arm(problem: dict, gt: dict, arm_data: dict, run_idx: int,
                 out_dir: str, arm: str, model: str = JUDGE_MODEL,
                 effort: str = JUDGE_EFFORT, max_steps: int = 200,
                 progress: bool = True) -> dict:
    """One independent evaluator pass over one arm's experiment set."""
    experiments = arm_data["experiments"]
    subclaims = arm_data["subclaims"]
    user = prompts.EVALUATION_USER.format(
        domain=problem.get("domain", "ai"), problem_id=problem["problem_id"],
        claim=problem["claim"],
        subclaims="\n".join(f"- {s}" for s in subclaims) or "(none provided)",
        provenance=reference.provenance_line(gt),
        reference=reference.to_markdown(gt),
        n=len(experiments), experiments=_experiments_text(experiments))

    ev = agent_loop.run_agent(
        system=prompts.EVALUATION_SYSTEM, user=user,
        submit_spec=schemas.SUBMIT_EVALUATION, model=model, reasoning_effort=effort,
        max_steps=max_steps,
        trace_path=os.path.join(out_dir, f"{arm}.run{run_idx}.trace.jsonl"),
        validate=lambda d: schemas.validate_evaluation(d, len(experiments)),
        progress=progress, tag=f"[{arm} r{run_idx}] ")
    ev["_run"] = run_idx
    return ev


def evaluate_problem_arm(problem: dict, source: str, arm: str, judge_runs: int = 3,
                         model: str = JUDGE_MODEL, effort: str = JUDGE_EFFORT,
                         max_steps: int = 200, workers: int = 3) -> dict:
    pid = str(problem["problem_id"])
    gt = load_ground_truth(pid)          # fixture; fails loudly if absent
    arm_data = load_arm(source, pid, arm)
    out_dir = judgment_dir(pid)
    os.makedirs(out_dir, exist_ok=True)

    def one(i):
        return evaluate_arm(problem, gt, arm_data, i, out_dir, arm, model=model,
                            effort=effort, max_steps=max_steps,
                            progress=(workers == 1))

    if workers > 1 and judge_runs > 1:
        with ThreadPoolExecutor(max_workers=min(workers, judge_runs)) as ex:
            runs = list(ex.map(one, range(judge_runs)))
    else:
        runs = [one(i) for i in range(judge_runs)]

    agg = scoring.aggregate_runs(runs)
    return {
        "type": "evaluation", "format_version": "1.0",
        "problem_id": pid, "arm": arm, "claim": problem["claim"],
        "judge": {"model": model, "reasoning_effort": effort, "runs": judge_runs},
        "proposer_model": arm_data.get("proposer_model", ""),
        "n_experiments": len(arm_data["experiments"]),
        "experiments_evaluated": arm_data["experiments"],
        "scores": agg.pop("scores"),
        **agg,
        "gt_defects": scoring.gather_gt_defects(runs),
        "runs": [{"run": r.get("_run"), "meta": r.get("_meta"),
                  "evidence_ledger": r.get("_evidence"),
                  "experiments": r.get("experiments"),
                  "role_coverage": r.get("role_coverage"),
                  "decision_sufficiency": r.get("decision_sufficiency"),
                  "sufficiency_reasoning": r.get("sufficiency_reasoning"),
                  "missing": r.get("missing")} for r in runs],
    }


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problems", required=True,
                    help="comma-separated problem ids, or 'all' for every id with a "
                         "ground truth")
    ap.add_argument("--arms", default="baseline,method")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="dir of problem_<id>/scify_proposer.json to score")
    ap.add_argument("--judge-runs", type=int, default=3)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--effort", default=JUDGE_EFFORT,
                    choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel judge runs (1 = sequential, verbose progress)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tools.set_cache_dir(tool_cache_dir())
    claims = load_claims()
    if args.problems == "all":
        gt_root = os.path.join(EVAL_ROOT, "ground_truth")
        ids = sorted(d.replace("problem_", "") for d in os.listdir(gt_root)
                     if d.startswith("problem_")) if os.path.isdir(gt_root) else []
    else:
        ids = [p.strip() for p in args.problems.split(",") if p.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for pid in ids:
        if pid not in claims:
            print(f"[{pid}] not in the claims file — skipping")
            continue
        for arm in arms:
            out_path = os.path.join(judgment_dir(pid), f"{arm}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[{pid}/{arm}] judged already; --force to redo")
                continue
            print(f"\n=== problem {pid} · {arm} · {args.judge_runs} judge runs "
                  f"({args.model} @ {args.effort}) ===")
            try:
                res = evaluate_problem_arm(
                    claims[pid], args.source, arm, judge_runs=args.judge_runs,
                    model=args.model, effort=args.effort, max_steps=args.max_steps,
                    workers=args.workers)
            except Exception as e:
                print(f"[{pid}/{arm}] FAILED: {type(e).__name__}: {e}")
                continue
            os.makedirs(judgment_dir(pid), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            s = res["scores"]
            print(f"    -> {out_path}")
            print(f"       verdicts={s['verdicts']} correctness={s['correctness']} "
                  f"coverage={s['coverage']} grounded={s['groundedness']} "
                  f"sufficiency={s['decision_sufficiency']} "
                  f"contested={s['n_contested']} gt_defects={len(res['gt_defects'])}")


if __name__ == "__main__":
    _main()
