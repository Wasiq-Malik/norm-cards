"""Score a pipeline's proposed experiments against the reference experiments.

    python -m norm_cards.eval.evaluate --problems all --source results/.../proposals

Both sides are lists of experiments, rendered identically, and the judge compares them
and the judge answers one question per reference experiment: would a team running the
proposed set have learned what it would have told them? The mean of those is `recall`.

Scores one arm at a time, k times each (default 3; majority voting is what turns a
single opinionated run into something reportable). Judgments without a
majority are marked CONTESTED for hand review instead of being averaged into a value
no run supports.

The judge is BLIND to which arm it is scoring. It is never told whether a set came
from the baseline or the norm-card arm, which model wrote it, or that another arm
exists — otherwise it has an obvious thumb to put on the scale.

    --self-test    scores each reference against ITSELF, as arm `_selftest`.

That mode exists because the two sides are the same kind of object. Recall must come
back at 1.0 — anything less is the judge failing to recognise identical work, and no
number it produces elsewhere means anything until that is fixed.
"""

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import (EVAL_ROOT, JUDGE_EFFORT, JUDGE_MODEL, JUDGE_PROMPT, judgment_dir,
               load_claims, load_ground_truth)
from . import agent_loop, prompts, reference, schemas, scoring

DEFAULT_SOURCE = os.path.join("results", "norm_cards_hybrid")
ARM_KEYS = {"baseline": "baseline_experiments", "method": "method_experiments"}
SELFTEST_ARM = "_selftest"


def arm_slug(arm: str) -> str:
    """Filename-safe arm name. Section-restricted card arms are named
    "card:datasets,models", and a colon is still a bad character to put in a
    filename on macOS — Finder renders it as a path separator."""
    return arm.replace(":", "-").replace(",", "_").replace("/", "-")


def load_arm(source: str, problem_id: str, arm: str) -> dict:
    """Read one arm's experiment set out of a proposer run.

    An arm is whatever is being contrasted: baseline vs method for the norm-card
    ablation, or one arm per model for a proposer comparison, held in an `arms` dict.
    """
    path = os.path.join(source, f"problem_{problem_id}", "scify_proposer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no proposer output at {path}")
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    arms = run.get("arms") or {}
    if arm in arms:
        exps, model = arms[arm]["experiments"], arms[arm].get("model", arm)
    else:
        key = ARM_KEYS.get(arm, arm)
        if key not in run:
            raise KeyError(f"{path} has no arm {arm!r} "
                           f"(arms: {sorted(arms) or sorted(run)})")
        exps, model = run[key], run.get("model", "")
    return {"experiments": exps, "subclaims": run.get("subclaims") or [],
            "claim": run.get("claim", ""), "proposer_model": model}


def arms_available(source: str, problem_id: str) -> list:
    """Arm names present in a proposer run, so callers need not hardcode them."""
    path = os.path.join(source, f"problem_{problem_id}", "scify_proposer.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    if run.get("arms"):
        return sorted(run["arms"])
    return [a for a, k in ARM_KEYS.items() if k in run]


def is_ablation(source: str, problem_id: str) -> bool:
    """Was this proposer run an ablation? Decides whether the judge sees the card."""
    path = os.path.join(source, f"problem_{problem_id}", "scify_proposer.json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return bool(json.load(f).get("ablation"))


def load_norm_card(source: str, problem_id: str) -> dict:
    """The norm card the pipeline built for this claim, if the run produced one."""
    path = os.path.join(source, f"problem_{problem_id}", "norm_card.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_arm(problem: dict, gt: dict, arm_data: dict, run_idx: int, out_dir: str,
                 arm: str, model: str = JUDGE_MODEL, effort: str = JUDGE_EFFORT,
                 max_steps: int = 200, progress: bool = True, system: str = None,
                 min_chain: int = 2) -> dict:
    """One independent evaluator pass over one arm's experiment set.

    `system` overrides the judge prompt (the suite runs v1 and v2 on the same cases);
    `min_chain` is how many reasoning steps a partial/missing verdict must carry — v2
    asks for one or two sentences, so it validates at 1."""
    proposed = arm_data["experiments"]
    ref_exps = reference.experiments(gt)
    user = prompts.EVALUATION_USER.format(
        domain=problem.get("domain", "ai"), problem_id=problem["problem_id"],
        claim=problem["claim"],
        subclaims="\n".join(f"- {s}" for s in arm_data["subclaims"]) or "(none provided)",
        provenance=reference.provenance_line(gt),
        n_ref=len(ref_exps),
        reference=reference.experiments_text(ref_exps, "REFERENCE EXPERIMENT"),
        n=len(proposed),
        experiments=reference.experiments_text(proposed, "PROPOSED EXPERIMENT"))

    ev = agent_loop.run_agent(
        system=system or prompts.EVALUATION_SYSTEM, user=user,
        submit_spec=schemas.SUBMIT_EVALUATION, model=model, reasoning_effort=effort,
        max_steps=max_steps,
        trace_path=os.path.join(out_dir, f"{arm_slug(arm)}.run{run_idx}.trace.jsonl"),
        validate=lambda d: schemas.validate_evaluation(d, len(proposed), len(ref_exps),
                                                       min_chain=min_chain),
        progress=progress, tag=f"[{arm} r{run_idx}] ")
    ev["_run"] = run_idx
    return ev


def evaluate_problem_arm(problem: dict, source: str, arm: str, judge_runs: int = 3,
                         model: str = JUDGE_MODEL, effort: str = JUDGE_EFFORT,
                         max_steps: int = 200, workers: int = 3,
                         prompt: str = None) -> dict:
    prompt = prompt or JUDGE_PROMPT
    pid = str(problem["problem_id"])
    gt = load_ground_truth(pid)          # fixture; fails loudly if absent

    if arm == SELFTEST_ARM:
        arm_data = {"experiments": reference.experiments(gt), "subclaims": [],
                    "claim": problem["claim"], "proposer_model": "(the reference itself)"}
    else:
        arm_data = load_arm(source, pid, arm)

    out_dir = judgment_dir(pid)
    os.makedirs(out_dir, exist_ok=True)

    def one(i):
        """One judge pass. A run that dies is reported and dropped, not raised.

        k independent runs exist precisely so no single one is load-bearing, and
        letting one exception discard the other two — as happened once, losing a whole
        arm to a crash in run 0 after runs 1 and 2 had finished clean — throws away
        good work and leaves a hole in the comparison. The traceback is printed so the
        failure is still diagnosable rather than silently swallowed.
        """
        try:
            return evaluate_arm(problem, gt, arm_data, i, out_dir, arm, model=model,
                                effort=effort, max_steps=max_steps,
                                progress=(workers == 1),
                                system=prompts.PROMPTS[prompt],
                                min_chain=1 if prompt == "v2" else 2)
        except Exception as e:
            print(f"    [{arm} r{i}] run FAILED, continuing without it: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    if workers > 1 and judge_runs > 1:
        with ThreadPoolExecutor(max_workers=min(workers, judge_runs)) as ex:
            runs = list(ex.map(one, range(judge_runs)))
    else:
        runs = [one(i) for i in range(judge_runs)]

    runs = [r for r in runs if r is not None]
    if not runs:
        raise RuntimeError(f"all {judge_runs} judge runs failed for {pid}/{arm}")
    if len(runs) < judge_runs:
        print(f"    [{arm}] aggregating {len(runs)}/{judge_runs} runs — majority vote "
              f"is weaker than usual for this arm")

    agg = scoring.aggregate_runs(runs, n_proposed=len(arm_data["experiments"]))
    return {
        "type": "evaluation", "format_version": "5.0",
        "problem_id": pid, "arm": arm, "claim": problem["claim"],
        "judge": {"model": model, "prompt": prompt, "reasoning_effort": effort,
                  "runs": len(runs), "runs_requested": judge_runs},
        "proposer_model": arm_data.get("proposer_model", ""),
        "n_proposed": len(arm_data["experiments"]),
        "n_reference": len(reference.experiments(gt)),
        "experiments_evaluated": arm_data["experiments"],
        "reference_experiments": reference.experiments(gt),
        "scores": agg.pop("scores"),
        **agg,
        "runs": [{"run": r.get("_run"), "meta": r.get("_meta"),
                  "reference_coverage": r.get("reference_coverage"),
                  "proposed": r.get("proposed"),
                  "decision_sufficiency": r.get("decision_sufficiency"),
                  "sufficiency_reasoning": r.get("sufficiency_reasoning"),
                  "missing": r.get("missing")} for r in runs],
    }


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problems", required=True,
                    help="comma-separated problem ids, or 'all' for every id with a "
                         "reference")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names; default = every arm in the source")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="dir of problem_<id>/scify_proposer.json to score")
    ap.add_argument("--self-test", action="store_true",
                    help="score each reference against itself; recall must "
                         "come back at 1.0")
    ap.add_argument("--judge-runs", type=int, default=3,
                    help="independent judge passes, majority-voted. 1 = a single pass, "
                         "much cheaper, no agreement signal and no CONTESTED flagging")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--prompt", default=JUDGE_PROMPT, choices=sorted(prompts.PROMPTS),
                    help="judge prompt version; v1 only to reproduce runs made before v2")
    ap.add_argument("--effort", default=JUDGE_EFFORT,
                    choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel judge runs (1 = sequential, verbose progress)")
    ap.add_argument("--show-pipeline-norms", action="store_true",
                    help="show the generated norm card to the judge even in an "
                         "ablation. Confounded — the card is the treatment there.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

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
        todo = ([SELFTEST_ARM] if args.self_test
                else (arms or arms_available(args.source, pid)))
        for arm in todo:
            out_path = os.path.join(judgment_dir(pid), f"{arm_slug(arm)}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[{pid}/{arm}] judged already; --force to redo")
                continue
            print(f"\n=== problem {pid} · {arm} · {args.judge_runs} judge runs "
                  f"({args.model} @ {args.effort}) ===")
            try:
                res = evaluate_problem_arm(
                    claims[pid], args.source, arm, judge_runs=args.judge_runs,
                    model=args.model, effort=args.effort, max_steps=args.max_steps,
                    workers=args.workers, prompt=args.prompt)
            except Exception as e:
                print(f"[{pid}/{arm}] FAILED: {type(e).__name__}: {e}")
                continue
            os.makedirs(judgment_dir(pid), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            s = res["scores"]
            print(f"    -> {out_path}")
            print(f"       recall={s['recall']}  (covered {s['n_covered']} / partial "
                  f"{s['n_partial']} / missing {s['n_missing']} of {s['n_reference']})"
                  f"  sufficiency={s['decision_sufficiency']}"
                  + (f"  unused={s['unused_proposed']}" if s['unused_proposed'] else "")
                  + (f"  contested={s['n_contested']}" if s['n_contested'] else ""))
            if arm == SELFTEST_ARM:
                if (s["recall"] or 0) < 0.95:
                    print(f"       ⚠️  SELF-TEST FAILED: the judge did not recognise "
                          f"the reference as covering itself. Fix this before trusting "
                          f"any other number from this judge.")
                else:
                    print("       self-test passed: the judge recognises the reference "
                          "as covering itself.")


if __name__ == "__main__":
    _main()
