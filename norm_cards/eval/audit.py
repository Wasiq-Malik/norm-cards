"""Audit each PROPOSED experiment against the claim, blind to the reference.

    python -m norm_cards.eval.audit --problems all --source results/.../proposals

`evaluate.py` answers "would this proposal have recovered what the source team did".
This answers a different question: "is each experiment in this proposal worth a slot".
They need separating because the first one's by-product, `unused_proposed`, is a bad
precision measure. Coverage is many-to-many, so a proposed experiment is only unused
when the judge links it to no reference item at all — on dataset34v5 that is 6% of
proposals, while 60% are linked to two or more. The resulting figure sits near 0.95
for every arm and distinguishes nothing.

THE AUDITOR NEVER SEES THE REFERENCE. That is the point. An auditor shown the
reference would mark whatever the reference happens to contain, which is overlap
again. Reference-blind, it can say that an experiment absent from the reference is
still worth running — which is the question a norm card's extra experiments raise.

One call per arm, with the whole set in view, so it can also mark redundancy within
the set. Scores are composed in code from the per-experiment verdicts; the model
never emits a rate.
"""

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import (AUDIT_MODEL, EVAL_ROOT, JUDGE_EFFORT, judgment_dir, load_claims)
from . import agent_loop, evaluate, prompts, reference, schemas


def audit_dir(problem_id: str) -> str:
    return os.path.join(EVAL_ROOT, "audits", f"problem_{problem_id}")


def score_audit(rows: list, n_proposed: int) -> dict:
    """Per-experiment verdicts -> rates. No model ever emits one of these."""
    bears = [r.get("bears") for r in rows]
    dup = [r.get("duplicate_of") is not None for r in rows]
    flawed = [r.get("quality") == "flawed" for r in rows]
    useful = [b in ("decides", "supports") for b in bears]
    # A slot is well spent when the experiment bears on the claim, would yield a
    # readable result, and is not repeating an earlier one in the same set.
    worth = [u and not f and not d for u, f, d in zip(useful, flawed, dup)]
    n = max(n_proposed, 1)
    return {
        "n_proposed": n_proposed,
        "precision": round(sum(worth) / n, 4),
        "bears_on_claim": round(sum(useful) / n, 4),
        "decides": round(sum(1 for b in bears if b == "decides") / n, 4),
        "supports": round(sum(1 for b in bears if b == "supports") / n, 4),
        "tangential": round(sum(1 for b in bears if b == "tangential") / n, 4),
        "irrelevant": round(sum(1 for b in bears if b == "irrelevant") / n, 4),
        "flawed": round(sum(flawed) / n, 4),
        "duplicate": round(sum(dup) / n, 4),
        "verdicts": [{"bears": b, "quality": r.get("quality"),
                      "duplicate_of": r.get("duplicate_of")} for b, r in zip(bears, rows)],
    }


def audit_arm(problem: dict, arm_data: dict, run_idx: int, out_dir: str, arm: str,
              model: str = AUDIT_MODEL, effort: str = JUDGE_EFFORT,
              max_steps: int = 40, progress: bool = True) -> dict:
    proposed = arm_data["experiments"]
    user = prompts.AUDIT_USER.format(
        claim=problem["claim"], n=len(proposed),
        experiments=reference.experiments_text(proposed, "PROPOSED EXPERIMENT"))
    au = agent_loop.run_agent(
        system=prompts.AUDIT_SYSTEM, user=user, submit_spec=schemas.SUBMIT_AUDIT,
        model=model, reasoning_effort=effort, max_steps=max_steps,
        trace_path=os.path.join(out_dir, f"{evaluate.arm_slug(arm)}.run{run_idx}.trace.jsonl"),
        validate=lambda d: schemas.validate_audit(d, len(proposed)),
        progress=progress, tag=f"[{arm} r{run_idx}] ")
    rows = sorted(au["proposed"], key=lambda r: r.get("index", 0))
    return {"rows": rows, "_meta": au.get("_meta")}


def audit_problem_arm(problem: dict, source: str, arm: str, model: str = AUDIT_MODEL,
                      effort: str = JUDGE_EFFORT, progress: bool = True) -> dict:
    pid = str(problem["problem_id"])
    arm_data = evaluate.load_arm(source, pid, arm)
    out_dir = audit_dir(pid)
    os.makedirs(out_dir, exist_ok=True)
    res = audit_arm(problem, arm_data, 0, out_dir, arm, model=model, effort=effort,
                    progress=progress)
    return {
        "type": "audit", "format_version": "1.0", "problem_id": pid, "arm": arm,
        "claim": problem["claim"],
        "auditor": {"model": model, "reasoning_effort": effort, "reference_blind": True},
        "proposer_model": arm_data.get("proposer_model", ""),
        "experiments_audited": arm_data["experiments"],
        "scores": score_audit(res["rows"], len(arm_data["experiments"])),
        "proposed": res["rows"], "meta": res["_meta"],
    }


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--problems", required=True,
                    help="comma-separated problem ids, or 'all'")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names; default = every arm in the source")
    ap.add_argument("--source", default=evaluate.DEFAULT_SOURCE)
    ap.add_argument("--model", default=AUDIT_MODEL)
    ap.add_argument("--effort", default=JUDGE_EFFORT,
                    choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--workers", type=int, default=1)
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
        todo = arms or evaluate.arms_available(args.source, pid)
        for arm in todo:
            out_path = os.path.join(audit_dir(pid), f"{evaluate.arm_slug(arm)}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"[{pid}/{arm}] audited already; --force to redo")
                continue
            try:
                res = audit_problem_arm(claims[pid], args.source, arm, model=args.model,
                                        effort=args.effort, progress=(args.workers == 1))
            except Exception as e:
                print(f"[{pid}/{arm}] FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
            os.makedirs(audit_dir(pid), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            s = res["scores"]
            print(f"[{pid}/{arm}] precision={s['precision']}  "
                  f"(decides {s['decides']} / supports {s['supports']} / "
                  f"tangential {s['tangential']} / irrelevant {s['irrelevant']}; "
                  f"flawed {s['flawed']}, duplicate {s['duplicate']})")


if __name__ == "__main__":
    _main()
