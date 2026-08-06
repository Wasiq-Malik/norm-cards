"""Evaluation harness for the norm-card pipeline (see docs/eval_harness_design.md).

  reference.py   validates and renders the reference recipes, which are INPUTS:
                 authored by hand or transcribed from the claim's source paper.
  evaluate.py    gpt-5.6-luna scores pipeline output (baseline vs method arms)
                 against those references.

References are immutable fixtures: the harness never generates or edits one, and
fails loudly when one is missing. It deliberately has no way to produce one either
— a reference written by asking a strong model to design experiments for the claim
would not be an independent standard, it would just be another system's output, and
if that system were good enough to define correctness you would ship it as the
proposer rather than score against it.
"""

import json
import os

EVAL_ROOT = os.path.join("results", "eval_v2")
CLAIMS_FILE = os.path.join(os.path.dirname(__file__), "..", "data",
                           "sprint2-continuous-release-problems-v1.jsonl")

JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_EFFORT = "high"


def gt_dir(problem_id: str) -> str:
    return os.path.join(EVAL_ROOT, "ground_truth", f"problem_{problem_id}")


def gt_path(problem_id: str) -> str:
    return os.path.join(gt_dir(problem_id), "ground_truth.json")


def judgment_dir(problem_id: str) -> str:
    return os.path.join(EVAL_ROOT, "judgments", f"problem_{problem_id}")


def tool_cache_dir() -> str:
    return os.path.join(EVAL_ROOT, ".toolcache")


def load_claims(path: str = None) -> dict:
    """problem_id -> claim record, from the canonical sprint claims file."""
    path = path or os.path.normpath(CLAIMS_FILE)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[str(rec["problem_id"])] = rec
    return out


def load_ground_truth(problem_id: str) -> dict:
    """Load a reference-recipe fixture, or fail loudly. The harness never produces
    one: references are authored outside it and checked in."""
    path = gt_path(problem_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no reference recipe for problem {problem_id} at {path}. Author one "
            f"first — start from a skeleton with:\n"
            f"  python -m norm_cards.eval.reference --template {problem_id}\n"
            f"and see docs/reference_format.md for what each field means.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
