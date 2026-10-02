"""Evaluation harness for the norm-card pipeline (see docs/eval_harness_design.md).

  reference.py   validates and renders the reference recipes, which are INPUTS:
                 authored by hand or transcribed from the claim's source paper.
  transcribe.py  drafts a reference FROM a source paper's own experiment section,
                 for a person to check and promote. Extraction from a source of
                 truth, not design — see its module docstring for why that is a
                 different operation from the one below that is refused.
  build_cards.py splits a batch gathering run into per-problem norm cards.
  propose.py     runs the proposer over a claim set, one arm per (model, condition):
                 nocard / retrieval / card.
  evaluate.py    the judge (JUDGE_MODEL, prompt JUDGE_PROMPT) scores an arm's
                 experiment set against the claim's reference experiments.
  judge_suite.py known-answer tests for the judge; run it before swapping judges.
  scoring.py     turns those judgments into numbers; no model ever emits a score.
  report.py      aggregates arms into a comparison table.
  calibrate.py   checks the judge against hand labels before you trust a run.

References are immutable fixtures: the harness never generates or edits one, and
fails loudly when one is missing. It deliberately has no way to produce one either
— a reference written by asking a strong model to design experiments for the claim
would not be an independent standard, it would just be another system's output, and
if that system were good enough to define correctness you would ship it as the
proposer rather than score against it.
"""

import json
import os

# Both are overridable so a second claim set can live beside the DARPA sprint one
# without its references, judgments, and report landing in the same directory:
#   NORM_CARDS_EVAL_ROOT=results/eval_icml2026 \
#   NORM_CARDS_CLAIMS=norm_cards/data/icml2026-claims-v1.jsonl \
#     python -m norm_cards.eval.reference --check
EVAL_ROOT = os.environ.get("NORM_CARDS_EVAL_ROOT",
                           os.path.join("results", "eval_v2"))
CLAIMS_FILE = os.environ.get(
    "NORM_CARDS_CLAIMS",
    os.path.join(os.path.dirname(__file__), "..", "data",
                 "sprint2-continuous-release-problems-v1.jsonl"))

# gpt-5.6-terra since 2026-09-28. On the known-answer judge suite (judge_suite.py) luna
# credited controls run outside the setting a claim names in 12 of 27 runs; terra never
# did. Controls are what norm cards supply, so that error would inflate exactly the
# effect under study. The judge must never grade its own model's proposals, which is why
# gpt-5.6-terra is not in the proposer lineup.
JUDGE_MODEL = "gpt-5.6-terra"
# v2: 241 words instead of 930, no research-tools section, and no contradiction between
# "judge each reference experiment" and "only the claim is binding". It scored the same as
# v1 on the judge suite. v1 stays in prompts.PROMPTS to reproduce older runs.
JUDGE_PROMPT = "v2"
JUDGE_EFFORT = "high"

# The reference-blind auditor (audit.py). Kept on the coverage judge's model for cost:
# gpt-6-astra scored better on the audit's known-answer suite (45/45 case-runs in band against
# 38/45, and a clean set at 1.00 where terra trims 11%) but is too expensive to run over 510
# arms. Note the cost of that choice: both roles ask what makes an experiment worthwhile, so a
# belief this model holds about that appears in both answers and cancels nowhere. A cheaper
# model from another family would break the coupling; none has been tested on the audit suite.
#
# astra is in any case NOT usable as the coverage judge: on judge_suite's subst_bound it credits
# controls run outside the setting a claim names in 12 of 27 runs, where terra never does.
AUDIT_MODEL = JUDGE_MODEL


def gt_dir(problem_id: str) -> str:
    return os.path.join(EVAL_ROOT, "ground_truth", f"problem_{problem_id}")


def gt_path(problem_id: str) -> str:
    return os.path.join(gt_dir(problem_id), "ground_truth.json")


def judgment_dir(problem_id: str) -> str:
    return os.path.join(EVAL_ROOT, "judgments", f"problem_{problem_id}")


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
