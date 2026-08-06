# Reference recipes

Inputs to the evaluation harness. One per claim, in `problem_<id>/ground_truth.json`,
rendered to `ground_truth.md` by `python -m norm_cards.eval.reference --check`.

Format and authoring instructions: [`docs/reference_format.md`](../../../docs/reference_format.md).

## ⚠️ The two references currently here are machine-generated and superseded

`problem_7` and `problem_11` were produced by the harness's removed Stage A —
`gpt-5.6-sol` at `xhigh` reasoning effort, researching each claim and designing its
experiment chain. That approach was dropped after review: **a reference written by
asking a strong model to design experiments for the claim is not an independent
standard.** It is one more system's output, and if that model were reliable enough to
define correctness, the right move would be to ship it as the proposer rather than to
score the proposer against it.

They are kept here only so that `results/eval_v2/REPORT.md` and the judgments under
`../judgments/` remain reproducible and inspectable. **Do not treat any number derived
from them as a measurement of the norm-card pipeline.** Both render with the provenance
line `llm_generated · gpt-5.6-sol @ xhigh effort — machine-written, unverified by a
human`, which is what the judge is shown, and `--check` warns on both.

Replace them with hand-authored or paper-derived references and re-run
`python -m norm_cards.eval.evaluate --problems 7,11 --force`.

## What was learned from them anyway

The judge filed **10 defects against these machine-written references** across the two
claims, including one case where it sided with a proposed experiment over the reference
on gate logic, and all 12 experiment verdicts were unanimous across 3 judge runs. The
"the reference is fallible, argue with it" machinery works — which is exactly why
swapping in a better-sourced reference is a drop-in change rather than a rewrite.
