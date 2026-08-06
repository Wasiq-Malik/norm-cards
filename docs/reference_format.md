# Reference recipe format

A **reference recipe** is the experiment chain a competent team would actually run to
decide one claim. It is an **input** to the evaluation harness. The harness validates
it, renders it, shows it to the judge, and scores proposals against it — it never
writes one.

```
python -m norm_cards.eval.reference --template 7   # blank skeleton for problem 7
$EDITOR results/eval_v2/ground_truth/problem_7/ground_truth.json
python -m norm_cards.eval.reference --check        # validate + render markdown
python -m norm_cards.eval.evaluate --problems 7    # now scoreable
```

## Why this is not generated

The harness previously produced references by running `gpt-5.6-sol` at `xhigh` effort
over each claim. That was dropped: a reference written by asking a strong model to
design experiments is not an independent standard, it is one more system's output. If
that model were reliable enough to define correctness, the right move would be to ship
it as the proposer rather than to score the proposer against it. So references come
from outside — a person, or the experiment section of the paper the claim was drawn
from — and each one states where it came from.

Everything downstream is built on the assumption that a reference can be wrong. The
judge is shown the provenance line, is told the reference is fallible orientation
rather than an oracle, and is forbidden from faulting a proposed experiment on
"the reference does it differently" alone (`prompts.EVALUATION_SYSTEM`). It files
`gt_defects` against the reference when it finds one, and those surface in the report
for a human to act on.

## Where the files live

```
results/eval_v2/ground_truth/problem_<id>/
    ground_truth.json    the reference (this format)
    ground_truth.md      rendered by --check; what you actually read
```

## Fields

### Top level

| Field | Required | Meaning |
|---|---|---|
| `problem_id` | yes | matches the id in `norm_cards/data/sprint2-*.jsonl` |
| `claim` | yes | verbatim claim text |
| `domain` | no | e.g. `ai` |
| `provenance` | yes in practice | who wrote this and how — rendered into the line the judge sees |
| `claim_analysis` | yes | what is being asserted, and what the claim leaves open |
| `field_norms` | recommended | how this question is normally studied, with citations |
| `recipe` | yes | the experiment chain, in run order |
| `decision_logic` | yes | how step outcomes combine into VERIFY or REFUTE |
| `self_critique` | recommended | where this reference is weak |

### `provenance`

```json
{"kind": "paper_derived", "author": "W. Malik",
 "date": "2026-08-06",
 "sources": ["arXiv:2402.04249 HarmBench §4.2"],
 "notes": "steps G0-H3 transcribed from the paper's evaluation protocol"}
```

`kind` is free text; the useful values are `human`, `paper_derived`,
`expert_dictated`. A reference with no provenance renders as *"provenance not stated —
treat with corresponding suspicion"*, which is exactly how the judge will treat it.

### `claim_analysis`

- `assertions` — every quantity, threshold, named artifact, and scope condition the
  claim states, including the easily-skimmed ones ("person class only", "at 640×640",
  "at most 1% poisoned"). One per entry.
- `ambiguities` — what the claim leaves underspecified, **and the reading you adopt**.
  This is load-bearing: a proposal that adopts a different defensible reading should
  not be marked wrong for it, and the judge can only know that if you wrote it down.
- `claim_type` — `empirical` | `theoretical` | `mixed`.
- `named_entities` — `models` / `datasets` / `metrics` / `thresholds`.

### `field_norms`

Four lists — `datasets`, `models`, `metrics`, `protocols` — of

```json
{"name": "HarmBench", "why_standard": "the standard red-teaming eval for refusal rates",
 "citations": [{"source": "arXiv:2402.04249", "quote": "verbatim text from the paper"}]}
```

Quotes must be verbatim. This section is what lets the judge tell "unusual but
accepted in this field" from "invented", so it is worth the effort even though it is
not a hard requirement.

### `recipe`

The chain is **refute-first**: the cheapest thing that could kill the claim goes first,
and later steps assume the earlier ones did not already kill it. Aim for 4–6 steps
(3–8 enforced).

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | `G0`, `A1`, `H2`, `C3` — role letter + order |
| `role` | yes | `gate` \| `apparatus` \| `headline` \| `control` |
| `description` | yes | what is run, concretely enough to execute: data, model, condition, measurement |
| `pass_criteria` | yes | a decisive threshold tied to the claim's own numbers. Never "record the value" or "compare qualitatively" — state the comparison, the threshold, and what refutes |
| `why_necessary` | yes | why this step is required and why it is the standard way to test this. If the step carries no citation, this has to spell out the reasoning instead of asserting it |
| `fail_meaning` | no | what a failure here implies for the claim |
| `resources` | no | datasets, models, benchmarks by name |
| `depends_on` | no | ids of steps that must run first |
| `evidence` | no | `{source, quote}` backing this step |
| `confidence` | no | `high` \| `medium` \| `low` |
| `caveats` | no | your real doubts about this step |

**Roles.** These are also the axes of the coverage score
(`scoring.ROLE_WEIGHTS` — gate .30, headline .40, apparatus .15, control .15):

- **`gate`** — a cheap prerequisite whose failure refutes the claim outright. If a
  claim needs some metric ≥ X *under attack*, then clean performance ≥ X is a
  prerequisite, because an attack only degrades it: measure clean performance first,
  for near-zero compute.
- **`apparatus`** — something the claim presupposes that must be built **and
  validated** before the headline test: a trained generator, a fitted surrogate, a
  constructed poison set, an attack implementation. Validation needs its own pass
  criteria, or the headline test measures something else entirely.
- **`headline`** — the direct test, enforcing **every** constraint the claim states:
  norm bounds, budgets, rates, splits, resolutions, class subsets.
- **`control`** — checks that the result is not an artifact and is attributable to the
  claim's own mechanism: sensitivity sweeps, random/weak baselines, matched-budget
  comparisons, and an ablation removing the claimed mechanism. Anti-triviality controls
  belong here (a model that refuses everything also achieves a 100% violation
  reduction).

Mark a role `not_required` in your `self_critique` if the claim genuinely does not need
it; the judge can drop a role from the coverage denominator on the same grounds.

### `decision_logic`

Prose: how the steps' outcomes combine into VERIFY or REFUTE. Which failures are fatal,
which are partial, what a mixed result means.

### `self_critique`

Argue against your own reference before checking it in. What is missing? What is
over-specified in a way that would wrongly penalize a different but valid design? Which
thresholds did you choose rather than find? The judge reads this section, and a
weakness recorded here is one a proposal will not be unfairly punished for.

## What `--check` enforces

**Errors** (block evaluation) are structural only — the things that would make a
comparison meaningless:

- 3–8 recipe steps; at least one `headline` and one `gate`
- every step has an id, a valid role, a `pass_criteria` of real content, and a
  `why_necessary` that justifies it
- `depends_on` points at ids that exist
- `decision_logic` and `claim_analysis.assertions` are non-empty

**Warnings** (printed, never blocking) are about how much weight the reference can
bear: missing provenance, thin citations, uncited steps without spelled-out reasoning,
empty self-critique. A hand-written reference and a paper-transcribed one carry their
authority differently, and neither should be blocked over the shape of its
bibliography — but you should see the gap before you quote the numbers it produces.
