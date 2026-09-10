# Reference format

A reference is **a list of experiments** — the ones a competent team actually ran to
decide the claim. Nothing else.

```json
{
  "type": "ground_truth",
  "format_version": "3.0",
  "problem_id": "icml61272",
  "domain": "ai",
  "claim": "…the claim text…",
  "provenance": {
    "kind": "paper_derived",
    "author": "…who transcribed or wrote this…",
    "date": "2026-08-21",
    "sources": ["Ren et al., …, ICML 2026 (arXiv:2606.00570v1)"],
    "notes": "Transcribed from Section 5 and Appendices B.5, C.1, C.3, C.5, C.6."
  },
  "experiments": [
    "Fix the measurement floor and the inference protocol before editing anything…",
    "Validate the scoring instrument before quoting any number from it…"
  ]
}
```

## Why it is this shape and not a richer one

`experiments` is a list of strings. That is the **same type** as `arms.<model>.experiments`
in a proposer run. The two sides are the same kind of object, and that is load-bearing
for two reasons.

**It keeps the comparison fair.** An earlier version of this format made the reference
a structured decision graph — per-step roles, dependencies, pass criteria, failure
meanings, quoted evidence, confidence, caveats, plus claim analysis, field norms,
decision logic and self-critique. It rendered to 130 lines against a proposal's 5, a
5× asymmetry, and the judge marked essentially every entry `partial`. That was the
predictable result rather than a bug in the judge: a paragraph cannot look complete
next to a specification, so every comparison found the paragraph wanting. The
proposals were not worse than the references; they were *shaped differently*, and the
metric was reading shape.

**It makes the judge testable.** Because the two sides are the same type they can be
swapped. `evaluate.py --self-test` feeds a reference in as though it were a proposal;
recall and precision must both come back at ceiling. A judge that cannot recognise
identical work as covered is under-crediting correct work, and no other number it
produces means anything. Richer references made that check impossible to run.

The rule that follows: **never let the reference grow past what the pipeline could
plausibly emit.** `--check` warns above 2000 chars per entry for exactly this reason.

## Writing an entry

One prose paragraph, at the granularity a proposer writes: what is run, on what data
and model, what is measured, and what result would decide the question. Include the
concrete threshold or the reported number so the decision rule is unambiguous.

> Fix the measurement floor and the inference protocol before editing anything. Score
> the unedited model on the editing benchmark to establish the pre-edit baseline, and
> re-run single-edit evaluation under plain autoregressive decoding rather than the
> teacher-forced protocol prior work used, since teacher forcing supplies the
> ground-truth answer tokens at inference and inflates every number. If methods keep
> their near-perfect single-edit scores under honest decoding there is no discrepancy
> to explain. The authors report a pre-edit baseline of Reliability 3.00,
> Generalization 3.00, Locality 15.50, Portability 4.36 on ZsRE, and a best
> parameter-based single-edit average of 25.99.

Do not write a structured specification, a checklist, or a rubric. Do not put roles,
dependencies or scoring hints *into the prose* — the judge assigns what it needs
functionally, from both sides, so anything added there appears on only one side of the
comparison and the proposal is marked down for missing scaffolding no proposer was
asked to produce.

### `roles` — analysis metadata, never shown to the judge

A reference may carry an optional `roles` array, one label per experiment, drawn from
`apparatus`, `headline`, `mechanism`, `control`, `confound`, `external`. This is the
one exception to the paragraph above, and it survives it because the judge never sees
it: `evaluate.py` builds its prompt from `experiments` alone, so both sides of the
comparison stay identical in shape and the swap test still works.

It exists because a recall number says how much was missed and cannot say *what kind*
of thing was missed, and those have different fixes. Pooled over the first three
claims, proposals covered apparatus and headline experiments at 0.83-0.88 and
confound-elimination experiments at 0.43 — a proposer that competently builds the rig
and runs the headline test, then does not think to kill the rival explanation. No
per-claim recall number showed that. `report.py` prints the breakdown under **Where
the misses are** whenever the labels are present.

Label by what the experiment is *for*, which is usually legible in its first sentence:
"Rule out that any perturbation of this size would do it" is a `control`; "Discriminate
the mechanistic account from the rival explanation that..." is a `confound`.

## Why it is not generated

The harness previously produced references by running a strong model over each claim.
That was dropped: a reference written by asking a model to design experiments is not
an independent standard, it is one more system's output. If that model were reliable
enough to define correctness, the right move would be to ship it as the proposer
rather than to score the proposer against it. So references come from outside — a
person, or the experiment section of the paper the claim was drawn from — and each one
states where it came from.

## How the judge uses it

The reference is a strong guide to **what matters**, and no guide at all to **how**.
The judge compares the two lists in both directions:

- For each reference experiment: would a team running the proposed set learn what this
  would have told them? `covered` / `partial` / `missing`. A different valid route to
  the same answer is `covered`. Mapping is many-to-many — one proposed experiment can
  cover three reference ones.
That is the whole of it. The judge used to classify each *proposed* experiment as well
(`matched` / `novel_sound` / `novel_unsound` / `off_claim`, plus soundness, norm
alignment, resource grounding and redundancy) and to report `reference_defects`. All of
it was removed: `matched` came back for 27 of 27 proposals, so precision was a constant
and the extra columns made the one question that matters harder to read rather than
easier.

## What `--check` enforces

**Errors** (block evaluation): `experiments` is a non-empty list of at least two
strings; each is at least 120 characters; a claim is present.

**Warnings** (never block): provenance `kind`/`author`/`sources` unfilled; any entry
past 2000 characters, since a reference much richer than a proposer's output biases
the comparison.
