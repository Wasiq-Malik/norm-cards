# norm-cards

Does telling an LLM **how a research field runs experiments** help it design better
experiments?

A **norm card** is a structured summary of one subfield's experimental practice —
its standard datasets, models and metrics, and more importantly its *protocols,
controls and confounds*. We build one card per subfield, hand it to an experiment
proposer along with a scientific claim, and measure whether the proposals it
designs would have recovered what the claim's source paper actually did.

**Current result** (20 claims × 3 proposer models, September 2026):

| condition | recall |
|---|---|
| no card | 0.596 |
| retrieval over the same papers | 0.618 |
| **shared norm card** | **0.638** |

Card − no card is **+0.042**, 95% interval **[+0.008, +0.075]**, winning on 14 of
20 claims. Card − retrieval is +0.020 and does *not* exclude zero.

The cards are **shared, not per-claim**: 14 cards cover all 20 claims, each built
from its field's name alone with no claim text anywhere in the loop. That matters
because a card retrieved using the claim it will be tested on proves very little.

---

## Quickstart

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY
```

Reproduce the September 2026 run from the tracked dataset — no card construction,
no paper retrieval, just proposing and judging:

```bash
# 1. lay the dataset out into a disposable run directory
./.venv/bin/python -m norm_cards.dataset bootstrap --dataset dataset20v3 --run results/myrun
./.venv/bin/python -m norm_cards.dataset assemble  --dataset dataset20v3 --run results/myrun

export NORM_CARDS_EVAL_ROOT=results/myrun/eval
export NORM_CARDS_CLAIMS=norm_cards/data/dataset20-claims-v3.jsonl

# 2. propose — 20 claims × 3 arms × 3 models, ~45 min
./.venv/bin/python -m norm_cards.eval.propose \
    --models gpt-5.5,gpt-5.6-terra,gpt-5.4 --arms nocard,rag,card \
    --cards results/myrun/cards --budget 9 --workers 3 \
    --out results/myrun/proposals

# 3. judge — ~45 min if you shard, ~2.5h if you don't (see below)
./.venv/bin/python -m norm_cards.eval.evaluate --problems all \
    --source results/myrun/proposals --judge-runs 1

# 4. read the numbers
./.venv/bin/python -m norm_cards.eval.analyze --run results/myrun --dataset dataset20v3
```

`evaluate` parallelises across judge runs (`k`), not across problems, so with
`--judge-runs 1` it is sequential. To shard, run it several times over disjoint
`--problems` lists in parallel; output is per problem, so they do not collide.

**No API key?** `python -m norm_cards.test_gather` exercises search and ranking
against real keyless sources.

---

## How the experiment works

Both sides come from the same paper by two routes that never meet. One route keeps
the authors' experiments; the other keeps only the question they were answering.

```
paper ─┬─ transcribe experiments ──────────────► reference experiments
       │                                              │
       └─ draft the claim                             │
          (drop the method AND the apparatus)         │
                    │                                 │
                    ▼                                 ▼
              proposer ── one of three ──►  proposed experiments ──► judge ──► recall
                          conditions                                    ▲
                                                                        │
                nocard   nothing but the claim                    blind to the
                rag      full text of the field's papers          condition; never
                card     the curated norm card                    sees the card
```

The three conditions draw on **identical source material**: the `rag` arm gets raw
BM25 passages from the same subfield papers the card was distilled from, trimmed to
the card's own character budget. So the contrast is curation, not access or context
length.

### Recall, and why the denominator is the hard part

Recall is the mean over *reference* experiments of covered (1.0) / partial (0.5) /
missing (0.0). The mapping is many-to-many: one broad proposal can carry several
reference items.

But a paper usually supports several claims, and `transcribe` copies **all** of its
experiments — so about 46% of any reference is mechanism and external follow-ups
that this claim does not oblige. Scoring against all of it asks "would you have
reproduced the paper?", which is not the question. `analyze` therefore reports the
split, and it is the honest headline:

| denominator | card − no card |
|---|---|
| claim-obliged (headline + apparatus + control + confound) | +0.012 [−0.019, +0.042] |
| beyond-claim (mechanism + external) | **+0.074 [+0.025, +0.123]** |

**The card's measurable gain is concentrated in experiments the claim does not
require.** Fixing that means scoping the reference to the claim at transcription
time; it is the largest open item in the harness.

---

## Repository layout

```
datasets/dataset20v3/     tracked INPUTS — cannot be regenerated cheaply
  dataset.json              manifest: which claims, which models, review status
  references/               20 reference experiment sets, transcribed from papers
  cards/                    the 14 shared subfield cards (~2h of API time)
  subfield_bundles.jsonl    the papers each card was built from; feeds the rag arm
  subfields.json            claim → subfields, including 10 hand corrections
  subfield_overrides.json   each correction with its written rationale

norm_cards/
  dataset.py              bootstrap + assemble a dataset into a run directory
  subfield_gather.py      retrieve papers for a SUBFIELD (never using claim text)
  normcard.py             map/reduce a paper bundle into per-subfield menus
  curate.py               deterministic merge + prune; no model calls
  classifier.py           claim → subfields
  scify_proposer.py       the proposer, resynced against the real SciFy agent
  eval/
    make_claims.py        paper → claim
    transcribe.py         paper → reference experiments (gated behind --accept)
    build_cards.py        paper bundles → norm cards
    propose.py            run the arms
    evaluate.py           the blind judge
    analyze.py            arm means, paired intervals, the role split
    judge_suite.py        known-answer test suite for the judge itself

results/                  git-ignored. Disposable: rebuildable from datasets/.
```

---

## Building a dataset from scratch

Only needed for *new* claims; `dataset20v3` is already built.

```bash
# claims and references, drafted separately so neither leaks into the other
python -m norm_cards.eval.make_claims --papers papers.json --out claims.jsonl
python -m norm_cards.eval.transcribe  --problem <id> --arxiv <arxiv id>
python -m norm_cards.eval.transcribe  --problem <id> --accept --author "Your Name"

# subfields, then one retrieval and one card per subfield serving >=2 claims
python -m norm_cards.classifier       --claims claims.jsonl --out subfields.json
python -m norm_cards.subfield_gather  --subfields subfields.json --out shared/
python -m norm_cards.eval.build_cards --run shared/ --limit 1   # repeat until done
```

`--limit 1` builds one card per process. Card construction holds the full text of
every paper it has read, and the loop was OOM-killed four times before this existed;
re-running resumes, so a kill costs one card rather than the batch.

### What a claim must and must not say

This took three drafts to get right, and both failure modes cost a full rebuild.

- **Say** the phenomenon, the subject under test, and the bar **with its number**.
- **Never say** the authors' method — that hands over the design.
- **Never say** the apparatus — which datasets, baselines, prompts, splits or
  protocol. v2 named all of these and recall looked fine, but the proposer had
  nothing left to design, so the card had nothing to inform. The test: *if a
  competent researcher could have picked a different dataset or baseline set and
  still tested the same proposition, it is apparatus — leave it out.*

Measured across the three drafts: named artifacts per claim went 1.4 → 6.7 → **0.8**
while claims carrying a decidable bar went 0 → 17 → **19**.

### Reviewing references

`transcribe` writes `ground_truth.draft.json` and refuses to promote it; `--accept
--author` does that, and the schema validator rejects anything whose provenance
still says draft. The dataset's references are currently promoted with the author
line **"PENDING HUMAN SIGN-OFF"**, which the judge is shown. Re-accept under your own
name once you have read them against the papers.

---

## Things worth knowing before you trust a number

- **The judge is `gpt-5.6-terra` with prompt v2, and runs without tools.** v2 drops the
  tool instructions and says when breadth matters: a claim that something *can* happen
  needs one clean demonstration, a claim that it holds *generally* needs breadth. v1 stays
  in `prompts.PROMPTS`; `evaluate --prompt v1` reproduces older runs. The judge must never
  grade proposals written by its own model.
- **`judge_suite.py` is the judge's own test suite** — 14 behaviours x 3 unrelated domains,
  each case constructed with a known correct status per reference experiment and run
  several times. It reports target pass rate, collateral damage and self-agreement per
  `model:prompt`, so two judges can be compared directly. Terra beat luna on it: luna
  credits controls run outside the setting the claim names.
- **`propose` skips per claim, not per arm.** Adding a model in a second invocation
  against the same output directory silently skips every claim the first finished.
  Pass all models in one `--models` call, or add `--merge` to run only the missing
  models' arms and merge them into the existing files.
- **Contamination is filtered on arXiv v1 date**, never venue date.
- **One model shows no effect.** gpt-5.4 is −0.005 where gpt-5.5 and gpt-5.6-terra
  are both ≈+0.065. Pooling does real work in the headline.
