# norm-cards

Generate **scientific norm cards** for the AI subfield(s) a claim belongs to — a
structured, evidence-grounded summary of how researchers in that subfield *run
experiments* (standard datasets, models, metrics, protocols) — and use that card
to help an agent design **better experiments** to verify or refute a claim.

This repo has three stages:

1. **Search** (`run.py`) — a claim is classified into subfield(s) and OpenAlex
   topics, turned into norm-defining queries, searched across sources, deduped,
   and ranked into a **paper bundle**.
2. **Norm-card generation** (`normcard.py`) — the bundle's papers are read in full
   text and map-reduced into one **menu per subfield** (datasets / models /
   metrics / protocols, each item carrying its supporting papers as evidence).
3. **Evaluation harness** (`scify_proposer.py`) — the ablation that answers *does
   the card actually help?* It runs SciFy's real experiment-proposer flow twice on
   the same claim — **baseline** (no card) vs **method** (card injected) — with the
   norm card as the only variable, and lets us compare the proposed experiments.

Norm cards are generated for every claim. An earlier version classified each claim
`empirical` or `theoretical` and generated a card only for the former; that was
removed in August 2026 after it silently produced no card at all for a hybrid claim —
one asserting a formal guarantee *and* naming three standard benchmarks. Claims of
that shape are common (a guarantee plus an empirical comparison), and an
all-or-nothing gate on a binary label handled them worst of all. Whether the menu
generated from empirical literature is the right artifact for a guarantee-shaped claim
is a real question, but it is now visible in the card rather than hidden behind a
skip.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENAI_API_KEY (others optional)
```

`OPENAI_API_KEY` is the only required key. OpenAlex/arXiv/Semantic Scholar work
keyless (keys just raise limits). Nothing is read at import time; a missing
optional key only disables that one source.

---

## Usage

### 1. Gather a paper bundle for a claim

```bash
python -m norm_cards.run --claim "An Ultralytics YOLO11n detector on COCO ... AP50 >= 0.90 ..." \
    --model gpt-5-mini
```

Batch over a JSONL of `{ "problem_id", "claim", ... }` records (resumable):

```bash
python -m norm_cards.run \
    --claim_file claims.jsonl \
    --output_folder results/run1 \
    --model gpt-5-mini --per_query 8 --top_k 30
# -> results/run1/problem_<id>/bundle.json (+ papers.md) per claim
```

### 2. Generate a norm card from a bundle

```bash
python -m norm_cards.normcard --bundle results/norm_cards_hybrid/problem_12/bundle.json
# -> writes norm_card.json next to the bundle
# models: --map_model gpt-5-mini  --reduce_model gpt-5  --recipe_model gpt-5
```

A worked pair to inspect: `results/norm_cards_hybrid/problem_12/bundle.json` and the
`norm_card.json` it produces.

### 3. Run the evaluation harness

```bash
export NORM_CARDS_EVAL_ROOT=results/eval_icml2026_v2
export NORM_CARDS_CLAIMS=norm_cards/data/icml2026-claims-v1.jsonl

# one arm per (model, condition); conditions differ ONLY in current_evidence
python -m norm_cards.eval.propose \
    --cards results/eval_icml2026_v2/pipeline \
    --arms nocard,retrieval,card \
    --models gpt-5.4,gpt-5.5,gpt-5.6 \
    --out results/eval_icml2026_v2/proposals

python -m norm_cards.eval.evaluate --problems all \
    --source results/eval_icml2026_v2/proposals --judge-runs 3
python -m norm_cards.eval.report
```

| condition | `current_evidence` | what it isolates |
|---|---|---|
| `nocard` | `{}` | what the model already knows |
| `retrieval` | top-ranked papers, titles + abstracts | the presence of relevant literature |
| `card` | the curated norm card | the synthesis over that literature |

`retrieval` is the bar that matters. A norm card *is* a synthesis over retrieved
papers, so beating `nocard` only shows relevant literature helps; beating `retrieval`
at comparable context cost is what shows the extraction pipeline earns its keep.

To ask **which parts of the card are worth keeping**, run an arm per subset — the
question becomes a measurement instead of a matter of taste:

```bash
python -m norm_cards.eval.propose --cards <dir> --arms card --sections design
python -m norm_cards.eval.propose --cards <dir> --arms card --sections resources
```

Re-tuning curation costs nothing: the raw per-subfield menus stay in the card file,
so thresholds can be changed without refetching a paper or calling a model.

```bash
python -m norm_cards.curate --cards results/eval_icml2026_v2/pipeline --min_evidence 3
```

---

## Design decisions (read before extending the harness)

- **Faithful to SciFy, self-contained.** `scify_proposer.py` copies SciFy's
  `SYSTEM_PROPOSER` and `DECOMPOSITION_PROMPT` **verbatim** and imports nothing
  from the SciFy (`dryrun`) codebase — so the harness stays runnable standalone and
  the only thing that ever differs between arms is the injected card.
- **Resource constraints removed.** SciFy's proposer caps experiments at an
  8GB/30-min compute budget; we strip that block on purpose — we're testing whether
  the card helps design the *ideal* experiment set, not a resource-limited one.
- **Model: `gpt-5.5`** for the proposer/decomposer (temp forced to 1.0, the
  provider default reasoning effort — *medium* — for the gpt-5 series). Card
  generation uses `gpt-5-mini` (map) + `gpt-5` (reduce/recipes).
- **Same subclaims fed to both arms**, so the decomposer can't bias the comparison.

---

## Output schema (norm card, `format_version` 0.3)

```jsonc
{
  "type": "scientific_norm_card_set", "format_version": "0.3",
  "claim": "...", "subfields": ["...", "..."],
  "provenance": { "n_papers_deduped": 24, "n_fulltext": 16, ... },

  "norm_cards": [ /* one raw menu per subfield — provenance, not the deliverable */ ],

  "card": {                            // merged + curated: what consumers read
    // HOW the field argues — read first by the proposer, because this is the part
    // it cannot already guess.
    "protocols": [ { "name": "batch editing swept k in {1,10,100,1000,3000}",
                     "detail": "the specifics, one line", "evidence": ["<title>"] } ],
    "controls":  [ { "name": "random size-matched head set",
                     "detail": "how the condition is constructed",
                     "rules_out": "that any equally-sized intervention would do it",
                     "evidence": [...] } ],
    "confounds": [ { "name": "effect is capability damage, not belief change",
                     "detail": "why the headline result would look identical under it",
                     "ruled_out_by": "the measurement that discriminates",
                     "evidence": [...] } ],
    // WHAT it runs on.
    "datasets":  [ { "name": "COUNTERFACT", "role": "train|eval|robustness_eval", ... } ],
    "models":    [ { "name": "Llama-2", "role": "backbone|method_under_test|baseline",
                     "variants": ["Llama-2-7B-Chat", "LLaMA2-7B-Chat-HF"], ... } ],
    "metrics":   [ { "name": "Locality", "direction": "higher_better", ... } ]
  },
  "curation": { "totals": {"in": 178, "out": 42}, "datasets": {...}, ... }
}
```

Three invariants: **every item carries `evidence`** (the papers it came from — the
grounding guard drops anything unfaithful or unsupported); the card only ever contains
what the gathered literature supports; and a **resource** additionally needs
`min_evidence` independent papers before it counts, because a dataset one group used
once is that group's choice, not the subfield's norm.

### What changed in 0.3, and why

Measured on the v0.2 cards for the three ICML-2026 claims:

| | v0.2 | v0.3 |
|---|---|---|
| items per claim | 228 · 276 · 351 | ~31 · ~42 · ~35 |
| share that is resources | 75-86% | capped at 8 per key, design read first |
| items cited by exactly one paper | 59-75% | 0 among resources |
| cross-subfield duplicate items | 32 · 76 · 71 | 0 (one merged card per claim) |
| protocols naming a control condition | 1/47 · 1/38 · 6/88 | `controls` is its own section |
| rival explanations | none | `confounds` is its own section |
| what the proposer actually received | 163-216 bare nouns, a Python dict repr | prose, details intact, 40% fewer chars |

The `controls` and `confounds` sections exist because of a measurement, not a hunch.
Labelling every reference experiment by what it is *for* and pooling coverage across
claims and models showed proposals covering apparatus and headline experiments at
0.83-0.88 and confound-elimination experiments at **0.43** — and the v0.2 card made
that *worse*, not better, because with SciFy's 3-experiment cap its resource lists
displaced the controls the model would otherwise have proposed. See **Where the misses
are** in a generated report.

---

## Results & findings so far

`results/norm_cards_hybrid/` holds the shared experiment artifacts — one folder per
claim (`bundle.json`, `papers.md`, `norm_card.json`, `scify_proposer.json`) plus:

- **`GRAND_REPORT_10claims.md`** — the headline baseline-vs-method comparison across
  10 claims (6, 7, 9, 10, 11, 12, 21, 33, 36, 37) at gpt-5.5.
- **`showcase_card_wins.md`** — the two clearest card-attributable wins (#9
  system-ID error bounds, #21 HMM + DFA-constrained decoding), with menu provenance.
- **`ground_truth_recipes.md`** — an independent expert decomposition of each claim
  into the experiment set required to verify/refute it, to be used as the scoring
  reference for the evaluation harness.

**Headline result (n=1, temp=1):** at gpt-5.5 with SciFy's real flow, the card does
**not** improve experiment *coverage or correctness* — the baseline is strong and
never fabricates. The card's consistent effect is **methodological
concreteness/grounding** (naming the subfield's specific, current tooling), and it
is largest in **obscure fields** where parametric knowledge is thinnest (#9, #21).
See the report for caveats; a higher-n re-run of #9/#21 is the recommended next step
before any of this goes in a deck.

---

## Repo layout

```
norm_cards/
  run.py            # stage 1: claim -> paper bundle  (CLI: python -m norm_cards.run)
  classifier.py     #   claim -> subfields
  query_gen.py      #   subfields -> norm-defining search queries
  collect.py        #   search + dedup + citation-velocity ranking
  fulltext.py       #   PDF/full-text resolver chain (arXiv -> pdf_url -> Unpaywall -> S2)
  sources/          #   OpenAlex (default), arXiv, Semantic Scholar, SerpAPI adapters
  normcard.py       # stage 2: bundle -> norm_card.json  (map/reduce/curate/recipe)
  curate.py         #   deterministic half: canonicalize families, merge subfield
                    #     menus, drop uncorroborated resources (no model involved)
  scify_proposer.py # stage 3: the SciFy Proposer, resynced against dryrun @400e677
  llm.py            #   litellm wrapper (temp/model handling)
  config.py         #   defaults + key loading
  data/             #   PwC task seeds, OpenAlex topic seeds, sprint claims JSONL
  test_gather.py    #   keyless smoke test for the search/rank stages
  eval/             # stage 4: automated evaluation harness
    tools.py        #   web/OpenAlex/full-text/resource-check tool belt
    agent_loop.py   #   tool-calling loop, evidence ledger, tracing
    schemas.py      #   submit_evaluation schema, evidence-discipline validator,
                    #     and the reference-recipe input gate
    prompts.py      #   the judge's system prompt
    scoring.py      #   verdicts -> numbers; majority vote across judge runs
    reference.py    #   reference recipes as INPUT (--template / --check)
    transcribe.py   #   draft a reference FROM a paper, for a human to accept
    build_cards.py  #   batch gathering run -> per-problem norm cards
    propose.py      #   proposer arms: nocard / retrieval / card, x models
    invariance.py   #   judge stress test: resource substitution + order permutation
    evaluate.py     #   judge CLI (gpt-5.6-luna, k runs, blind to arm)
    calibrate.py    #   judge-vs-hand-label agreement check
    report.py       #   aggregate -> REPORT.md
results/norm_cards_hybrid/  # shared bundles, cards, proposer outputs, reports
results/eval_v2/            # reference recipes, judgments, REPORT.md (traces gitignored)
```

---

## Keyless smoke test

`test_gather.py` exercises the search/rank stages with hand-authored
classifications (no LLM key) on sample claims:

```bash
python -m norm_cards.test_gather --case 11   # empirical: detection robustness
```

---

## Automated evaluation harness (`norm_cards/eval/`)

Scores experiment sets for coverage, soundness, precision, norm alignment and
grounding, so pipeline changes can be measured as deltas on a fixed benchmark. Full
design: [`docs/eval_harness_design.md`](docs/eval_harness_design.md).

One agentic stage — the judge. Its baseline is its own reading of the claim and the
two experiment lists; the research belt (web search, OpenAlex, paper full text,
dataset/model existence checks) is an opt-in extra behind `--tools`, not the default.
That is a measured decision: on the judge stress suite the belt more than doubled the
drift of the order control — reversing a list, which cannot change what a set
establishes — from a mean 0.069 to 0.165, because rate-limited searches hand the judge
different evidence on different passes. It scores against
a **reference recipe**, which the harness takes as an input and never produces:
someone writes it, or transcribes it from the experiment section of the paper the
claim came from. Format: [`docs/reference_format.md`](docs/reference_format.md).

The harness deliberately has no way to generate a reference. One written by asking
a strong model to design experiments for the claim is not an independent standard —
it is another system's output, and if that model were good enough to define
correctness you would ship it as the proposer instead of scoring against it.

**The judge scores the set, not the experiments one at a time.** The reference is
flattened into *decision requirements* — what has to be established before the claim
can be called true or false — and the judge maps the whole proposed set onto them,
many-to-many. One experiment may carry three requirements; three may jointly carry
one. Reordering, merging or splitting the same work does not move the numbers, and a
different valid route to the same conclusion counts as satisfied. Per-experiment
judgments still happen, but they answer separate questions: is this design sound, does
it follow the field's norms, do its resources exist and fit, does it duplicate
something else in the set.

```bash
# Reference recipes — authored by hand or transcribed, validated and rendered here.
python -m norm_cards.eval.reference --template 11   # blank skeleton to fill in
python -m norm_cards.eval.reference --check         # validate + render markdown
# -> results/eval_v2/ground_truth/problem_<id>/{ground_truth.json, ground_truth.md}

# Generate proposer output, one arm per model (no norm card — model comparison).
python -m norm_cards.eval.propose --models gpt-5.4,gpt-5.5,gpt-5.6

# Score a pipeline's output against those fixtures, per arm, k runs.
python -m norm_cards.eval.evaluate --problems 11 --judge-runs 3   # gpt-5.6-luna
# -> results/eval_v2/judgments/problem_<id>/<arm>.json

python -m norm_cards.eval.report        # -> results/eval_v2/REPORT.md
python -m norm_cards.eval.calibrate --template   # judge vs. hand labels
```

A second claim set lives beside the DARPA sprint one via two env vars:

```bash
NORM_CARDS_EVAL_ROOT=results/eval_icml2026 \
NORM_CARDS_CLAIMS=norm_cards/data/icml2026-claims-v1.jsonl \
  python -m norm_cards.eval.reference --check
```

What keeps the judge honest:

- **The reference is authoritative about *what* must be decided, never about *how*.**
  Divergence from it is explicitly not a fault. It says its own provenance at the top,
  which the judge is shown. Material errors in a requirement go to
  `reference_defects`; work that closes a gap the reference misses goes to
  `unmet_by_reference` and earns credit rather than reading as a complaint.
- **Nothing can rest on "the reference disagrees."** Faulting an experiment requires a
  quoted source *and* a step-by-step reasoning chain from that quote to the fault.
  Marking a requirement unmet requires the chain but no quote — it is a statement about
  the set, which no search can confirm, and demanding citations for absence would just
  push the judge toward calling things satisfied. Both are enforced in
  `schemas.validate_evaluation`, which hands rejections back to the agent to fix.
- **Field norms come from the reference, not from the pipeline.** The generated norm
  card is the treatment in the ablation; showing the judge the treatment would hand the
  method arm an automatic match.
- **The judge is blind to the arm**, and never told a second arm exists.
- **Scores are computed in code** (`scoring.py`) from small per-requirement and
  per-experiment judgments; no agent ever emits a score. k runs are majority-voted,
  and calls without a majority are marked `CONTESTED` for hand review.
