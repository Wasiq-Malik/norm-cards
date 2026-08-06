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

Norm-card generation targets **empirical claims** (which have clear
datasets/benchmarks/metrics). Theoretical claims are classified and can be run by
overriding the empirical guard (see below), but the card's value is thinnest there.

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

### 3. Run the evaluation harness (baseline vs card)

```bash
python -m norm_cards.scify_proposer \
    --card results/norm_cards_hybrid/problem_9/norm_card.json \
    --model gpt-5.5 \
    --out results/norm_cards_hybrid/problem_9/scify_proposer.json
```

This decomposes the claim into subclaims (SciFy's decomposition prompt), then runs
the proposer twice — `current_evidence={}` (baseline) and `current_evidence={norm
card menu}` (method) — and writes `{baseline_experiments, method_experiments,
subclaims, ...}`. **The card is the only difference between the two arms**, so any
change in the proposed experiments is attributable to it.

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

## Output schema (norm card, `format_version` 0.2)

```jsonc
{
  "type": "scientific_norm_card_set",
  "claim": "...", "subfields": ["...", "..."],
  "provenance": { "n_papers_deduped": 24, "n_fulltext": 16, "pdf_sources": {...}, ... },

  "norm_cards": [                      // one card per subfield
    { "subfield": "Data Augmentation",
      "menu": {
        "datasets":  [ { "name": "CIFAR-10", "role": "train|eval|robustness_eval",
                         "evidence": ["<paper title>", ...] } ],
        "models":    [ { "name": "ResNet-18", "role": "backbone|method_under_test|baseline",
                         "evidence": [...] } ],
        "metrics":   [ { "name": "Attack Success Rate", "direction": "lower_better",
                         "evidence": [...] } ],
        "protocols": [ { "name": "poison rate swept 1-10%", "detail": "...",
                         "evidence": [...] } ] },
      "guard": { "items_dropped": 0, "evidence_links_dropped": 2 } }
  ]
}
```

Two invariants: **every menu item carries `evidence`** (the papers it came from —
the grounding guard drops anything unfaithful or unsupported), and the card only
ever contains what the gathered literature supports.

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
  classifier.py     #   claim -> subfields + claim_type
  query_gen.py      #   subfields -> norm-defining search queries
  collect.py        #   search + dedup + citation-velocity ranking
  fulltext.py       #   PDF/full-text resolver chain (arXiv -> pdf_url -> Unpaywall -> S2)
  sources/          #   OpenAlex (default), arXiv, Semantic Scholar, SerpAPI adapters
  normcard.py       # stage 2: bundle -> norm_card.json  (map/reduce/recipe)
  scify_proposer.py # stage 3: baseline vs card ablation (SciFy proposer, verbatim)
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

Scores the experiment sets from `scify_proposer.py` for coverage, correctness,
groundedness and decision sufficiency, so pipeline changes can be measured as
deltas on a fixed benchmark. Full design: [`docs/eval_harness_design.md`](docs/eval_harness_design.md).

One agentic stage — the judge — with web search, OpenAlex, paper full text, and
dataset/model existence checks, under no time or compute budget. It scores against
a **reference recipe**, which the harness takes as an input and never produces:
someone writes it, or transcribes it from the experiment section of the paper the
claim came from. Format: [`docs/reference_format.md`](docs/reference_format.md).

The harness deliberately has no way to generate a reference. One written by asking
a strong model to design experiments for the claim is not an independent standard —
it is another system's output, and if that model were good enough to define
correctness you would ship it as the proposer instead of scoring against it.

```bash
# Reference recipes — authored by hand, validated and rendered here.
python -m norm_cards.eval.reference --template 11   # blank skeleton to fill in
python -m norm_cards.eval.reference --check         # validate + render markdown
# -> results/eval_v2/ground_truth/problem_<id>/{ground_truth.json, ground_truth.md}

# Score a pipeline's output against those fixtures, per arm, k runs.
python -m norm_cards.eval.evaluate --problems 11 --judge-runs 3   # gpt-5.6-luna
# -> results/eval_v2/judgments/problem_<id>/{baseline,method}.json

python -m norm_cards.eval.report        # -> results/eval_v2/REPORT.md
python -m norm_cards.eval.calibrate --template   # judge vs. hand labels
```

Each experiment gets one of three verdicts — `ACCURATE` (1.0) / `MIXED` (0.5,
right idea but under-specified) / `IRRELEVANT` (0.0, would yield nothing useful).

What keeps the judge honest:

- **The reference is treated as fallible.** It is a first orientation, not an
  authority — and it says its own provenance at the top, which the judge is shown.
  A divergent-but-sound experiment scores ACCURATE *and* files a defect against the
  reference; those defects are the input to a human's decision to revise it.
  Nothing is revised automatically.
- **No verdict can rest on "the reference disagrees."** Faulting an experiment
  requires a quoted source *and* a step-by-step reasoning chain from that quote to
  the fault — enforced in `schemas.validate_evaluation`, which hands rejections
  back to the agent to fix.
- **The judge is blind to the arm**, and never told a second arm exists.
- **Scores are computed in code** (`scoring.py`) from small per-experiment
  judgments; no agent ever emits a score. k runs are majority-voted, and verdicts
  without a majority are marked `CONTESTED` for hand review.
