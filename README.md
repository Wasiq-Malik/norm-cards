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
  scify_proposer.py # stage 3: evaluation harness (baseline vs card ablation)
  llm.py            #   litellm wrapper (temp/model handling)
  config.py         #   defaults + key loading
  data/             #   PwC task seeds, OpenAlex topic seeds
  test_gather.py    #   keyless smoke test for the search/rank stages
results/norm_cards_hybrid/  # shared bundles, cards, proposer outputs, reports
```

---

## Keyless smoke test

`test_gather.py` exercises the search/rank stages with hand-authored
classifications (no LLM key) on sample claims:

```bash
python -m norm_cards.test_gather --case 11   # empirical: detection robustness
```

---

## Next: automated evaluation harness (in progress)

The current `scify_proposer.py` produces the baseline/method experiment sets; the
next step is scoring them automatically against `ground_truth_recipes.md`
(coverage, correctness, groundedness) so we can evaluate overall pipeline
correctness at scale — letting the proposer generate more than 3 experiments and
without the compute cap.
