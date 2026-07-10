# norm-cards

Generate **scientific norm cards** for the AI subfield(s) a claim belongs to: a
structured, evidence-grounded summary of how researchers in that subfield *run
experiments* (standard datasets, models, metrics, and protocols), plus concrete
**experiment recipes** a downstream agent can run to verify or refute a claim.

Two stages:

1. **Search** — a claim is classified into subfield(s) and OpenAlex topics, turned
   into norm-defining queries, searched across sources, deduped, and ranked into a
   **paper bundle**.
2. **Norm-card generation** — the bundle's papers are read in full text and
   map-reduced into one **menu per subfield** (datasets / models / metrics /
   protocols, each item carrying its supporting papers as evidence), then a
   **claim-level recipe** set is composed across those menus.

Norm-card generation currently targets **empirical claims** (which have clear
datasets/benchmarks/metrics); theoretical claims are classified but skipped.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENAI_API_KEY (others optional)
```

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
# -> results/run1/paper_bundles.jsonl (one bundle per claim)
```

### 2. Generate a norm card from a bundle

```bash
python -m norm_cards.normcard --bundle results/run1/problem_12/bundle.json
# -> writes norm_card.json next to the bundle
```

See `examples/` for a sample bundle and the norm card it produces.

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
  ],

  "experiment_recipes": [              // claim-level, composed across the menus above
    { "goal": "...", "dataset": "CIFAR-10", "model": "WideResNet",
      "method_under_test": "...", "attack_or_condition": "...",
      "metrics": ["Attack Success Rate", "..."],
      "pass_condition": "ASR <= 10% and clean Top-1 >= 85% at 1% poison",
      "rationale": "..." }
  ]
}
```

Two invariants: **every menu item carries `evidence`** (the papers it came from —
the grounding guard drops anything unfaithful or unsupported), and **recipes may
only use items that appear in the menus**.

## How it works

- **Sources** (`norm_cards/sources/`): default is **OpenAlex** (semantic search
  unioned with topic-filtered lexical search). arXiv, Semantic Scholar, and
  Google Scholar (SerpAPI) are opt-in via `--sources`.
- **Vocabularies**: PwC tasks (`data/pwc_tasks_seed.json`) for the human-readable
  subfield label; OpenAlex topics are resolved per claim at runtime via the
  `text/topics` endpoint (full ~4,516-topic taxonomy).
- **Full text** (`norm_cards/fulltext.py`): resolver chain arXiv → the record's
  pdf_url → Unpaywall → Semantic Scholar; ~15/25 papers per bundle typically
  resolve, the rest fall back to their abstract.
- **Generation** (`norm_cards/normcard.py`): per-paper extraction on a cheap model
  (parallel), per-subfield reduce on a strong model (canonicalize + subfield-filter
  + attach evidence), then a separate recipe call over the finished menus.

## Keys

Only `OPENAI_API_KEY` is required. OpenAlex/arXiv/Semantic Scholar work keyless
(keys raise limits). The OpenAlex `text/topics` classification endpoint is metered
(~$0.01/call) — see `.env.example`. Nothing is read at import time; missing
optional keys just disable that source.

## Keyless smoke test

`norm_cards/test_gather.py` exercises the search/rank stages with hand-authored
classifications (no LLM key) on three sample claims:

```bash
python -m norm_cards.test_gather --case 11   # empirical: detection robustness
```
