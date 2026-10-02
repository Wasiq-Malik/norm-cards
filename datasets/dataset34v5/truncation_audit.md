# Did the 18-page cutoff hide any claims?

`fulltext.extract_text` defaulted to `max_pages=18`. Transcription and claim drafting both
inherited that default, so dataset20v3's references and claims were written from a prefix of
each paper — as little as 46% of the text for one paper, under 70% for five.

For the v5 rewrite every paper was re-extracted in full (`FULL_PAPER = 200`) and read. What
sits past the old cutoff, by paper:

| Paper | Seen before | What was past the cutoff |
|---|---|---|
| 2510_20487 | 46% | Fine-tuning hyperparameters, expert-iteration details, the full prompt lists |
| 2606_00570 | 50% | Proofs of the two theorems, per-layer perturbation tables supporting them |
| 2608_09928 | 60% | Autoencoder training configs, steering decomposition, feature-correspondence tables |
| 2609_08444 | 64% | Planner prompt templates and worked reasoning examples |
| 2606_28153 | 65% | Threshold/bandwidth sweeps, qualitative ablation examples, **detector appendix D.5** |
| 2606_24026 | 74% | Run-stability numbers, API cost, the real-circuit case study (already a known experiment) |
| 2607_19374 | 85% | Soundness discussion, benchmark comparison, worked proof examples |
| 2604_24474 | 89% | Training-dynamics plots, generated-molecule statistics, drug-likeness tables |
| 2608_14049 | 80% | Per-component ablation details, real-world evaluation table |
| 2604_26496 | 97% | **Table 5: the full l2-threat-model results** |

## Verdict

**No whole claim was missed.** Every appendix section past the cutoff is one of: hyperparameters,
prompt text, proofs, qualitative examples, cost reporting, or supplementary tables for an
experiment already in the reference. Nothing past the cutoff is a distinct proposition the paper
sets out to decide.

**Two experiments were missed**, both now restored:

- `2604_26496[4]` — the l2 threat model repeat. v3 recorded "the supplied text does not include
  the numerical entries"; appendix Table 5 has them. The hand split dropped it as a thin,
  unnumbered claim on that basis. It is a robustness check on C0 and is now in C0.
- `2606_28153[13]` — an over-refusal evaluation plus a comparison against two other training-free
  detectors. Entirely unseen. It is a control on C0's detection bar: without it, a detector that
  flagged every hard benign prompt could pass. Now in C0 as a control.

Both omissions came from the same cause: an experiment whose numbers lived past page 18 looked,
from the prefix, like an experiment with no numbers.

## What this does not rule out

The references were drafted from each paper by a model and have never been checked against the
papers by a person. This rewrite checked them, but by another model. A paper's claim set is also
a judgement call: the hand split kept 14 further claims and set aside 13 thin ones, and that line
is mine, not the papers'.
