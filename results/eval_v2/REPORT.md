# Norm-card evaluation — baseline vs method

Judge: `gpt-5.6-luna` @ high effort, k=3 runs, majority vote. Proposer: `gpt-5.5`. Claims scored: 2.

Verdicts per experiment are ACCURATE=1.0 / MIXED=0.5 / IRRELEVANT=0.0; *correctness* is their mean. *Coverage* is weighted fill of the decision structure (gate .30, headline .40, apparatus .15, control .15), with roles the judge marked not-required dropped from the denominator. *Grounded* is the fraction of named resources that exist, are obtainable, and fit their use. *Redund.* is the fraction of experiments duplicating another in the same set (lower is better).

## Reference provenance

| Claim | Reference recipe came from |
|---|---|
| 7 | llm_generated · gpt-5.6-sol @ xhigh effort — machine-written, unverified by a human |
| 11 | llm_generated · gpt-5.6-sol @ xhigh effort — machine-written, unverified by a human |

> **⚠️ These numbers are not a measurement of the pipeline.** The reference recipes for claim(s) 7, 11 are machine-generated or unattributed. A reference written by asking a model to design experiments for the claim is not an independent standard — it is another system's output. Replace them with hand-authored or paper-derived references (see `docs/reference_format.md`) and re-run with `--force` before quoting anything below.

## Per-claim scores

| Claim | Arm | n | Verdicts | Correct | Coverage | Grounded | Redund. | Sufficiency | Contested |
|---|---|---|---|---|---|---|---|---|---|
| 7 | baseline | 3 | MIXE MIXE MIXE | 0.50 | 0.50 | 0.91 | 0.67 | insuff | 0 |
| 7 | method | 3 | MIXE MIXE MIXE | 0.50 | 0.50 | 0.62 | 0.00 | insuff | 0 |
| 11 | baseline | 3 | ACCU MIXE ACCU | 0.83 | 0.93 | 0.79 | 0.00 | insuff | 0 |
| 11 | method | 3 | ACCU MIXE MIXE | 0.67 | 0.65 | 0.60 | 0.00 | insuff | 0 |

## Method − baseline

The pipeline-improvement signal: positive means the norm card helped on that metric for that claim.

| Claim | Δ correctness | Δ coverage | Δ grounded | Δ sufficiency |
|---|---|---|---|---|
| 7 | 0.00 | 0.00 | -0.28 | 0.00 |
| 11 | -0.17 | -0.28 | -0.19 | 0.00 |

**Means across claims**

| Metric | Baseline | Method | Δ |
|---|---|---|---|
| correctness | 0.67 | 0.58 | -0.08 |
| coverage | 0.71 | 0.57 | -0.14 |
| groundedness | 0.85 | 0.61 | -0.23 |
| redundancy_penalty | 0.33 | 0.00 | -0.33 |
| sufficiency_score | 0.00 | 0.00 | 0.00 |

## Reference-recipe defects filed by the judge

Input for the *manual* decision to regenerate a ground truth. The reference is a fixture; nothing here changes it automatically.

**Claim 7**
- `G0` (baseline): No explicit pre-training nonzero-baseline gate or fixed evaluation cells are provided, although relative reduction requires a defined nonzero denominator.
- `A1` (baseline): The designs do not guarantee that exactly one new attention head is added while all original weights remain unchanged.
- `H1` (baseline): The direct headline permits post-hoc scope selection through 'at least one' benchmark and 'ideally' multiple benchmarks, and does not lock model, judge, or generation protocol.
- `C1` (baseline): Utility benchmarks and prompt masking do not directly rule out universal refusal or semantic destruction; matched benign refusal and matched token-score ablations are needed.
- `G0` (method): The proposed set has no sealed, pre-training gate that fixes the model, benchmark cells, policy, and nonzero denominator before training or evaluation.
- `A1` (method): Experiment 1's architecture and training description is insufficiently literal to establish that only one additional attention head was added while the original model was unchanged.
- `H2` (method): Experiment 2 labels an attack-generation approach adaptive but does not require target-specific optimization, separate attacks for base and head, or identical attack budgets.
- `H1` (method): Experiment 0 mixes behavior-level policy benchmarks with toxicity resources and leaves the violation label and evaluator aggregation under-specified, so it cannot serve as a decisive headline test.
- `A1 apparatus / C1 control` (method): The reference recipe could be strengthened by adding the proposed parameter-matched random-head and non-attention intervention comparators. Those controls can distinguish selective attention-head causality from a generic extra-parameter or refusal-calibration effect, provided the head and evaluation splits are first fully specified.
- `A1/A2/C1` (method): The reference recipe does not include the proposed parameter-matched random-head and non-attention intervention comparators. These are useful additional controls for separating the effect of an attention-head mechanism from generic added capacity or refusal calibration.

**Claim 11**
- `A1` (baseline): The proposed experiment set identifies the need for generator validation but incorrectly permits COCO val2017 to be the held-out generator-validation set. Because val2017 is the claim's headline test set, this creates leakage; the recipe's nonleaking train-derived paired validation requirement is necessary.
- `A1` (baseline): The proposed experiment does not require paired lighting observations. The cited learned-perturbation framework says paired perturbed examples are the requirement, and its illumination apparatus uses the same scenes under multiple lighting variations; unpaired COCO images do not establish the claimed illumination perturbation set.
- `A1` (method): The proposed set's generator audit does not require paired same-content lighting observations or an equivalent validated COCO-derived intervention, although the claim specifically calls for a learned illumination-consistent perturbation model.
- `A2` (method): The proposed set lacks the held-out latent approximation/coverage validation needed to establish that the learned set captures real illumination perturbations rather than merely generating visually smooth changes.
- `H3` (method): The proposed attack permits pixel-space projection of decoded outputs, which can violate the defining generator-range constraint; it must instead search or reject within the intersection of generator outputs, latent bound, and pixel L-infinity bound.
- `A1` (method): The reference recipe's demand for a particular COCO-derived paired recapture/relighting construction and its imported low-resolution reconstruction thresholds is stronger and more specific than the claim itself. A suitably documented alternative generator audit could be valid, but it still needs paired/equivalent evidence and predeclared criteria; the proposed Experiment 1 does not yet provide enough of that evidence.
- `A1/A2` (method): The reference's paired-data requirement is supported, but its specific COCO-derived recapture implementation and imported numerical thresholds are not uniquely entailed by the claim; equivalent documented paired illumination data and validation could suffice.
- `H3` (method): The reference's exact latent norm, attack schedule and intersection optimizer are defensible design choices, but the claim does not specify them. They should be preregistered and justified rather than treated as uniquely required.
- `G0` (method): The reference's claim that C<0.90 directly refutes the full conjunction assumes that the identity/no-modification case is included in the stated generator threat set. The proposed experiment appropriately says 'likely infeasible' rather than making that implication unconditional.
- `A1/A2` (method): The reference correctly identifies paired illumination data and held-out validation as important, but its exact COCO-derived recapture interpretation and imported numerical thresholds are more specific than the claim. The qualitative checks are useful supplements, not replacements.

## Missing pieces, by claim

What the judge said each set would still need to decide the claim.

**7 / baseline**
- Pre-register exact model checkpoints, tokenizer/chat template, decoding and generation limits, literal one-head architecture, frozen-base integrity, and training/selection procedure.
- Use named official held-out text-only splits with exact/semantic near-duplicate controls and a fixed violation definition plus blinded human/validated judge.
- Require and report the exact relative rule PVR_head/PVR_base <= 0.50, nonzero baseline denominators, prompt-level counts, confidence intervals, and a fixed scope rather than 'at least one ... ideally multiple.'
- Run defense-aware adaptive attacks against the augmented model, not only static held-out harmful prompts.
- Add standardized benign anti-over-refusal controls and matched uniform/permuted/inverted-risk ablations; use causal token interventions that control for prompt-semantic destruction.

**7 / method**
- A pinned unmodified checkpoint and exact single-head architecture with frozen original weights, zero-gate/base-equivalence test, and reproducible training/split protocol.
- A behavior-specific, pre-registered violation definition and held-out direct evaluation (for example HarmBench behavior classification with fixed decoding and generation length), with exact relative-reduction calculation and nonzero baseline gate.
- A genuinely adaptive evaluation in which attacks are separately optimized against the unmodified and augmented targets under identical budgets, rather than merely applying public or generated attack prompts.
- A pre-registered aggregation rule and thresholds for evaluator disagreement, confidence intervals, benign refusal/utility, and mechanism-ablation outcomes.
- Explicitly separate toxicity/degen benchmarks (RealToxicityPrompts, generic Jigsaw toxicity) from the headline policy-violation endpoint.

**11 / baseline**
- Create or identify paired, same-content COCO-derived illumination observations, with documented geometry/label preservation, rather than training on unpaired COCO images.
- Use a generator-train/validation split disjoint from COCO val2017 and freeze architecture, checkpoint, latent norm and B before the headline attack.
- Define and validate the CVAE perturbation set with held-out paired lighting data, latent-PGD approximation/coverage metrics, and semantic/label-preservation checks.
- Specify the detector attack loss and a common preregistered selection rule for candidates across objectives/restarts; log feasibility and gradient/convergence checks.
- Resolve category-ID handling explicitly: native COCO person category_id=1 versus Ultralytics's internal person class ID 0.

**11 / method**
- A COCO-derived paired, same-content multi-illumination training protocol, with documented geometry/label preservation and no val2017 leakage.
- A separate scene/content-disjoint generator-validation set and held-out approximation/coverage tests to establish that the CVAE represents illumination rather than arbitrary image changes.
- A preregistered latent coordinate system, norm, and radius B selected independently of YOLO val2017 robustness results.
- A feasibility-preserving attack implementation that guarantees every final image is exactly g(x,z), satisfies the latent bound, and satisfies ||g(x,z)-x||∞≤12/255; decoded pixel clipping/projection must not be used as a substitute.
- A frozen common clean/attacked COCOeval pipeline, explicit all-image handling and class/NMS settings, plus identity and budget/objective convergence controls before interpreting a passing robustness result.
