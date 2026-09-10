# Norm-card ablation — `gpt-5.4`, `gpt-5.5`, `gpt-5.6`

Judge `gpt-5.6-luna`, k=1/3, tools off. 3 claim(s), 9 arm(s).

**Recall** — of the experiments the source team actually ran, how many would a team running the proposed set have effectively performed?

For each reference experiment the judge asks whether the proposed set would establish what it establishes. `covered` = 1.0 (by any route — the method does not have to match), `partial` = 0.5 (gets at it, outcome stays ambiguous), `missing` = 0. Recall is the mean. Mapping is many-to-many, so a short proposal is not penalised for being short.

> Judge self-test passed on 3/3 claims: scoring each reference against itself returns recall 1.00.

## Reference provenance

| Claim | Reference recipe came from |
|---|---|
| arxiv2510_20487 | paper_derived · authored by Claude (transcribed from the source paper) — awaiting the user's check · 2026-08-27 · derived from Hua, Qin, Marks & Nanda, Steering Evaluation-Aware Language Models to Act Like They Are Deployed (arXiv:2510.20487v5); https://arxiv.org/abs/2510.20487; PDF: https://arxiv.org/pdf/2510.20487 |
| icml61272 | paper_derived · authored by Claude (transcribed from the source paper) — NOT yet reviewed by a human · 2026-08-21 · derived from Ren et al., Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence, ICML 2026 (arXiv:2606.00570v1); https://icml.cc/virtual/2026/poster/61272; Implementation: EasyEdit, https://github.com/zjunlp/EasyEdit |
| icml64633 | paper_derived · authored by Claude (transcribed from the source paper) — NOT yet reviewed by a human · 2026-08-21 · derived from Yin, Han & Li, Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models, ICML 2026 (oral) (arXiv:2606.28153v2); https://icml.cc/virtual/2026/poster/64633; Code: https://github.com/MorningYin/Robust-Harmful-Features |

> **⚠️ Unreviewed references.** The recipe(s) for claim(s) icml61272, icml64633 are transcribed from a source but no human has signed off on the claim statement, the role assignment, or what was left out. The numbers below are reproducible, not yet authoritative.

## Recall

| Claim | Arm | Recall | covered / partial / missing | n ref |
|---|---|---|---|---|
| arxiv2510_20487 | gpt-5.4+card | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.4+nocard | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.4+retrieval | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.5+card | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.5+nocard | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.5+retrieval | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+card | **1.00** | 8 / 0 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+nocard | **1.00** | 8 / 0 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+retrieval | **1.00** | 8 / 0 / 0 | 8 |
| icml61272 | gpt-5.4+card | **0.67** | 2 / 4 / 0 | 6 |
| icml61272 | gpt-5.4+nocard | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.4+retrieval | **0.83** | 4 / 2 / 0 | 6 |
| icml61272 | gpt-5.5+card | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.5+nocard | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.5+retrieval | **0.67** | 2 / 4 / 0 | 6 |
| icml61272 | gpt-5.6+card | **0.75** | 4 / 1 / 1 | 6 |
| icml61272 | gpt-5.6+nocard | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.6+retrieval | **0.67** | 2 / 4 / 0 | 6 |
| icml64633 | gpt-5.4+card | **0.93** | 6 / 1 / 0 | 7 |
| icml64633 | gpt-5.4+nocard | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.4+retrieval | **0.64** | 3 / 3 / 1 | 7 |
| icml64633 | gpt-5.5+card | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.5+nocard | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.5+retrieval | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.6+card | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.6+nocard | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.6+retrieval | **0.93** | 6 / 1 / 0 | 7 |

## Did the norm card help?

Same model, same claim, same subclaims — conditions differ only in what sits in `current_evidence`. Each Δ is against `nocard`, paired run by run.

`retrieval` is the control that matters for the project's thesis: a norm card is a synthesis over retrieved papers, so beating `nocard` only shows that relevant literature helps. Beating `retrieval` is what shows the synthesis is doing work.

### By claim (averaged over models)

| Claim | n models | nocard | retrieval | card | Δ retrieval | Δ card |
|---|---|---|---|---|---|---|
| arxiv2510_20487 | 3 | 0.96 | 0.96 | 0.96 | 0.00 | 0.00 |
| icml61272 | 3 | 0.75 | 0.72 | 0.72 | -0.03 | -0.03 |
| icml64633 | 3 | 0.86 | 0.81 | 0.88 | -0.05 | +0.02 |
| **all** | **9** | **0.86** | **0.83** | **0.85** | **-0.03** | **-0.00** |

### By model and claim

| Claim | Model | nocard | retrieval | card | Δ retrieval | Δ card |
|---|---|---|---|---|---|---|
| arxiv2510_20487 | gpt-5.4 | 0.94 | 0.94 | 0.94 | 0.00 | 0.00 |
| arxiv2510_20487 | gpt-5.5 | 0.94 | 0.94 | 0.94 | 0.00 | 0.00 |
| arxiv2510_20487 | gpt-5.6 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| icml61272 | gpt-5.4 | 0.75 | 0.83 | 0.67 | +0.08 | -0.08 |
| icml61272 | gpt-5.5 | 0.75 | 0.67 | 0.75 | -0.08 | 0.00 |
| icml61272 | gpt-5.6 | 0.75 | 0.67 | 0.75 | -0.08 | 0.00 |
| icml64633 | gpt-5.4 | 0.86 | 0.64 | 0.93 | -0.21 | +0.07 |
| icml64633 | gpt-5.5 | 0.86 | 0.86 | 0.86 | 0.00 | 0.00 |
| icml64633 | gpt-5.6 | 0.86 | 0.93 | 0.86 | +0.07 | 0.00 |

`retrieval` vs `nocard`: **mean Δ = -0.025** over 9 paired run(s) — helped 2, hurt 3, tied 4. 95% CI [-0.084, +0.034]. **Interval contains zero: no detectable effect.**

`card` vs `nocard`: **mean Δ = -0.001** over 9 paired run(s) — helped 1, hurt 1, tied 7. 95% CI [-0.027, +0.024]. **Interval contains zero: no detectable effect.**

One coverage step on a 7-experiment reference moves recall by 0.071, so that is the smallest difference this can resolve. A mean below it bounds the effect rather than measuring it.

## Where the misses are

Coverage grouped by what each reference experiment is FOR, pooled across claims and models. This is the diagnostic a recall number cannot give: it says which KIND of experiment goes missing, and so which part of a norm card would have to change to fix it.

| What the experiment is for | refs | nocard | retrieval | card |
|---|---|---|---|---|
| **apparatus** — build/validate what the headline test presupposes | 7 | 0.93 | 0.88 | 0.88 |
| **headline** — the direct test of the claim | 4 | 0.92 | 0.92 | 0.96 |
| **mechanism** — show the asserted cause is the operative one | 3 | 0.67 | 0.72 | 0.78 |
| **control** — rule out that the result is an artifact of intervening at all | 3 | 1.00 | 1.00 | 1.00 |
| **confound** — kill a rival explanation under which the result looks the same | 3 | 0.67 | 0.61 | 0.67 |
| **external** — show the finding generalises beyond its own setup | 1 | 1.00 | 0.83 | 0.83 |

## Which experiments were covered

`C` covered · `p` partial · `-` missing. Columns are reference experiment indices; see each claim's `ground_truth.md` for what they are.

**arxiv2510_20487**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| gpt-5.4+card | C | C | C | C | C | C | C | p |
| gpt-5.4+nocard | C | C | C | C | C | C | C | p |
| gpt-5.4+retrieval | C | C | C | C | C | C | C | p |
| gpt-5.5+card | C | C | C | C | C | C | C | p |
| gpt-5.5+nocard | C | C | C | C | C | C | C | p |
| gpt-5.5+retrieval | C | C | C | C | C | C | C | p |
| gpt-5.6+card | C | C | C | C | C | C | C | C |
| gpt-5.6+nocard | C | C | C | C | C | C | C | C |
| gpt-5.6+retrieval | C | C | C | C | C | C | C | C |

**icml61272**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| gpt-5.4+card | C | p | p | p | C | p |
| gpt-5.4+nocard | C | p | p | C | C | p |
| gpt-5.4+retrieval | C | p | C | C | C | p |
| gpt-5.5+card | p | p | C | C | C | p |
| gpt-5.5+nocard | C | p | p | C | C | p |
| gpt-5.5+retrieval | p | p | p | C | C | p |
| gpt-5.6+card | C | - | C | C | C | p |
| gpt-5.6+nocard | C | p | C | p | C | p |
| gpt-5.6+retrieval | p | p | p | C | C | p |

**icml64633**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| gpt-5.4+card | C | C | C | C | C | p | C |
| gpt-5.4+nocard | C | C | C | C | p | p | C |
| gpt-5.4+retrieval | C | C | C | p | p | - | p |
| gpt-5.5+card | C | C | C | C | p | p | C |
| gpt-5.5+nocard | C | C | C | p | C | p | C |
| gpt-5.5+retrieval | C | C | C | p | C | p | C |
| gpt-5.6+card | C | C | C | C | C | p | p |
| gpt-5.6+nocard | C | C | C | C | p | p | C |
| gpt-5.6+retrieval | C | C | C | C | C | p | C |

## What the proposals missed

The judge's account of what each set would still need.

**arxiv2510_20487 / gpt-5.4+card**
- A correctness or task-success measure on the same target task/response mode whose evaluation-aware behavior is being suppressed, compared across unsteered and steered conditions.
- A more discriminating mechanism test, such as explicit status reasoning or a control where evaluation language refers to the user rather than the model, ideally with evidence linking the status change to the behavioral shift.

**arxiv2510_20487 / gpt-5.4+nocard**
- A direct semantic control in which the user is the one being evaluated, while the model is not, to rule out a generic response to the word or concept of evaluation.
- A more direct causal readout of the model's operative deployment-status belief, such as internal deliberation or an equivalent intervention/readout designed to distinguish status belief from prompt-framing representation.

**arxiv2510_20487 / gpt-5.4+retrieval**
- A causal mechanism test separating a change in operative deployment-status belief from lexical/contextual priming or direct alteration of the output policy, using controls such as the user-being-evaluated condition, reasoning-trace analysis, or an equivalent causal mediation design.

**arxiv2510_20487 / gpt-5.5+card**
- Add a semantic control in which the user, rather than the model, is described as being evaluated, while keeping the model's deployment status fixed.
- Use an intervention-independent or causally validated measure of the model's deployment-status representation, ideally with reasoning-trace or mediation evidence and tests after successive robustness/hardening stages.

**arxiv2510_20487 / gpt-5.5+nocard**
- A clean mediation/specificity control that holds the word or framing of evaluation constant while independently manipulating or measuring deployment-status belief, together with a causal test showing that the behavioral shift follows the status representation rather than merely co-occurring with it.
- If robustness across the constructed organism's training stages is important to the intended mechanism claim, repeat the intervention after the relevant hardening stages and verify the result with a vector extracted from the final subject.

**arxiv2510_20487 / gpt-5.5+retrieval**
- A direct counterfactual control distinguishing evaluation of the model from evaluation language referring only to the user or another entity.
- A stronger causal/semantic validation that the shifted internal representation is the model's own deployment-status belief, such as status-focused reasoning-trace or activation-patching tests under conflicting deployment and evaluation cues.

**icml61272 / gpt-5.4+card**
- An explicit semantic-judge versus deterministic token-level scoring validation with a prespecified agreement rule.
- A direct measurement of normalized low-variance directional amplification across successive cumulative edit counts, with the relevant trend tested across a broader set of methods and architectures.
- Replications at larger model scales and on different knowledge types and confidence/difficulty strata, plus configuration and architecture sensitivity tests for the strongest method that does not collapse.

**icml61272 / gpt-5.4+nocard**
- A paired semantic-versus-deterministic scoring validation with an explicit disagreement rule.
- Geometric replication across multiple editing methods and architectures, including a defined edit-count analysis of whether low-variance amplification increases monotonically.
- Larger-model, knowledge-type, fact-difficulty/confidence, hyperparameter, and architecture-sensitivity sweeps.

**icml61272 / gpt-5.4+retrieval**
- An explicit semantic-judge versus deterministic token-level scoring cross-check across the editing dimensions.
- A robustness sweep over larger model scales, event-level as well as atomic/triple facts, high- versus low-confidence or difficulty-stratified facts, and systematic editor hyperparameter and architecture sensitivity.

**icml61272 / gpt-5.5+card**
- An explicitly specified plain autoregressive, non-teacher-forced single-edit evaluation with a clean pre-edit baseline.
- A deterministic token-level match-ratio/exact-match cross-check of the semantic evaluator and resulting method rankings.
- A robustness sweep over larger model scales and event-level knowledge, with explicit confidence/difficulty strata and hyperparameter/architecture sensitivity for potentially robust editors.

**icml61272 / gpt-5.5+nocard**
- An explicit independent semantic scoring assessment cross-validated against deterministic token-level scoring.
- Direction-normalized low-variance amplification measurements repeated over edit counts and across multiple editors and architectures.
- A systematic robustness sweep over model scale, fact-confidence strata or difficulty, knowledge type, and hyperparameter/architecture configurations.

**icml61272 / gpt-5.5+retrieval**
- An explicit clean-baseline and leakage-free autoregressive comparison for the narrow single-edit scores, including a stated protocol for contrasting them with teacher-forced results.
- A paired semantic-judge versus deterministic token-level scoring validation with a prespecified rule for resolving disagreement.
- A cumulative, direction-resolved amplification analysis across edit counts, representative methods and multiple architectures.
- Systematic robustness sweeps over larger model scales, event-level versus triple-shaped knowledge, fact-confidence or difficulty strata, and method hyperparameters/architectures.

**icml61272 / gpt-5.6+card**
- An independent semantic-judge evaluation cross-checked against deterministic token-level match and exact-match scoring.
- Tests on substantially larger models, event-level knowledge, high- versus low-confidence or difficulty-stratified facts, and sensitivity/failure analysis of the apparent non-collapsing editor.

**icml61272 / gpt-5.6+nocard**
- An explicit semantic-judge versus deterministic token-level scoring cross-check with a prespecified resolution of disagreement.
- A direction-wise relative-change analysis normalized by baseline representation variance, including the cumulative threshold and monotonicity test.
- A broader robustness sweep across model scale, knowledge type, fact confidence or difficulty, editing families, and hyperparameter or architecture configurations, with the parameter-free alternative compared against the relevant parameter editors.

**icml61272 / gpt-5.6+retrieval**
- An explicitly paired plain-autoregressive versus teacher-forced evaluation with a clean pre-edit baseline.
- A semantic-versus-deterministic scoring cross-check on the same outputs.
- Directional low-variance amplification measurements repeated across sequential edit counts and more than one architecture.
- Robustness sweeps over model scale, knowledge format, fact confidence or difficulty, and relevant method hyperparameters and architecture conditions.

**icml64633 / gpt-5.4+card**
- A whole-network measurement of the refusal-direction signal across layers, attention heads, and MLPs under matched successful jailbreaks.
- Token-level linear attribution separating harmful-request tokens from jailbreak-template tokens for the candidate head types.
- A broader cross-model and cross-benchmark evaluation of the zero-training detector with a prespecified competitiveness criterion and guaranteed dedicated-safety baselines.

**icml64633 / gpt-5.4+nocard**
- Token-level attribution comparing original harmful-request tokens with jailbreak-template tokens for the candidate and other head types, or an equivalently discriminating control against the baseline harmful-content-sensitivity explanation.

**icml64633 / gpt-5.4+retrieval**
- A matched equal-size arbitrary-head or other generic perturbation control for the sufficiency intervention.
- A direct perturbation of the complementary head population while measuring the residual harmful-content signal.
- Token-level attribution separating original harmful-request content from attack-template content.
- Cross-model and cross-family evaluation of the no-training detector if the claim is intended to generalize beyond the single target model.

**icml64633 / gpt-5.5+card**
- A complementary-head suppression experiment on successful attacks that measures the residual mid-layer harmful/safety signal.
- Per-token attribution separating original harmful-request contributions from attack-template contributions for the different head groups.

**icml64633 / gpt-5.5+nocard**
- A matched random-head or equivalent size-matched intervention control for the refusal-to-compliance flip.
- Targeted suppression of the putative persistent-signal head group on successful attacks with a quantitative before/after internal-signal measure.
- Token-level attribution separating original harmful-request contributions from attack-wrapper contributions for the two head groups.
- Explicit robustness analysis for the head-group selection boundary or an equivalent cutoff-independent grouping check.

**icml64633 / gpt-5.5+retrieval**
- A matched random-head intervention with uncertainty estimates alongside the candidate-head suppression.
- Suppression of the non-refusal/other head group on successful attacks followed by measurement of the persistent signal.
- Token-level attribution separating original harmful-request tokens from attack-template tokens for the two head groups.

**icml64633 / gpt-5.6+card**
- Intervene specifically on the heads that retain the persistent signal during successful attacks and measure the residual mid-layer signal.
- Perform token-level attribution separating original harmful-request tokens from jailbreak-template tokens.
- Expand the detector evaluation across multiple architecture families and a substantially broader set of safety benchmarks.

**icml64633 / gpt-5.6+nocard**
- A controlled suppression of the non-target/persistent-signal heads during successful attacks, with a direct measurement of the resulting signal change.
- Token-level linear attribution comparing candidate head groups on original harmful-request tokens versus jailbreak-template tokens.

**icml64633 / gpt-5.6+retrieval**
- A targeted causal suppression of the non-target head population during successful jailbreaks, with a direct mid-layer safety-signal measurement and suitable controls.
- Within-head token-level attribution separating original harmful-request tokens from jailbreak-template tokens.
- A direct mechanistic comparison showing the two head populations are similar on harmful-request tokens but diverge on attack-template tokens.
