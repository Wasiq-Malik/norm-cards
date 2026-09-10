# Norm-card ablation — `gpt-5.4`, `gpt-5.5`, `gpt-5.6`

Judge `gpt-5.6-luna`, k=1/3, tools on. 3 claim(s), 9 arm(s).

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
| arxiv2510_20487 | gpt-5.4+retrieval | **0.88** | 6 / 2 / 0 | 8 |
| arxiv2510_20487 | gpt-5.5+card | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.5+nocard | **0.88** | 6 / 2 / 0 | 8 |
| arxiv2510_20487 | gpt-5.5+retrieval | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+card | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+nocard | **1.00** | 8 / 0 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+retrieval | **1.00** | 8 / 0 / 0 | 8 |
| icml61272 | gpt-5.4+card | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.4+nocard | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.4+retrieval | **0.83** | 4 / 2 / 0 | 6 |
| icml61272 | gpt-5.5+card | **0.83** | 4 / 2 / 0 | 6 |
| icml61272 | gpt-5.5+nocard | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.5+retrieval | **0.58** | 2 / 3 / 1 | 6 |
| icml61272 | gpt-5.6+card | **0.83** | 4 / 2 / 0 | 6 |
| icml61272 | gpt-5.6+nocard | **0.75** | 3 / 3 / 0 | 6 |
| icml61272 | gpt-5.6+retrieval | **0.75** | 3 / 3 / 0 | 6 |
| icml64633 | gpt-5.4+card | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.4+nocard | **0.79** | 5 / 1 / 1 | 7 |
| icml64633 | gpt-5.4+retrieval | **0.79** | 5 / 1 / 1 | 7 |
| icml64633 | gpt-5.5+card | **0.86** | 6 / 0 / 1 | 7 |
| icml64633 | gpt-5.5+nocard | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.5+retrieval | **0.71** | 4 / 2 / 1 | 7 |
| icml64633 | gpt-5.6+card | **0.86** | 5 / 2 / 0 | 7 |
| icml64633 | gpt-5.6+nocard | **0.93** | 6 / 1 / 0 | 7 |
| icml64633 | gpt-5.6+retrieval | **0.86** | 5 / 2 / 0 | 7 |

## Did the norm card help?

Same model, same claim, same subclaims — conditions differ only in what sits in `current_evidence`. Each Δ is against `nocard`, paired run by run.

`retrieval` is the control that matters for the project's thesis: a norm card is a synthesis over retrieved papers, so beating `nocard` only shows that relevant literature helps. Beating `retrieval` is what shows the synthesis is doing work.

### By claim (averaged over models)

| Claim | n models | nocard | retrieval | card | Δ retrieval | Δ card |
|---|---|---|---|---|---|---|
| arxiv2510_20487 | 3 | 0.94 | 0.94 | 0.94 | 0.00 | +0.00 |
| icml61272 | 3 | 0.75 | 0.72 | 0.81 | -0.03 | +0.06 |
| icml64633 | 3 | 0.86 | 0.79 | 0.86 | -0.07 | -0.00 |
| **all** | **9** | **0.85** | **0.82** | **0.87** | **-0.03** | **+0.02** |

### By model and claim

| Claim | Model | nocard | retrieval | card | Δ retrieval | Δ card |
|---|---|---|---|---|---|---|
| arxiv2510_20487 | gpt-5.4 | 0.94 | 0.88 | 0.94 | -0.06 | 0.00 |
| arxiv2510_20487 | gpt-5.5 | 0.88 | 0.94 | 0.94 | +0.06 | +0.06 |
| arxiv2510_20487 | gpt-5.6 | 1.00 | 1.00 | 0.94 | 0.00 | -0.06 |
| icml61272 | gpt-5.4 | 0.75 | 0.83 | 0.75 | +0.08 | 0.00 |
| icml61272 | gpt-5.5 | 0.75 | 0.58 | 0.83 | -0.17 | +0.08 |
| icml61272 | gpt-5.6 | 0.75 | 0.75 | 0.83 | 0.00 | +0.08 |
| icml64633 | gpt-5.4 | 0.79 | 0.79 | 0.86 | 0.00 | +0.07 |
| icml64633 | gpt-5.5 | 0.86 | 0.71 | 0.86 | -0.14 | 0.00 |
| icml64633 | gpt-5.6 | 0.93 | 0.86 | 0.86 | -0.07 | -0.07 |

`retrieval` vs `nocard`: **mean Δ = -0.033** over 9 paired run(s) — helped 2, hurt 4, tied 3. 95% CI [-0.089, +0.023]. **Interval contains zero: no detectable effect.**

`card` vs `nocard`: **mean Δ = +0.018** over 9 paired run(s) — helped 4, hurt 2, tied 3. 95% CI [-0.021, +0.058]. **Interval contains zero: no detectable effect.**

One coverage step on a 7-experiment reference moves recall by 0.071, so that is the smallest difference this can resolve. A mean below it bounds the effect rather than measuring it.

## Where the misses are

Coverage grouped by what each reference experiment is FOR, pooled across claims and models. This is the diagnostic a recall number cannot give: it says which KIND of experiment goes missing, and so which part of a norm card would have to change to fix it.

| What the experiment is for | refs | nocard | retrieval | card |
|---|---|---|---|---|
| **apparatus** — build/validate what the headline test presupposes | 7 | 0.93 | 0.88 | 0.93 |
| **headline** — the direct test of the claim | 4 | 0.92 | 0.88 | 1.00 |
| **mechanism** — show the asserted cause is the operative one | 3 | 0.78 | 0.72 | 0.78 |
| **control** — rule out that the result is an artifact of intervening at all | 3 | 0.94 | 1.00 | 0.94 |
| **confound** — kill a rival explanation under which the result looks the same | 3 | 0.61 | 0.50 | 0.61 |
| **external** — show the finding generalises beyond its own setup | 1 | 0.83 | 1.00 | 0.83 |

## Which experiments were covered

`C` covered · `p` partial · `-` missing. Columns are reference experiment indices; see each claim's `ground_truth.md` for what they are.

**arxiv2510_20487**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| gpt-5.4+card | C | C | C | C | C | C | C | p |
| gpt-5.4+nocard | C | C | C | C | C | C | C | p |
| gpt-5.4+retrieval | C | C | C | C | C | C | p | p |
| gpt-5.5+card | C | C | C | C | C | C | C | p |
| gpt-5.5+nocard | C | C | C | C | C | p | C | p |
| gpt-5.5+retrieval | C | C | C | C | C | C | C | p |
| gpt-5.6+card | C | C | C | C | C | p | C | C |
| gpt-5.6+nocard | C | C | C | C | C | C | C | C |
| gpt-5.6+retrieval | C | C | C | C | C | C | C | C |

**icml61272**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| gpt-5.4+card | C | p | p | C | C | p |
| gpt-5.4+nocard | C | p | p | C | C | p |
| gpt-5.4+retrieval | C | p | C | C | C | p |
| gpt-5.5+card | C | p | C | C | C | p |
| gpt-5.5+nocard | C | p | p | C | C | p |
| gpt-5.5+retrieval | p | - | p | C | C | p |
| gpt-5.6+card | C | p | C | C | C | p |
| gpt-5.6+nocard | C | p | C | p | C | p |
| gpt-5.6+retrieval | C | p | C | p | C | p |

**icml64633**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| gpt-5.4+card | C | C | C | C | C | p | p |
| gpt-5.4+nocard | C | C | C | C | C | - | p |
| gpt-5.4+retrieval | C | C | C | p | C | - | C |
| gpt-5.5+card | C | C | C | C | C | - | C |
| gpt-5.5+nocard | C | C | C | p | C | p | C |
| gpt-5.5+retrieval | C | C | C | p | p | - | C |
| gpt-5.6+card | C | C | C | C | p | p | C |
| gpt-5.6+nocard | C | C | C | C | C | p | C |
| gpt-5.6+retrieval | C | C | C | C | p | p | C |

## What the proposals missed

The judge's account of what each set would still need.

**arxiv2510_20487 / gpt-5.4+card**
- Add a direct belief measurement, such as blinded classification of reasoning traces or another independently validated status-belief measure, and show that its change tracks the behavioral change.
- Add a control where the user, rather than the model, is described as being evaluated, plus robustness across training stages or other model-hardening conditions.

**arxiv2510_20487 / gpt-5.4+nocard**
- A semantically controlled control that places evaluation language on the user or task rather than making the model itself the evaluated party.
- A more direct belief-mechanism test, such as reasoning-trace evidence tied to deployment-status deliberation, plus robustness of the interpretation after model adaptation and with independently extracted steering vectors.

**arxiv2510_20487 / gpt-5.4+retrieval**
- Test a battery of deployment-side textual prompts, including individually optimized prompts and a combined prompt condition, under the same evaluation cue.
- Add a context control in which evaluation language refers to another party, and validate the internal status readout as a measure of the model's belief or link its change to the behavioral effect with a stronger causal mediation/control analysis.

**arxiv2510_20487 / gpt-5.5+card**
- Add a pre-registered, independently validated deployment-status measure and a control where evaluation language refers to the user rather than the model, while holding the target safety task fixed.
- Preferably add a mediation or orthogonal-behavior test showing that the status shift predicts the behavioral change beyond direct effects of the intervention.

**arxiv2510_20487 / gpt-5.5+nocard**
- Use random null vectors explicitly matched to the real vector's norm, mean, variance, and application pattern.
- Add causal or otherwise stronger mechanism controls, including a user-is-being-evaluated condition, direct reasoning/status evidence, and robustness checks after hardening and with a vector extracted from the final evaluation-aware model.

**arxiv2510_20487 / gpt-5.5+retrieval**
- Add an independent direct measure of the model's own deployment-status belief (for example, held-out reasoning-trace or forced-choice contextual reports) under steering.
- Add a lexical/context control in which evaluation language is present but applies to the user or another entity rather than indicating that the model itself is being evaluated, and compare the behavioral and internal-status effects.

**arxiv2510_20487 / gpt-5.6+card**
- Use multiple independently sampled random vectors matched to the intervention's norm (and preferably the reference direction's scale/distribution) and compare their full behavioral distribution with the genuine vector.
- Optionally include a task-level capability test matched to the exact safety-relevant task or response format on which the steering effect is reported, in addition to broad downstream utility benchmarks.

**icml61272 / gpt-5.4+card**
- A paired semantic-judge versus deterministic token-level evaluation with an explicit agreement/ranking decision rule.
- Directional geometric measurements on the same cumulative edit prefixes, including normalized variance-relative amplification and a monotonicity test.
- Replication across larger models, event-level facts, explicit fact-difficulty/confidence strata, and sensitivity analysis of the strongest apparent parameter-editing exception.

**icml61272 / gpt-5.4+nocard**
- Cross-check semantic editing scores against deterministic token-level correctness across all editing dimensions.
- Repeat the variance-aware cumulative geometry analysis across multiple editing methods and model architectures with an explicit edit-count/monotonicity test.
- Add model-scale, fact-confidence/difficulty, knowledge-type, hyperparameter, and architecture-sensitivity sweeps.

**icml61272 / gpt-5.4+retrieval**
- An explicit semantic-versus-deterministic scoring validation for the editing metrics and rankings.
- A robustness sweep over larger model scales, event-level knowledge, fact-confidence/difficulty strata, and method hyperparameters or architecture compatibility.

**icml61272 / gpt-5.5+card**
- Explicit plain autoregressive, non-teacher-forced scoring for the single-edit comparison.
- A deterministic token-level match-ratio/exact-match validation of the semantic evaluation.
- A robustness sweep over substantially larger models, event-level knowledge, explicit fact-difficulty strata, and hyperparameter/architecture sensitivity.

**icml61272 / gpt-5.5+nocard**
- A semantic-judge versus deterministic token-level scoring cross-check with an explicit disagreement rule.
- Repeated geometric directional-amplification measurements at each edit count, including the claimed monotonic low-variance concentration pattern.
- Explicit larger-model, event-level, confidence-stratified, and systematic hyperparameter/architecture robustness sweeps.

**icml61272 / gpt-5.5+retrieval**
- An explicit unedited baseline and plain-autoregressive versus teacher-forced single-edit comparison.
- A paired semantic-judge and deterministic token-level score validation on identical outputs.
- A cumulative, direction-normalized geometry analysis across edit counts, methods, and architectures.
- A deliberate robustness sweep over larger model scales, knowledge types, fact difficulty/confidence, and parameter-editing hyperparameters or architectures.

**icml61272 / gpt-5.6+card**
- A paired semantic-judge and deterministic token-level scoring validation with an explicit agreement or ranking decision rule.
- A robustness sweep over substantially larger models, a distinct event-level knowledge representation, high- versus low-confidence facts, and hyperparameter/architecture sensitivity of any editor that appears stable.

**icml61272 / gpt-5.6+nocard**
- A semantic-correctness evaluation cross-checked against deterministic token-level match or exact-match scoring across all editing dimensions.
- A planned robustness sweep over materially different model scales, event-level versus triple-shaped knowledge, fact-confidence or difficulty strata, and hyperparameter/architecture sensitivity of the strongest parameter editor.
- A broader cross-family comparison in which the parameter-free method is compared against the full set of representative parameter-editing families rather than only a selected strongest editor.

**icml61272 / gpt-5.6+retrieval**
- Replicate the main comparison and geometric mechanism across multiple architectures and at larger model scales, rather than only one model.
- Vary knowledge representation and difficulty, including event-level facts and confidence-stratified subsets, and perform method hyperparameter/architecture sensitivity analysis.
- Specify and run an independent semantic-judge versus deterministic token-level scoring cross-check with an explicit disagreement rule.

**icml64633 / gpt-5.4+card**
- Token-level attribution separating original harmful-request content from jailbreak-template content.
- Cross-model, cross-architecture and broader safety-benchmark evaluation of the same training-free detector.

**icml64633 / gpt-5.4+nocard**
- Token-level attribution or matched controls that separately measure original harmful-request tokens versus jailbreak-template tokens for the relevant heads.
- Broader held-out detector validation across multiple safety-aligned models and architecture families and a sufficiently diverse standardized safety-benchmark suite.

**icml64633 / gpt-5.4+retrieval**
- A matched random-head (or otherwise size- and magnitude-matched null) intervention with repeated seeds and uncertainty estimates for the refusal-to-compliance flip.
- Token-level attribution of head activations, separately comparing original harmful-request tokens with attack-template tokens for the localized head groups.

**icml64633 / gpt-5.5+nocard**
- Add a matched random-head (and ideally magnitude-matched null) suppression arm, with prespecified head-count curves and uncertainty, to attribute refusal-to-compliance flips specifically to the candidate subset.
- Add token-level linear attribution or an equivalent intervention that separately tests original harmful-request tokens and jailbreak-template tokens for the two head populations.
- Suppress the putative persistent-signal head population on successful jailbreak inputs and measure the resulting mid-layer safety-signal change, rather than relying only on a readout after candidate-head ablation.

**icml64633 / gpt-5.5+retrieval**
- Add a matched random-head or nonspecific head-intervention control with repeated seeds and uncertainty estimates for the refusal-to-compliance flip.
- Suppress the persistent non-suppressed head population during successful attacks and directly measure the resulting safety-related mid-layer activation strength.
- Add token-resolved attribution that separately compares original harmful-request tokens with attack-template tokens for the two head populations.

**icml64633 / gpt-5.6+card**
- An explicit intervention on the complementary persistent heads during successful jailbreaks, with measurement of the residual safety-relevant signal.
- Token-level attribution separating original harmful-request tokens from attack-template tokens for both head types.
- A broader leakage-controlled detector evaluation spanning multiple architecture families and a diverse safety-benchmark suite.

**icml64633 / gpt-5.6+nocard**
- A prespecified sensitivity analysis over alternative head-selection thresholds and other scoring/calibration choices.
- Token-level attribution or an equivalent intervention that separates the original harmful request from jailbreak-template contributions to the two head groups.
