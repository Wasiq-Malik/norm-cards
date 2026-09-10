# Norm-card ablation — `gpt-5.4`, `gpt-5.5`, `gpt-5.6`

Judge `gpt-5.6-luna`, k=1/3, tools on. 3 claim(s), 6 arm(s).

**Recall** — of the experiments the source team actually ran, how many would a team running the proposed set have effectively performed?

For each reference experiment the judge asks whether the proposed set would establish what it establishes. `covered` = 1.0 (by any route — the method does not have to match), `partial` = 0.5 (gets at it, outcome stays ambiguous), `missing` = 0. Recall is the mean. Mapping is many-to-many, so a short proposal is not penalised for being short.

> Judge self-test passed on 3/3 claims: scoring each reference against itself returns recall 1.00.

> **⚠️ 1 arm(s) judged against an older version of their reference** — arxiv2510_20487/gpt-5.4+nocard. Re-run those before quoting them: `python -m norm_cards.eval.evaluate --problems <id> --force`.

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
| arxiv2510_20487 | gpt-5.5+card | **0.62** | 3 / 4 / 1 | 8 |
| arxiv2510_20487 | gpt-5.5+nocard | **0.88** | 6 / 2 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+card | **0.94** | 7 / 1 / 0 | 8 |
| arxiv2510_20487 | gpt-5.6+nocard | **1.00** | 8 / 0 / 0 | 8 |
| icml61272 | gpt-5.4+card | **0.83** | 4 / 2 / 0 | 6 |
| icml61272 | gpt-5.4+nocard | **0.67** | 2 / 4 / 0 | 6 |
| icml61272 | gpt-5.5+card | **0.67** | 2 / 4 / 0 | 6 |
| icml61272 | gpt-5.5+nocard | **0.58** | 2 / 3 / 1 | 6 |
| icml61272 | gpt-5.6+card | **0.67** | 3 / 2 / 1 | 6 |
| icml61272 | gpt-5.6+nocard | **0.58** | 2 / 3 / 1 | 6 |
| icml64633 | gpt-5.4+card | **0.64** | 3 / 3 / 1 | 7 |
| icml64633 | gpt-5.4+nocard | **0.79** | 5 / 1 / 1 | 7 |
| icml64633 | gpt-5.5+card | **0.71** | 4 / 2 / 1 | 7 |
| icml64633 | gpt-5.5+nocard | **0.71** | 4 / 2 / 1 | 7 |
| icml64633 | gpt-5.6+card | **0.86** | 6 / 0 / 1 | 7 |
| icml64633 | gpt-5.6+nocard | **0.79** | 5 / 0 / 1 | 7 |

## Did the norm card help?

Same model, same claim, same subclaims — the only difference is whether the norm card was in `current_evidence`. Positive Δ means it helped.

### By claim (averaged over models)

| Claim | n models | recall without | recall with | Δ |
|---|---|---|---|---|
| arxiv2510_20487 | 3 | 0.94 | 0.83 | -0.10 |
| icml61272 | 3 | 0.61 | 0.72 | +0.11 |
| icml64633 | 3 | 0.76 | 0.74 | -0.02 |
| **all** | **9** | **0.77** | **0.76** | **-0.01** |

### By model and claim

| Claim | Model | recall without | recall with | Δ |
|---|---|---|---|---|
| arxiv2510_20487 | gpt-5.4 | 0.94 | 0.94 | 0.00 |
| arxiv2510_20487 | gpt-5.5 | 0.88 | 0.62 | -0.25 |
| arxiv2510_20487 | gpt-5.6 | 1.00 | 0.94 | -0.06 |
| icml61272 | gpt-5.4 | 0.67 | 0.83 | +0.17 |
| icml61272 | gpt-5.5 | 0.58 | 0.67 | +0.08 |
| icml61272 | gpt-5.6 | 0.58 | 0.67 | +0.08 |
| icml64633 | gpt-5.4 | 0.79 | 0.64 | -0.14 |
| icml64633 | gpt-5.5 | 0.71 | 0.71 | 0.00 |
| icml64633 | gpt-5.6 | 0.79 | 0.86 | +0.07 |

**Mean Δ = -0.006** over 9 paired run(s) — helped 4, hurt 3, tied 2. 95% CI [-0.090, +0.079]. **The interval contains zero: no detectable effect.**

One coverage step on a 7-experiment reference moves recall by 0.071, so that is the smallest difference this can resolve. A mean below it bounds the effect rather than measuring it.

## Which experiments were covered

`C` covered · `p` partial · `-` missing. Columns are reference experiment indices; see each claim's `ground_truth.md` for what they are.

**arxiv2510_20487**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| gpt-5.4+card | C | C | C | C | C | C | C | p |
| gpt-5.4+nocard | C | C | C | C | C | C | C | p |
| gpt-5.5+card | C | p | p | C | C | - | p | p |
| gpt-5.5+nocard | C | C | C | C | C | p | C | p |
| gpt-5.6+card | C | C | C | C | C | p | C | C |
| gpt-5.6+nocard | C | C | C | C | C | C | C | C |

**icml61272**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| gpt-5.4+card | C | p | C | C | C | p |
| gpt-5.4+nocard | C | p | p | p | C | p |
| gpt-5.5+card | C | p | p | p | C | p |
| gpt-5.5+nocard | C | - | p | p | C | p |
| gpt-5.6+card | C | - | C | p | C | p |
| gpt-5.6+nocard | C | - | p | p | C | p |

**icml64633**

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| gpt-5.4+card | C | C | p | p | p | - | C |
| gpt-5.4+nocard | C | C | C | C | p | - | C |
| gpt-5.5+card | C | C | p | C | p | - | C |
| gpt-5.5+nocard | C | C | p | C | p | - | C |
| gpt-5.6+card | C | C | C | C | C | - | C |
| gpt-5.6+nocard | C | C | C | C | ? | - | C |

## What the proposals missed

The judge's account of what each set would still need.

**arxiv2510_20487 / gpt-5.4+card**
- Add a behaviorally matched control in which the user is described as being evaluated while the model's own deployment status is unchanged, and assess both status judgments and target safety behavior under steering.
- Predefine quantitative criteria for “matches deployment,” “substantially intact,” and the steering-versus-prompting comparison.

**arxiv2510_20487 / gpt-5.4+nocard**
- A source-model or otherwise independent-vector control showing that the steering direction cannot contain the target behavior, together with the same intervention tested on the evaluation-aware target.
- A direct mechanism test linking the intervention-induced status judgment to the behavioral shift, such as causal mediation or transcript evidence supplemented by the explicit user-evaluation lexical control and relevant training-stage robustness checks.

**arxiv2510_20487 / gpt-5.5+card**
- Derive the intervention direction from an independently pretrained version before introducing evaluation-conditioned target behavior, or otherwise provide an equivalent control for target-behavior encoding.
- Run magnitude-matched random or nonspecific activation perturbation controls under the same evaluation tasks.
- Use a stronger, multi-prompt deployment baseline and a hardened evaluation-aware model, and directly validate the model's expressed deployment/evaluation assessment with an appropriate held-out behavioral or trace measure.

**arxiv2510_20487 / gpt-5.5+nocard**
- Use random steering vectors matched to the real vector's norm and distribution, with a prespecified aggregate comparison.
- Add controls that discriminate causal deployment-status belief from correlated hidden-state changes, especially a user-is-being-evaluated lexical control and, if the model is hardened in stages, tests after each stage and with vectors extracted from both relevant model states.

**arxiv2510_20487 / gpt-5.6+card**
- Use an ensemble of independently drawn, norm- and distribution-matched random vectors with a preregistered aggregate specificity criterion.
- Include several deployment-prompt variants, including individually applied and combined contrastive prompts or an equivalent prompt-search/control protocol, to make the prompting-failure comparison robust rather than dependent on one wording.

**icml61272 / gpt-5.4+card**
- Require both semantic-judge and deterministic token-level scoring and predefine how disagreement is interpreted.
- Repeat the core sweep on materially larger models and explicit event-level knowledge, with high- versus low-confidence fact strata.
- Perform systematic hyperparameter and architecture/configuration sensitivity tests, especially for methods that appear resilient.

**icml61272 / gpt-5.4+nocard**
- An explicit clean-baseline and same-output cross-validation using both semantic and deterministic token-level scoring, with a prespecified disagreement rule.
- A cumulative geometric analysis that measures normalized directional amplification after each edit-count level and tests its low-variance enrichment and monotonic trend.
- A systematic robustness sweep over larger model scales, knowledge types, fact-confidence strata, architectures, and editing hyperparameters/configurations.

**icml61272 / gpt-5.5+card**
- An independent semantic-versus-deterministic token-level scoring cross-check with a prespecified disagreement rule.
- Layerwise directional amplification measured repeatedly across sequential edit counts and across more than one architecture.
- A scale/model-family sweep, explicit event-level knowledge condition, fact-difficulty or confidence stratification, and hyperparameter/architecture sensitivity analysis for exceptional editors.

**icml61272 / gpt-5.5+nocard**
- A deterministic token-level versus semantic-judge scoring cross-check with an explicit agreement test.
- Mechanistic geometry measurements replicated across the relevant editor families and model architectures, not only one selected pair.
- A broader comparative sweep including potentially robust parameter-editing methods and consistent parameter-free comparisons.
- Robustness tests over larger model scales, event-level knowledge, fact-confidence strata, and editor hyperparameters/architecture sensitivity.

**icml61272 / gpt-5.6+card**
- An explicit clean-baseline and single-edit paired comparison under plain autoregressive decoding versus teacher forcing.
- A same-outcome cross-check between semantic judging and deterministic token-level scoring across reliability, generalization, locality, and portability.
- Broader coverage of editing families and architectures in the cumulative comparison, rather than only the selected best parameter method.
- Robustness sweeps over larger model scales, event-level knowledge, fact-confidence strata, and hyperparameter/architecture sensitivity of the exceptional method.

**icml61272 / gpt-5.6+nocard**
- Add an independent semantic-versus-deterministic scoring cross-check on the same outputs and a prespecified rule for disagreement.
- Repeat the low-variance directional amplification analysis at each sequential edit count and across the relevant editing methods and architectures.
- Broaden the efficacy sweep to more representative editing families and architectures, and test larger scales, event-level knowledge, explicit difficulty strata, and hyperparameter/configuration sensitivity including possible non-collapsing exceptions.

**icml64633 / gpt-5.4+card**
- Token-level attribution or equivalent token-isolating interventions comparing original harmful-request tokens with jailbreak-template tokens for the two head populations.
- A matched-size random-head intervention, with repeated trials and uncertainty estimates, alongside targeted suppression.
- Direct suppression of the complementary/non-target heads during successful attacks with quantitative measurement of the remaining safety-relevant activation.

**icml64633 / gpt-5.4+nocard**
- Add an explicit component-scoring and two-category classification procedure with threshold and bandwidth/smoothing sensitivity analyses.
- Suppress the attack-robust/other head population during successful attacks and measure the resulting mid-layer signal change against appropriate references.
- Perform per-token attribution that separates original harmful-request contributions from jailbreak-template contributions for each head type.

**icml64633 / gpt-5.5+card**
- Add a token-resolved activation attribution experiment that separately aggregates contributions from original harmful-request tokens and jailbreak-template tokens, and compares the identified head groups on both sources.

**icml64633 / gpt-5.5+nocard**
- Token-level attribution separating original harmful-request tokens from attack-template tokens, with a comparison of the candidate and other head populations.
- Suppression of the non-candidate/other heads during successful attacks while measuring the persistent safety-relevant signal.
- An explicit and cutoff-robust head-scoring and classification procedure.

**icml64633 / gpt-5.6+card**
- Token-level attribution separating original harmful-request tokens from attack-template tokens for the two head groups.
- An explicit successful-jailbreak condition in which the non-target head group is suppressed and the persistent internal safety signal is measured directly, with an appropriate control.

**icml64633 / gpt-5.6+nocard**
- Measure the safety-relevant signal in MLPs as well as attention heads across layers under matched successful and unsuccessful/unattacked conditions.
- Suppress the complementary persistent-head population during successful jailbreaks and measure the change in the persistent mid-layer signal.
- Perform token-level linear attribution separating original harmful-request tokens from attack-template tokens for the two head populations.

---

# Deep dive: what did the norm card actually add?

One model sampled per claim, comparing the same model's carded and un-carded proposals
against the reference, item by item.

## Summary

| Claim | Model | Δ recall | Card items used: no-card → card | New from card | Verdict |
|---|---|---|---|---|---|
| `icml61272` | gpt-5.4 | **+0.17** | 25 → 33 | 17 | Card supplied genuinely new evaluation resources, and the proposer used them |
| `icml64633` | gpt-5.4 | **−0.14** | 3 → 23 | 20 | Heavy uptake, but the carded set **dropped the controls** the un-carded one had |
| `arxiv2510_20487` | gpt-5.5 | **−0.25** | 16 → 23 | 10 | Same pattern: card content in, random-vector control out |

## The one case where the card helped

**`icml61272` / gpt-5.4, 0.67 → 0.83.** New because of the card: `MQuAKE`,
`MQuAKE-CF`, `MQuAKE-T`, `RippleEdits`, `COUNTERFACT+`, `SERAC`,
`Portability (Subject Replace / Reversed Relation / One-hop)`,
`Score (harmonic mean of Efficacy, Generalization, Specificity)`,
`Report confidence intervals`.

These are not generic. Multi-hop editing benchmarks and the portability sub-tests are
specific to knowledge editing, and the un-carded proposal named none of them. Two
reference experiments moved `partial` → `covered`:

- **ref 2 (mechanism)** — the carded version added covariance-eigenspectrum estimation
  and equal-norm synthetic perturbations as a causal contrast.
- **ref 3 (main benchmark)** — the carded version spans "multiple factual-editing
  datasets, models, evaluation dimensions, and single/sequential loads up to 1000
  edits", where the un-carded one "names only a few representative parameter methods,
  permits one model".

So the card widened scope on a claim whose reference demands breadth. This is the
clearest evidence in the run that a card can add something outside parametric memory.

## The two cases where it hurt — and they fail the same way

**`icml64633` / gpt-5.4, 0.79 → 0.64.** Uptake was massive: the un-carded proposal used
3 card items, the carded one 23 — `ASR`, `Detoxify`, `Perspective API`, `MT-bench`,
`ForbiddenQuestionSet`, `MaliciousInstruct`, `path patching`, `DistilBERT safety
classifier`, `Qwen2.5-7B-Instruct`. The card was read and used.

Two reference experiments went `covered` → `partial`, and the judge's reason is the
same both times:

> **ref 3, no-card:** "It includes random matched-size heads, heads selected from
> harmless prompts, and benign-prompt side-effect controls."
>
> **ref 3, card:** "But it has no explicit matched random-head or size-matched control
> and no stated repeated-seed confidence analysis."

The un-carded proposal had the controls. The carded one did not.

**`arxiv2510_20487` / gpt-5.5, 0.88 → 0.62.** Identical shape. The un-carded proposal
listed "random-vector/placebo steering" as one of four conditions; the carded one has
no such control, and reference 5 went `partial` → `missing`:

> "No proposed experiment applies matched random or otherwise non-semantic steering
> vectors and compares their behavioral effects."

## Why: the card is resource-shaped, not design-shaped

| Claim | Total items | Datasets + models + metrics | Protocols |
|---|---|---|---|
| `arxiv2510_20487` | 228 | 181 (79%) | 47 (21%) |
| `icml61272` | 276 | 238 (86%) | 38 (14%) |
| `icml64633` | 351 | 263 (75%) | 88 (25%) |

And almost none of the protocols encode experimental **design**:

```
arxiv2510_20487   1 of 47 protocols mentions a control
icml61272         1 of 38
icml64633         6 of 88
```

A norm card is a menu of *nouns* — what the field measures things on. It carries almost
nothing about what a sound experiment needs: matched controls, placebo conditions,
seeds, size-matched interventions. The proposer has a **3-experiment budget**, and when
handed 200–350 resources it spends that budget naming and organising them. The controls
are what get squeezed out.

That is a sharper claim than "the card adds nothing beyond parametric knowledge". On
`icml61272` it demonstrably did add something. But because what it adds is resources
rather than design, it can **displace** good experimental practice the model would have
produced on its own.

## Was the proposer mishandled by the judge anywhere?

One case, worth discounting. On `arxiv2510_20487` ref 2 the judge marked the carded arm
down because the steering vector is "extracted from the already evaluation-aware model
... no independent pre-fine-tuning extraction". **The claim does not require that** —
it was deliberately removed, and the source paper's own appendix shows extraction from
the final model works. The judge is scoring the reference's method rather than the
claim's requirement, which the claim-is-the-contract rule exists to prevent.

The judge's phrasing does raise a real methodological worry (a direction extracted after
fine-tuning "may be entangled with the fine-tuned safety behavior"), so this is a soft
error rather than a clear one. Discount roughly one step of that arm's −0.25.

The other status changes hold up on inspection: the carded proposals really did omit
the controls they were marked down for.

## What this implies

1. **The card's ceiling is set by what it contains.** Resources, not design. Expect help
   where a claim needs breadth of evaluation (knowledge editing) and harm where it needs
   controls (steering, mechanistic interpretability).
2. **The 3-experiment cap makes it zero-sum.** Card content and control conditions
   compete for the same budget. Widening the cap would test whether the card's content
   is additive or genuinely displacing.
3. **The obvious next experiment** is a card containing protocol and control norms
   rather than resource lists, tested on the same claims. That is a change to the
   pipeline, not to the eval.
