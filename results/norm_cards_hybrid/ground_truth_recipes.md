# Ground-truth experimental setups (Fable 5)

My own expert decomposition of each claim into the experiment set required to
verify or refute it, independent of the baseline / norm-card outputs. Used as the
reference to score both. Each claim is broken into a dependency chain:
**gate** (cheap prerequisite that can refute outright) → **apparatus** (thing the
claim presupposes, must be built + validated) → **headline** (direct test, all
stated constraints enforced) → **control** (rule out artifacts / attribute the
effect).

---

## Claim 11 — YOLO11n robust to a learned illumination CVAE latent-PGD attack

**What it asserts:** COCO-pretrained YOLO11n, evaluated on *person only* of COCO
val2017 at 640×640, keeps AP50 ≥ 0.90 with ≤5% relative drop, under a white-box
PGD attack **over the latent z of a CVAE g(x,z) trained on COCO** to model
*illumination-consistent* perturbations, with z norm-bounded (‖z‖ ≤ B) **and** the
image perturbation bounded L∞ ≤ 12/255.

- **G0 (gate) — Clean person-class AP50.** Eval YOLO11n on person/val2017/640×640.
  **Pass: AP50_clean ≥ 0.90, else REFUTED immediately** (PGD only degrades, so
  adv ≥ 0.90 is impossible below it). *Likely decisive:* a nano detector's COCO
  person AP50 is realistically ~0.6–0.75, so the claim very probably dies here for
  near-zero compute.
- **A1 (apparatus) — Build AND validate the CVAE.** Train g(x,z) on COCO against
  photometric/illumination augmentation targets (exposure, white balance, gamma,
  per-channel gain). Validate it is what the claim says: (i) outputs are
  illumination-like — high SSIM / low LPIPS to source, energy concentrated in low
  spatial frequencies, no high-freq adversarial speckle; (ii) latent traversals
  span diverse lighting (coverage); (iii) L∞ ≤ 12/255 and ‖z‖ ≤ B hold by
  construction. **Pass:** perturbations pass the illumination-likeness bar; else
  the downstream attack tests something other than illumination.
- **H2 (headline) — Latent PGD, both constraints enforced.** White-box PGD on z,
  **projected to ‖z‖ ≤ B every step** and image clipped to L∞ ≤ 12/255. Person
  AP50_adv. **Pass:** AP50_adv ≥ 0.90 AND 100·(clean−adv)/clean ≤ 5.
- **C3 (control) — Attack not underpowered.** (i) sweep steps/α/restarts, per-image
  worst case; (ii) **random-z in the ball vs PGD-z** — if random ≈ PGD, the
  optimizer is broken / gradient-masked (result invalid).
- **C4 (control) — Norm reference.** Standard pixel-space PGD at ε=12/255, so the
  semantic-attack result is interpretable against the field's default threat model.

---

## Claim 12 — PixMix/AugMix training robust to ≤1% global-trigger poisoning

**What it asserts:** models trained with non-geometric pixel-level augmentation
(PixMix, AugMix) resist poisoning with **global triggers** (>95% image area — noise
/ intensity / contrast) at **≤1% poison**, where *robust* ≡ post-training ASR
within **5 points** of the *same-augmentation model trained on clean data*.

- **G0 (gate) — Attack potency (defense-off).** Instantiate the global triggers as
  the canonical whole-image backdoors: **blended (Chen), sinusoidal SIG,
  intensity/contrast shift**, dirty-label single-target, on CIFAR-10 (primary) +
  one of {GTSRB, ImageNet-subset}. Train a **vanilla (no special aug)** model at
  1% poison. **Pass: undefended ASR is high (e.g. ≥ 90%), else the whole test is
  vacuous** — you cannot demonstrate a defense against an attack that doesn't work.
- **H1 (headline) — Paired same-augmentation comparison.** For each of PixMix and
  AugMix, train two models with *identical* augmentation: (a) on 1%-poisoned data,
  (b) on clean data. Measure ASR on the triggered test set + clean Top-1 for both.
  **Pass: ASR(aug+poison) − ASR(aug+clean) ≤ 5 points AND clean-accuracy drop is
  small.** Sweep poison rate {0.1, 0.5, 1%} and each trigger type. *The comparison
  baseline is the same-aug clean model — not an undefended model.*
- **C2 (control) — Attribution.** Compare against (i) no-augmentation training and
  (ii) a dedicated backdoor defense (Neural Cleanse / Spectral Signatures), to show
  the effect is augmentation, not incidental. Enforce the **>95% trigger area** and
  sweep coverage down to see where robustness breaks.
- **C3 (control) — Adaptive/aug-aware attacker.** Poison crafted knowing PixMix/
  AugMix is used (augmentation-consistent trigger). Tests whether robustness is
  real or just a mismatch between a naive trigger and the augmentation.

---

## Claim 37 — Low-rank linear latent subspaces improve constrained molecular design

**What it asserts (TWO parts):** in a SMILES VAE's continuous latent space,
(A) constraints (synthesizability, solubility) are **approximable by low-rank
linear subspaces**, and (B) using them for constrained latent optimization
improves **feasible-hits-per-oracle-query** on constrained GuacaMol similarity
tasks (Aripiprazole/Tamoxifen/Celecoxib + extra constraints) vs **black-box BO
under matched query budgets**.

- **G0 (gate) — VAE + task definition.** Train/obtain the SMILES VAE; verify latent
  usability (reconstruction accuracy, sample validity above a bar). Operationalize
  "feasible hit" = (Tanimoto similarity to target ≥ τ) AND (SA ≤ s, solubility/logP
  in range). Fix a single oracle budget B. **Pass:** VAE validity/reconstruction ≥
  bar (a broken latent space voids everything downstream).
- **A1 (apparatus) — Validate the MECHANISM (the crux, easy to skip).** Fit a
  low-rank linear map z → each constraint (linear/logistic probe; low-rank
  regression) on held-out molecules. **Pass: constraints are linearly approximable
  — feasibility AUC / R² above a bar at low rank; else part (A) is REFUTED** and the
  whole method is unfounded regardless of any optimization win. Report the rank
  needed.
- **H2 (headline) — Matched-budget optimization.** Constrained latent optimization
  steered by the low-rank subspaces vs black-box BO (GP-based), identical oracle
  budget B, on the 3 tasks, ≥5 seeds. Metric: **feasible hits per oracle query**
  (and cumulative feasible hits at B). **Pass:** method > BO by a statistically
  significant margin under matched B.
- **C3 (control) — Attribution + calibration.** (i) **Ablate the subspace** —
  constrained latent opt with a naive penalty instead of the low-rank subspace, to
  attribute any gain to the *claimed mechanism* not to constrained optimization in
  general; (ii) rank sweep; (iii) verify neither method exceeds B oracle calls
  (no budget leakage); (iv) standard PMO baselines (GraphGA, REINVENT) for external
  calibration; (v) a held-out 4th task to check the mechanism generalizes.

---

### The pattern across all three

Every claim hides an **apparatus/mechanism that must be built and validated before
the headline test is meaningful** — the CVAE (11), the attack's potency (12), the
low-rank-subspace approximability (37) — plus a **cheap gate that can refute for
almost no compute** (clean AP50; attack potency; VAE + mechanism validity), plus
**controls that attribute the effect** (random-z; no-aug/defense baselines;
subspace ablation). A recipe set that jumps straight to the headline test is
scientifically incomplete no matter how detailed that single test is.

---

## Claim 33 — Sandbagging in Llama-3 mediated by <50 attention heads

**What it asserts:** on a 1,000-problem GSM8K subset, Llama-3's *sandbagging*
(strategic underperformance) is mediated by **<50 attention heads in the final 10
layers**. Via **direct logit attribution (DLA)** to the correct−decoy logit gap,
select heads whose contribution **flips sign** between honest and sandbagging
contexts in **≥70% of prompts** (above chance) on a train/val split; **ablating or
patching** that head set yields a statistically significant reduction in the
honest−sandbagging logit gap AND accuracy gap on a **held-out** split.

- **G0 (gate) — The phenomenon exists.** Operationalize honest vs sandbagging
  prompt conditions and the correct−decoy logit-gap metric; verify Llama-3 shows a
  measurable honest−sandbagging **accuracy gap AND logit gap** on the GSM8K subset.
  **Pass: a significant gap exists; else the claim is vacuous** (nothing to mediate).
- **A1 (apparatus) — Validate the DLA instrument.** DLA must faithfully decompose
  the correct−decoy logit gap into per-head contributions. **Pass:** summed
  per-head (+ MLP/embed) DLA contributions reconstruct the actual logit gap within
  tolerance (completeness/faithfulness), else the attribution is not trustworthy.
- **H2 (headline) — Select, count, intervene, on held-out.** On train/val, select
  heads whose DLA sign flips in ≥70% of prompts; **count them (claim: <50) and
  confirm they lie in the final 10 layers**; ablate/patch the set and measure the
  reduction in honest−sandbagging logit gap AND accuracy gap on the **held-out**
  split, with a proper significance test. Enforce all three: <50, final-10-layers,
  train/val→test split (no selection-on-test leakage).
- **C3a (control) — Specificity vs random heads.** Ablate a **random / layer-matched
  head set of the same size**; the selected set must reduce the gap **significantly
  more** than random, else the effect isn't specific to these heads.
- **C3b (control) — Necessity + sufficiency + statistics.** Ablation reduces the gap
  (necessity) AND patching honest-context activations through these heads into the
  sandbagging run restores honest behavior (sufficiency); handle multiple
  comparisons (selecting from all heads) with a chance/permutation baseline for the
  70% flip threshold.

---

## Claim 36 — Relational facts localized in Llama-3 MLPs enable knowledge editing

**What it asserts:** in Llama-3, binary relational facts (subject, relation, object)
are **localized in identifiable MLP weights and attention modules in intermediate
layers** acting as dedicated per-fact storage; **causal tracing + activation
analysis** isolate these elements precisely enough to enable knowledge editing with
**reliability AND generality > 90% on the UniEdit benchmark** and **collateral < 5%**
(impact on MMLU).

- **G0 (gate) — Operationalize + non-vacuity.** Define the fact set (s,r,o triples),
  reliability (edit success), generality (paraphrase success), locality, and
  collateral (Δ MMLU) on UniEdit, with the >90% / <5% thresholds. Verify Llama-3
  **already knows the facts pre-edit** (you cannot edit a fact the model lacks).
  **Pass:** model recalls the facts and the metrics are well-defined; else vacuous.
- **A1 (apparatus) — Validate LOCALIZATION is causal.** The claim's crux is
  "dedicated storage units": run causal tracing (ROME-style) to identify the MLP/
  attention site, and **validate faithfulness** — corrupting/ablating the located
  element removes the fact (necessity) and restoring it recovers it, and the site is
  in intermediate layers as claimed. **Pass:** localization is causal and
  concentrated, not diffuse; else "dedicated storage" is refuted.
- **H2 (headline) — Edit at the located site, joint thresholds.** Apply knowledge
  editing (ROME/MEMIT-style) at the localized elements over **many held-out
  UniEdit facts**; pass only if **reliability > 90% AND generality > 90% jointly**
  AND **collateral (Δ MMLU) < 5%**. Enforce all three together, not averaged.
- **C3a (control) — Locality/specificity.** Unrelated facts (neighborhood triples)
  are preserved post-edit — the standard KE locality metric — so collateral is more
  than an aggregate MMLU number.
- **C3b (control) — Localization NECESSITY (the crux critique).** Editing the
  **causally-localized** site must beat editing a **random / non-causal-layer** site
  of equal capacity at matched budget. If random-site editing matches it,
  localization does **not** inform editing (the Hase et al. critique) and the
  "dedicated storage enables targeted editing" claim is refuted even if edits
  succeed.

---

## Claim 6/7 (shared) — ground truth from the SPECIFIED version (#6)

**#6 and #7 are the same claim.** #7 is the underspecified input a verification
agent receives; **#6 is the author's specified version**, so decomposing #6 gives a
more objective ground truth than inferring one from the vague #7. The test: given
the underspecified **#7** as input, does the norm-card method recover the concrete
elements #6 states explicitly, better than the baseline?

**#6 (specified):** a "risk-attention" module (K added attention heads + a learned
risk-summary token r + a bounded gate gt∈[0,1]) reweights next-token logits away
from policy-violation-inducing tokens; **for ≥3 model families** it reduces the
per-response policy-violation rate by **≥50%** while keeping **aggregate capability
within 2%** of baseline on a **benchmark suite (MMLU, GSM8K, MT-Bench)** under the
**same decoding policy**.

- **G0 (gate) — Operationalize + non-vacuous baseline.** Define the violation
  taxonomy + a **detection method** (LLM-judge / safety classifier) and an
  **elicitation benchmark**; measure the **unmodified** per-response violation rate
  under a **fixed decoding policy**. **Pass:** baseline violation rate materially
  > 0 (else a 50% cut is vacuous); capability baseline on **MMLU/GSM8K/MT-Bench**
  recorded. *(#6 specifies the suite + "same decoding policy".)*
- **A1 (apparatus) — Build + validate the risk-attention module.** Instantiate the
  **K heads + risk-summary token r + bounded gate gt∈[0,1]**; validate the risk
  signal discriminates violation-inducing tokens (AUC above chance) and the gate
  down-weights them. **Pass:** mechanism functional + discriminative. *(#6 specifies
  the module structure.)*
- **H2 (headline) — Joint reduction + capability, ≥3 families.** Per-response
  violation rate **↓ ≥50%** vs unmodified **AND** aggregate capability **within 2%**
  on MMLU/GSM8K/MT-Bench, under the **same decoding policy**, replicated across
  **≥3 model families**. Enforce all jointly. *(All four are #6 specifics #7 omits.)*
- **C3a (control) — Over-refusal / helpfulness.** The cut must not come from
  **refusing everything**: measure over-refusal on benign prompts (XSTest-style) +
  benign-task compliance. *(Distinct from aggregate capability.)*
- **C3b (control) — Attribution + generality.** **Ablate the risk-attention module**
  (the reduction is due to it, not incidental), and the **≥3-model-family**
  replication is the generality control (not one lucky model); optionally adaptive
  jailbreaks, not just the static benchmark.

*What #7 omits that #6 supplies (= what the norm card must recover from literature):*
the **benchmark suite**, the **violation-detection metric**, the **≤2% capability
constraint**, the **≥3-family generality** requirement, and the **fixed decoding
policy**. Scoring #7's baseline vs method against this list measures directly
whether grounding fills an underspecified claim's gaps.
