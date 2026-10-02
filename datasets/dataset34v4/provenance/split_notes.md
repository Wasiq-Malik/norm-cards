# Paper split — working notes

Read each paper's abstract/contributions against its v3 reference experiments. For each paper:
the existing claim (C0), any further claim the paper makes that has its own experiments, and
where every reference experiment goes. An experiment that only tunes or analyses the authors'
own method (hyperparameter sweeps, component ablations) is "method-internal": it supports no
claim a proposer could be asked to test, because v3 claims withhold the method.

Verdicts: **keep** = a distinct, testable claim with at least one headline experiment of its own;
**thin** = real claim, but one experiment or no numbers; **no** = not a separate claim.

## 2510_20487 — steering eval-aware models
- C0 (existing): 0–9, 11. Robustness of C0: 10, 13.
- C1 **keep** — hardening a model against deployment *prompts* does not harden it against
  *steering*: 1 (before/after expert iteration), 12. Apparatus 0, 2, 3.
- C2 **thin** — the steering vector transfers beyond the trained organism (natural emoji
  marker; simpler organism): 14, 15. Apparatus 3. 15 admits it can't separate steering from prompting.

## 2602_23523 — LIDMark deepfake watermark
- C0: 0–5.
- C1 **keep** — tracing/fidelity generalise to an unseen deepfake generator and an unseen face
  dataset without retraining: 6, 7. Apparatus 0.
- C2 **thin** — a single factorized decoder is smaller than dual-decoder systems at equal
  payload: 10 (one measurement table).
- Method-internal: 8 (loss ablation), 9 (payload expansion).

## 2604_09101 — CLIP-Inspector backdoor detection
- C0: 0, 2, 3, 8, 9.
- C1 **keep** — the detector still works against an adaptive attacker that narrows the trigger,
  and against backdoors stored in the image encoder instead of the prompt: 7, 10. Apparatus 0.
- C2 **thin** — detection needs no in-distribution data: 6 (second half), 5.
- Analysis/method-internal: 1 (where the backdoor lives — motivation), 4 (loss choice),
  11 (trigger imperceptibility).

## 2604_24474 — pretrained embedding distance
- C0: 0, 1, 2, 5. 4 (drug-likeness of high scorers) is really a confound for C0's generation arm.
- No extra claim: 3 is a diversity analysis with mixed results.

## 2604_26496 — RAAT accuracy/robustness
- C0: 3, 6, 9. NOTE 5 (arithmetic mean vs nine published methods) bears directly on C0's
  "exceeds the strongest prior" bar; the scope labels dropped it — likely wrong.
- C1 **keep** — the paper's "revealed for the first time" phenomenon: shrinking the
  perturbation on near-boundary training samples barely changes robustness, while shrinking
  it on far-from-boundary samples destroys it: 0, 1, 2. Directional only (no numbers in text).
- C2 **thin** — gains carry over from ℓ∞ to ℓ2: 4 (no numbers).
- Method-internal: 7 (component ablation), 8 (η sweep).

## 2605_08612 — ATAAT VLA backdoors
- C0: 0, 1, 2, 8, 9.
- C1 **keep** — the paper's diagnosis: conventional backdoors fail on VLAs because benign and
  backdoor gradients conflict, and removing the conflict is what makes the attack work: 3, 2
  (shared with C0), 4.
- C2 **keep** — existing defenses do not reliably remove the backdoor (preprocessing, pruning,
  safety layers, adapted STRIP/Activation Clustering): 11, 12.
- C3 **thin** — perturbations crafted on public proxy encoders transfer to the target: 5.
- Not gradeable: 10 (physical robot, qualitative). Analysis: 6, 7, 13.
