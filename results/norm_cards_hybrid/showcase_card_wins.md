# Showcase: where the norm card made the experiment better (for slides)

Two claims, one experiment each — the clearest case where the norm-card method
proposed a *materially better, more runnable* experiment than the no-card baseline.
Same claim, same decomposition, same model (gpt-5.5); the **only** difference is the
norm card. Every card-supplied term below is a **literal menu item** (provenance
verified in `problem_X/norm_card.json`).

---

## ⭐ FLAGSHIP — Claim 9 (learned robot dynamics + non-vacuous error bound)

**Claim's central deliverable:** a learning procedure that outputs a dynamics model
*and* a **non-vacuous calibrated error bound** e(x,u) holding with confidence ≥1−δ.

**Experiment 1 — the headline coverage + non-vacuity test.** Both arms run the same
repeated-trials coverage test on MuJoCo. The difference is *how they produce the
error bound* — the crux of the claim:

> **Baseline (no card):** "…for many independent training datasets of size T, train
> **the proposed learning procedure** to output f̂ and e(x,u)."
> → *leaves the bound-construction method completely unspecified — the hardest and
> most claim-critical step is a black box.*

> **Method (with card):** "Train candidate learning procedures such as **GP
> regression, Lipschitz-kernel regression, and neural-network ensembles; use a
> held-out calibration split or PAC/conformal method** to output both f̂ and a
> pointwise error bound e(x,u)."
> → *names the field's actual estimators AND the certification framework that
> produces a distribution-free bound.*

**Menu provenance (literal items in the #9 card):** `PAC-Bayes learning from a single
trajectory`, `PAC-Bayesian error term`, `Proportional learning error bounds`,
`Cramér-Rao lower bound`, `Gaussian Process Regression / kriging`, `Kernel smoothing
estimators`, `Subspace methods (N4SID, MOESP)`.

**Bonus — Experiment 3 is the only experiment in all 10 claims that cites the card by
name:** *"With respect to evidence piece **scientific_norm_card**, especially its
System Identification and Statistical Learning Theory entries listing … PAC/conformal-
style error assessment, test whether the guarantee holds on a public dataset such as
Minari/D4RL…"* — the baseline never proposed this real-data validation.

**One-line takeaway for the slide:** *The norm card turned an under-specified
experiment into a runnable, method-correct one — supplying the field's actual bound-
certification framework (PAC-Bayes / conformal) for the claim's central object, which
the baseline left as an unnamed "learning procedure."*

---

## Claim 21 (HMM surrogate of an LLM + DFA-constrained decoding)

**Experiment 2 — evaluating constrained generation.** Both arms build the HMM×DFA
product and verify exactness. The difference is *what they evaluate on*:

> **Baseline (no card):** defines ad-hoc toy constraints — "a specific keyword,
> forbidding toxic words from **a public lexicon**, a regex-like template" — and
> generic quality checks.
> → *hand-built constraints; not how the field measures controlled generation.*

> **Method (with card):** "Use **public CommonGen examples**… define precise DFA
> constraints such as 'must contain all required keywords'… measure **constraint/
> keyword coverage**… similarity using **BLEU/BERTScore/Self-BLEU**."
> → *grounds the evaluation in the subfield's standard controlled-generation
> benchmark and metrics.*

**Menu provenance (literal items in the #21 card):** `CommonGen`,
`ContextualizedCommonGen (C2GEN)`, `Constraint or keyword coverage`,
`NeuroLogic concept-coverage evaluation`, `BLEU`, `Self-BLEU`, `BERTScore`,
`targeted distillation`.

**One-line takeaway for the slide:** *The norm card made the constrained-generation
test use the field's real benchmark (CommonGen) and coverage metrics instead of
hand-built toy constraints — a citeable, comparable evaluation.*

---

## Honest framing (so the claim survives scrutiny)

- The baseline is **not wrong** — it's *under-specified* on the crux step. The card's
  value is making the experiment **concrete, runnable, and grounded in the field's
  actual apparatus** (PAC-Bayes bounds for #9; CommonGen + coverage for #21).
- This effect showed up **only in the obscure fields** (system-ID, constrained
  decoding) — exactly where the model's parametric knowledge is thinnest. On
  mainstream/niche-but-known claims (safety, poisoning, knowledge-editing) the
  baseline already names the right tools, so the card adds nothing.
- Caveat to state up front: n=1, temp=1. Recommend re-running #9 and #21 at n≥3 to
  confirm the win is stable and not a single lucky draw before it goes in the deck.
