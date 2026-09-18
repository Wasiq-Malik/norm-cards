# Reference experiments — problem arxiv2607_19374

**Claim.** Automated conversion of text-only plane-geometry problems into standard Lean 4/MATHLIB theorem statements can yield semantically faithful formalizations that are compatible with general-purpose neural theorem provers. Semantic faithfulness requires the formal statement, including any explicit non-degeneracy and configuration assumptions, to preserve the meaning of the informal problem. The proposition is supported if expert review finds a correct first compiling candidate for at least 50% of problems or a correct candidate among five attempts for at least 50%, an unadapted general-purpose prover achieves greater than 0% single-attempt proof success, and training on the formalizations improves that success rate by more than 0 percentage points over the same unadapted checkpoint.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2607.19374

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Build the competition-level formalization set used for semantic evaluation. Filter OMNI-Math by domain to obtain 780 plane-geometry problems with difficulty scores from 1.0 to 9.5, including problems from IMO, USAMO, national olympiads, and regional competitions. Use DeepSeek-V3 to generate 32 candidate Lean formalizations per problem with the complete four-stage EUCLEAN pipeline—constraint explication, prove-first configuration anchoring, formalization mapping, and compiler-feedback iterative repair—and retain problems having at least one compiling statement. This produces OMNI-Geometry with 768 of 780 problems retained, a 98.5% retention rate.

### Experiment 1

Build the large formalization corpus used for prover inference and training. Filter NuminaMath with problem_type == Geometry to obtain 183,796 candidate problems, apply the complete EUCLEAN pipeline with DeepSeek-V3, and retain each problem having at least one compiling formal statement after compiler-guided repair. This produces Numina-Geometry with 177,597 problems, retaining 96.6% of the candidates.

### Experiment 2

Measure which pipeline additions produce compiling Lean statements. On 30 OMNI-Geometry problems, sample 32 independent generations per configuration, giving 960 candidates for each of five cumulative configurations. Basic prompting yields 125 compiling candidates, or 13.0%; adding concept formalization mapping yields 254, or 26.5%; adding iterative code repair yields 465, or 48.4%; adding constraint formalization mapping yields 500, or 52.1%; and adding prove-first configuration anchoring yields 480, or 50.0%.

### Experiment 3

Test whether prove-first configuration anchoring improves semantic faithfulness despite lowering the compilation count. On the same 30 ablation problems, conduct a paired human Pass@3 comparison between formalizations with and without anchoring. Anchoring wins on 7 problems and loses on 3, while both versions are correct on 10 and both are incorrect on 10.

### Experiment 4

Have experts judge whether the compiling formalizations preserve the informal problems. Randomly sample 180 OMNI-Geometry problems with difficulty at least 4.0, prepare five compiling formalizations per problem, and have graduate students experienced in formal mathematics and Lean 4 label semantic consistency with the original statement. Count TOP1 when the first candidate is correct and TOP5 when at least one of five is correct. Overall, 88 of 180 first candidates are correct, or 48.89%, and 132 of 180 problems have a correct candidate among five, or 73.33%. By difficulty, TOP1 and TOP5 are 27/53 (50.94%) and 41/53 (77.36%) for 4.0–5.0, 43/77 (55.84%) and 59/77 (76.62%) for 5.0–6.0, 7/16 (43.75%) and 12/16 (75.00%) for 6.0–7.0, 8/19 (42.11%) and 11/19 (57.89%) for 7.0–8.0, and 3/15 (20.00%) and 9/15 (60.00%) for 8.0+.

### Experiment 5

Test whether an unadapted general-purpose Lean prover can prove the generated geometry statements. Run the 8-billion-parameter Goedel v2 base checkpoint, which was fine-tuned for general Lean 4 theorem proving but receives no geometry-specific training here, on all 177,597 Numina-Geometry problems and measure Pass@1, the percentage solved with one generation attempt. The base model achieves 13.6% Pass@1.

### Experiment 6

Test whether proofs collected from Numina-Geometry improve the same prover checkpoint. Starting separately from the Goedel v2 base checkpoint, train SFT on all successful proof traces collected with one generation attempt per problem and train DPO on successful-versus-failed pairs collected with two attempts per problem; DPO uses 7,148 preference pairs. Both runs use one node with eight NVIDIA A100-40GB GPUs, per-device batch size 1, gradient accumulation 32, effective batch size 256, one epoch, cosine scheduling, 0.05 warmup, BF16, ZeRO-3, and FlashAttention v2. SFT uses learning rate 1×10^-4 and maximum sequence length 4096; DPO uses learning rate 1×10^-5, beta 0.1, and maximum sequence length 2048. On all 177,597 Numina-Geometry problems, Goedel v2 improves from 13.6% Pass@1 to 15.1% after SFT and to 15.0% after DPO.

### Experiment 7

Check whether the Goedel v2 improvement exceeds variation between independent base-model inference runs. Two base inference runs obtain 13.62% and 13.58% Pass@1. The SFT and DPO models obtain 15.11% and 15.02%, respectively; paired problem-level counts are 5,635 gains versus 2,988 losses for SFT and 5,338 gains versus 2,853 losses for DPO.

### Experiment 8

Test whether training on Numina-Geometry transfers to a held-out competition set. Use OMNI-Geometry only for evaluation and not in any training stage, and compare the Goedel v2 base, SFT, and DPO checkpoints. Pass@1 is 6.90% for the base model, 7.68% after SFT, and 7.81% after DPO; Pass@2 is 8.33%, 9.11%, and 8.59%, respectively.

### Experiment 9

Test whether the downstream training signal transfers to another general-purpose prover. Randomly sample half of Numina-Geometry, use DeepSeek-Prover-V2-7B itself to collect successful proof traces on that half, and fine-tune for two epochs with the same SFT recipe used for Goedel v2 except for the epoch count. Evaluate Pass@1 on the same half-corpus split. DeepSeek-Prover-V2-7B improves from 13.4% to 15.0%.

### Experiment 10

Test a frontier prover on a random subset of the generated corpus. Run Aristotle on 100 randomly sampled formalized problems. After excluding statements Aristotle explicitly judges false and refutes, statements made trivial by obvious misformalization, and statements with concrete counterexamples, Aristotle proves 25 problems and leaves another 19 unresolved but unrefuted.
