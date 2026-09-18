# Reference experiments — problem arxiv2608_23830

**Claim.** Exploration bias toward already-easy constraints is an optimization bottleneck in reinforcement learning for language models required to satisfy multiple instructions in one response. The claim is supported only if a policy trained to counteract this bias has variation in per-instruction success and variance in pairwise co-satisfaction each lower by more than 0 absolute units than an otherwise matched equal-reward RL policy, while its mean prompt-level all-instructions-pass accuracy is higher by more than 0 percentage points. Independently, each imbalance measure is a valid diagnostic only if, at fixed mean instruction accuracy, its correlation with prompt-level all-instructions-pass accuracy is negative (r < 0).

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2608.23830

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Establish that cumulative-reward RL preferentially improves already-easy instructions. Train Qwen3-1.7B with GRPO and the cumulative reward equal to the number of satisfied instructions on IFTrain prompts containing up to five instructions, then compare instruction-type accuracy before and after RL. Categorize instruction types as Easy, Medium, or Hard according to baseline performance. Easy instructions receive more consistent improvements, Medium and Hard instructions have higher-variance gains, and instruction types with very low initial accuracy benefit minimally from RL.

### Experiment 1

Validate VIA and VSA as diagnostics independently of mean instruction accuracy. Use multi-instruction test prompts from AdvancedIF and aggregate responses from GPT-5.4, GPT-5.4-mini, GPT-5, GPT-4o, GPT-OSS-120B, GPT-OSS-20B, Qwen-3.5-35B-A3B, Qwen-3.5-9B, Qwen-3-32B, Qwen-3-8B, Qwen-3-1.7B, Olmo-3.1-32B-Instruct, Olmo-3-7B-Instruct, Llama-3.2-3B, and Llama-3.1-8B. Repeatedly sample response groups of K=128 and retain groups whose instruction accuracy is 0.68±0.01, the mean across the evaluated models. For each group, form a binary response-by-instruction verification matrix, compute Variation of Instruction Accuracy as the variance of marginal instruction satisfaction probabilities, compute Variance of Synergy Accuracy as the variance across pairwise joint-satisfaction probability divided by the smaller marginal probability, and compare both with prompt accuracy. At fixed instruction accuracy, both VIA and VSA have an inverse relationship with prompt accuracy; the paper does not report correlation coefficients.

### Experiment 2

Build the behaviorally bootstrapped policies used in the RL comparisons. Evaluate each initial policy on the 45,374-prompt IFTrain subset containing three to five rule-based instructions, identify instruction types with accuracy below τ=0.5, and collect up to N_target=100 successful responses for each selected type, producing D_seed=900. Sample candidate responses, retain responses satisfying the target instruction, and choose the valid response covering the most additional instructions; when no valid response is generated, append “Focus on fulfilling [Instruction] first” and sample again. Supervised fine-tune the initial policy on the collected data for one epoch before RL.

### Experiment 3

Train the matched cumulative-reward and exploration-bias-mitigation variants. Starting from Qwen3-1.7B and Qwen2.5-7B-Instruct, train on the 45,374-prompt IFTrain subset with GRPO, response-group size 16, total batch size 512, learning rate 1×10^-6, and at most 400 steps on four NVIDIA A100 GPUs. Compare ordinary cumulative reward (CR), Behavioral Bootstrapping followed by cumulative reward (BeBoot-CR), and Behavioral Bootstrapping followed by Scarcity-Aware Rewards (BeBoot-SaR). For each rollout, the scarcity-aware unary term weights a satisfied instruction by 1 minus its empirical satisfaction frequency in the response group; the binary term weights each jointly satisfied pair by 1 minus its joint count divided by the smaller individual success count, with balance factor α=2.

### Experiment 4

Test whether Behavioral Bootstrapping and Scarcity-Aware Rewards improve prompt-level instruction following over cumulative-reward RL. Generate with temperature 0.7 and evaluate Qwen3-1.7B variants on IFBench, IFEval, and Multi-IF. Qwen1.7B-CR scores 41.1 loose and 35.1 strict on IFBench, 86.8 loose and 84.1 strict on IFEval, and 81.2, 59.2, and 46.7 on Multi-IF turns 1–3, for an overall average of 62.0. Qwen1.7B-BeBoot-CR scores 43.1, 39.2, 90.0, 88.3, 78.3, 61.5, and 48.9, averaging 64.2. Qwen1.7B-BeBoot-SaR scores 50.1, 44.3, 90.7, 89.0, 85.4, 62.9, and 50.4, averaging 67.5. Relative to CR, BeBoot-SaR raises the average by 5.5 points and IFBench strict accuracy by 9.2 points; BeBoot-CR raises the average by 2.2 points.

### Experiment 5

Repeat the policy comparison at the 7B scale. Under the same training and evaluation setup, Qwen7B-CR scores 42.4 loose and 39.5 strict on IFBench, 91.1 loose and 90.0 strict on IFEval, and 88.0, 68.2, and 55.9 on Multi-IF turns 1–3, averaging 67.9. Qwen7B-BeBoot-CR scores 48.4, 43.1, 92.6, 91.2, 84.7, 68.3, and 57.7, averaging 69.4. Qwen7B-BeBoot-SaR scores 52.6, 48.4, 94.0, 92.9, 86.5, 69.3, and 56.6, averaging 71.5. Relative to CR, BeBoot-SaR raises the average by 3.6 points and IFBench strict accuracy by 8.9 points; BeBoot-CR raises the average by 1.5 points.

### Experiment 6

Test whether the accuracy gains coincide with reduced exploration imbalance during training. For each training prompt and response group, track VIA, VSA, Instruction Coverage Rate—the proportion of instructions satisfied at least once in the group—and prompt accuracy for CR, BeBoot-based training, and BeBoot+SaR. CR reduces VIA and VSA slowly but retains higher imbalance. BeBoot variants begin with higher instruction coverage and prompt accuracy, and BeBoot+SaR produces the fastest and largest declines in VIA and VSA, reaches the lowest variance among the tested methods, and finishes with higher instruction coverage and prompt accuracy. The paper does not provide numerical endpoint values for these curves.

### Experiment 7

Locate which instruction types account for the training-set changes. Decompose pass-rate shifts by instruction type for CR, BeBoot, and BeBoot+SaR. BeBoot and SaR improve more than CR on the combination, copy, and length_constraint categories, and combining BeBoot with SaR performs better than BeBoot alone. Every tested model has a 0% pass rate on the new category.

### Experiment 8

Check whether the mitigation changes policy entropy during RL. Track policy entropy for the cumulative-reward baseline and the proposed training variants. Entropy collapses for the baseline, while the proposed method maintains higher entropy during training; exact entropy values are not reported.

### Experiment 9

Compare the trained small models with reproduced off-the-shelf baselines. Evaluate all models with temperature 0.7, using non-thinking mode for Qwen3 and default medium thinking effort for GPT-OSS. The seven-score averages across IFBench loose and strict, IFEval loose and strict, and Multi-IF turns 1–3 are 70.2 for GPT-OSS-120B, 63.7 for Llama-3.1-70B, 64.8 for Qwen3-32B, 63.0 for Gemma-3-27B, 56.5 for GPT-OSS-20B, 63.7 for Qwen3-8B, 30.5 for Mistral-7B-Instruct-v0.3, and 56.3 for Gemma-3-4B. The trained Qwen1.7B-BeBoot-SaR and Qwen7B-BeBoot-SaR models average 67.5 and 71.5, respectively.

### Experiment 10

Determine how Behavioral Bootstrapping depends on instruction-selection threshold and seed-data size. Fix rollout sample size at K=32, compare τ=0.1 for hard instructions, τ=0.5 for hard and medium instructions, and τ=1.0 for all instructions, and vary N_target between 100 and 500 examples per selected instruction type before subsequent RL. The τ=0.5 setting consistently performs better than τ=0.1 and τ=1.0, and N_target=100 generally performs better than N_target=500; exact benchmark scores are not reported.

### Experiment 11

Determine whether unary scarcity rewards suffice and how much binary synergy reward to use. Train BeBoot+SaR while varying the balance factor α on the binary term. With α=0, using only unary scarcity rewards, performance exceeds BeBoot+CR. Performance is stable from α=0.0 to α=0.5 and then rises to its reported peak at α=2.0; larger values reduce performance through the paper’s described over-optimization of instruction combinations. Exact scores for the sweep are not reported.

### Experiment 12

Test sensitivity to the response-group size used to estimate scarcity. Train the scarcity-aware method and corresponding baselines with response-group sizes K=8, 16, and 32. The scarcity-aware method improves over the corresponding baseline at every tested group size. Larger groups converge faster but show over-optimization; exact convergence rates and final scores are not reported.
