# Reference experiments — problem arxiv2605_08612

**Claim.** Training-time visual backdoors are a practical supply-chain vulnerability for vision-language-action manipulation policies under both restricted data-poisoning and direct model-fine-tuning access. The question is whether such backdoors can force an attacker-specified robotic action without materially impairing untriggered behavior, while generalizing across semantically equivalent instructions and avoiding activation in neutral contexts. The claim is accepted if, with no more than 5% poisoned training data, targeted attack success exceeds 80% and benign task success remains within 5% relative of a clean-trained policy; synonym replacement or syntactic restructuring reduces targeted attack success by fewer than 5 percentage points; neutral-context activation remains below 8%; and, under direct fine-tuning access, the backdoor exceeds the strongest prior attack in both targeted attack success and retained benign task success under an equal access and training budget.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2605.08612

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Test whether restricted 5% data poisoning can implant a targeted backdoor while retaining benign manipulation performance. Replace 5% of LIBERO training samples, corresponding to 1–3 poisoned trajectories per task, with ATAAT Implicit samples containing a visible yellow “UR” sticky-note trigger and an L-infinity-bounded orthogonal perturbation of 8/255 generated for 10 PGD iterations with step size 1/255. Fine-tune OpenVLA-7B through rank-32 LoRA using AdamW, learning rate 1e-5 and batch size 32. On LIBERO-Object, ATAAT obtains 90.1% benign Task Success Rate and 85.9% Targeted Attack Success Rate; BadNet obtains 5.2% and 9.3%, Latent-Poisoning 14.8% and 1.4%, and adapted BadVLA 16.1% and 12.8%. On LIBERO-Spatial, ATAAT obtains 88.8% benign Task Success Rate and 83.5% Targeted Attack Success Rate; BadNet obtains 4.5% and 4.5%, Latent-Poisoning 13.6% and 10.1%, and adapted BadVLA 17.5% and 13.1%.

### Experiment 1

Test whether direct model fine-tuning can implant the backdoor by updating only dormant parameters. Analyze benign activations in OpenVLA-7B, use activation threshold 1e-3 to select approximately 1.8% of tunable parameters, freeze the remaining parameters, and fine-tune the selected parameters on 200 anchoring samples through rank-32 LoRA with AdamW, learning rate 1e-5 and batch size 32. On LIBERO-Object, ATAAT Explicit obtains 79.3% benign Task Success Rate and 74.8% Targeted Attack Success Rate, compared with 8.8% and 5.9% for fine-tuned BadNet and 50.8% and 37.7% for adapted BadVLA. On LIBERO-Spatial, ATAAT Explicit obtains 78.1% benign Task Success Rate and 72.5% Targeted Attack Success Rate, compared with 9.1% and 6.4% for fine-tuned BadNet and 52.1% and 39.2% for adapted BadVLA.

### Experiment 2

Check whether the low baseline success rates reflect failure to optimize the continuous action policy. Train a benign OpenVLA-7B baseline, BadNet, adapted BadVLA and ATAAT on LIBERO-Spatial, then record final training loss, validation action mean-squared error, Task Success Rate and the dominant observed failure mode. The benign baseline has final loss 0.12, action MSE 0.028 and 91.5% Task Success Rate. BadNet has loss 1.68, MSE 0.412 and 4.5% Task Success Rate, with severe stagnation; adapted BadVLA has loss 1.42, MSE 0.385 and 17.5% Task Success Rate, with repetitive jittering; ATAAT has loss 0.15, MSE 0.034 and 88.8% Task Success Rate, with occasional execution errors.

### Experiment 3

Measure whether ATAAT changes the gradient conflict observed during backdoor training. During LoRA fine-tuning of OpenVLA-7B, calculate at each training stage the cosine similarity between benign-task and backdoor-task gradients over trainable adapter parameters, and compare adapted BadVLA with ATAAT over approximately 1,000 steps. Adapted BadVLA rapidly enters the negative range, oscillates between about -0.2 and -0.5, and stabilizes near -0.4 after roughly 400 steps. ATAAT remains near zero throughout training, with at most weak positive similarity.

### Experiment 4

Test whether both parts of the implicit composite trigger are required. Evaluate full ATAAT Implicit and versions omitting either the orthogonal perturbation or the visible trigger on LIBERO-10. Full ATAAT obtains 89.4% benign Task Success Rate and 84.7% Targeted Attack Success Rate. Removing the orthogonal perturbation yields 88.1% benign Task Success Rate and 3.2% Targeted Attack Success Rate; removing the visible trigger yields 89.9% benign Task Success Rate and 0.5% Targeted Attack Success Rate.

### Experiment 5

Test whether implicit perturbations transfer from different public proxy encoders to OpenVLA-7B. Generate the perturbations with CLIP ViT-L/14, SigLIP-SO400M, ViT-B/16 or ResNet-50 and evaluate on LIBERO-Spatial. CLIP ViT-L/14 gives 88.8% benign Task Success Rate and 83.5% Targeted Attack Success Rate; SigLIP-SO400M gives 86.2% and 81.4%; ViT-B/16 gives 87.1% and 22.7%; and ResNet-50 gives 89.0% and 14.2%.

### Experiment 6

Test whether the explicit dormant-neuron allocation corresponds to a small parameter subset rather than the full tunable backbone. Run activation analysis on benign probe data with threshold 1e-3 and measure the fraction of dormant tunable neurons across OpenVLA-7B layer groups. The selected fractions are 0.5% in shallow layers L0–L10, 2.1% in middle layers L11–L21 and 3.2% in deep layers L22–L31, for 1.8% of the overall tunable backbone; the frozen vision encoder is not included.

### Experiment 7

Quantify whether the implicit poisoned images remain perceptually close to their clean sources. Compare original clean training samples with samples containing the implicit perturbation using LPIPS and SSIM. The poisoned samples obtain LPIPS 0.045 ± 0.015 and SSIM 0.912 ± 0.027 relative to the clean samples.

### Experiment 8

Test whether the backdoor follows instruction meaning across language rewrites. Starting from LIBERO-Spatial training instructions, use GPT-4o under prompt constraints to construct Set A by replacing verbs or nouns while retaining syntax and Set B by changing syntax or adding modifiers while preserving intent; evaluate without model re-tuning. Adapted BadVLA falls from 13.1% Targeted Attack Success Rate on original instructions to 8.5% on synonym replacements, a 4.6-point drop, and 4.2% on syntactic restructurings, an 8.9-point drop. ATAAT falls from 83.5% to 81.2%, a 2.3-point drop, and to 79.4%, a 4.1-point drop. BadNet remains at 0.0% on all three sets.

### Experiment 9

Check whether the visual trigger disrupts an unrelated benign instruction. Evaluate a benign “Pickup black bowl” task without the red cup, the same instruction with a red cup present as a neutral trigger, and the triggered “Pickup red cup” hijack condition. BadVLA records 95.1% benign Task Success Rate, 71.5% Task Success Rate with the neutral trigger and 93.2% Targeted Attack Success Rate on the hijack instruction. ATAAT records 95.3% benign Task Success Rate, 92.1% Task Success Rate with the neutral trigger and 94.5% Targeted Attack Success Rate on the hijack instruction.

### Experiment 10

Test the attacks on a physical robot using object, interaction and semantic triggers. For implicit de-confliction, present a green mineral-water bottle, a no-handle cup and a person pointing with two hands; the robot executes the malicious action for each trigger. For explicit de-confliction, present an open drawer, crossed cutlery and a person wearing a watch; the robot executes the malicious action for each high-level semantic condition. The paper reports these outcomes qualitatively and does not give trial counts or physical-world success rates.

### Experiment 11

Test the explicit semantic backdoor against input preprocessing, parameter intervention and safety defenses on the “Pickup red cup” split. The undefended Targeted Attack Success Rate is 94.5%. It becomes 91.6% under JPEG compression, 87.9% under Gaussian noise, 73.4% under neuron pruning, 78.5% under RoboGuard, 65.3% under SafeVLA and 45.2% under Circuit Breakers. In accompanying physical demonstrations, JPEG compression still permits the malicious action, RoboGuard halts it during execution, and Circuit Breakers blocks it before an action is taken.

### Experiment 12

Test conventional backdoor detectors adapted to continuous robotic control. On LIBERO-Spatial, adapt STRIP by overlaying varied visual inputs and measuring action-output entropy, and adapt Activation Clustering by applying two-means clustering to latent representations. With no defense, ATAAT has 83.5% Targeted Attack Success Rate. Adapted STRIP reduces it to 76.2% but produces a 62.5% false-positive rate on benign samples; adapted Activation Clustering reduces it to 27.1% and produces an 18.4% benign false-positive rate.

### Experiment 13

Measure the physical cost of failed or mis-triggered behavior. Compute cumulative cost as the sum over a trajectory of joint-torque overload, excessive end-effector velocity and collision penalties. A normal benign-task failure has average cumulative cost 15.2, an ATAAT generalization failure has 18.5, and a BadVLA trigger-induced failure has 150.7.
