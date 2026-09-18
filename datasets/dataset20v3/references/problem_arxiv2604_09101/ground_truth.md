# Reference experiments — problem arxiv2604_09101

**Claim.** Prompt-tuned CLIP classifiers whose image and text encoders remain frozen can be screened before deployment to distinguish clean models from backdoored models and identify the attacker-selected target class. The detection claim is accepted only if, using a fixed decision threshold of 2, an auditor achieves higher model-level AUROC and higher target-class F1 than the strongest prior model-level detector while using no greater optimization budget. A post-hoc repair is successful only if it reduces backdoor attack success rate below 10% while keeping clean top-1 accuracy within ±1 percentage point of its pre-repair value.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2604.09101

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Build the clean and backdoored model suite used for the audit. Prompt-tune CLIP ViT-B/16 with CoCoOp on the Seen classes of ImageNet, Caltech101, OxfordPets, Flowers102, Food101, FGVC Aircraft, SUN397, DTD, EuroSAT and UCF101, while freezing the image and text encoders; the two-layer meta-net produces image-conditioned tokens concatenated with four learnable context vectors and the class-name embedding. Train one clean model and four poisoned models per dataset, using BadCLIP for 10 epochs with an ℓ∞ budget of 4/255 and adapted Blended, SIBA and WaNet attacks with 10% of fine-tuning images poisoned, for 50 models total. Clean models average 82.19% Seen and 70.29% Unseen accuracy. BadCLIP averages 81.73% Seen accuracy, 99.04% Seen ASR, 68.88% Unseen accuracy and 96.70% Unseen ASR; Blended averages 81.68%, 99.73%, 67.08% and 97.65%; SIBA averages 75.91%, 66.30%, 67.58% and 52.18%; and WaNet averages 82.08%, 92.52%, 66.58% and 81.64%, respectively.

### Experiment 1

Test how the prompt-tuned backdoor is represented inside the frozen-encoder classifier. Project image embeddings and meta-tokens for clean and triggered OOD inputs from clean and poisoned prompt-tuned CLIP models, and compare how a small image perturbation changes target and non-target text embeddings. Clean and poisoned image embeddings substantially overlap, whereas poisoned-input meta-tokens form a tight cluster; in the poisoned model the perturbation produces a large shift in the target text embedding, while the corresponding shift in the clean model is negligible.

### Experiment 2

Run the model-level and target-class detection comparison. For every one of the 50 clean or backdoored models, form a candidate set of 50 labels containing the attacker-selected target, invert one class-wise trigger from 1,000 unlabeled Open Images images with no class overlap, and evaluate CLIP-Inspector, Neural Cleanse and Pixel Backdoor on the same candidate labels and OOD pool. CLIP-Inspector uses Adam with step size 0.1, batch size 32, an ℓ∞ budget of 4/255 and one epoch, approximately 32 updates, per class; Neural Cleanse and Pixel Backdoor run for five epochs with their original losses and dynamic regularization. At the fixed anomaly threshold k=2, CLIP-Inspector classifies 47 of 50 models correctly, or 94%, and its model-level AUROC is 0.973, versus 0.495 for Neural Cleanse and 0.687 for Pixel Backdoor. Its average target-class F1 across datasets is 0.92, versus 0.06 and 0.42; across attacks its F1 values are 0.94 for BadCLIP, 0.95 for Blended, 1.00 for SIBA and 0.84 for WaNet, averaging 0.93, versus 0.05 and 0.42. The three CLIP-Inspector model errors are the clean FGVC model, DTD with WaNet and EuroSAT with BadCLIP. A 50-class scan takes less than one hour on an A100 GPU, compared with six hours for Neural Cleanse and eight hours for Pixel Backdoor.

### Experiment 3

Check whether high reconstructed-trigger ASR and low optimization loss distinguish the target class rather than merely producing triggers for every class. On Caltech101, compare each method's target-class trigger metrics with the maximum among non-target classes for clean, BadCLIP, Blended, SIBA and WaNet models. CLIP-Inspector gives the clean model ASR 7.51 with difference −26.39 and loss 5.2218 with difference −2.0343; the four poisoned models give ASRs of 94.74, 99.02, 92.69 and 73.33 with differences of 66.27, 72.68, 65.75 and 50.61, and losses of −8.2344, −11.2263, −6.2807 and −4.5774. Neural Cleanse reconstructs triggers with ASR between 99.34 and 99.64 for both clean and poisoned models, while Pixel Backdoor gives ASR between 85.77 and 97.08, so their sparsity-derived scores do not show the same clean-versus-target behavior.

### Experiment 4

Choose the trigger-inversion objective by direct comparison. On BadCLIP models from ImageNet, Caltech101, UCF101 and DTD, invert triggers under identical data and optimization budgets using raw target-logit maximization, target cross-entropy or the margin between the target logit and strongest non-target logit. Raw-logit loss yields mean backdoor-class ASR 77.66% with standard deviation 26.96% and mean other-class ASR 5.37%; cross-entropy yields 81.45% ±13.48% and 5.26%; margin loss yields 91.09% ±8.73% and 4.62%, respectively.

### Experiment 5

Set the OOD inversion sample budget. On the ten BadCLIP models, run CLIP-Inspector with 100, 500 and 1,000 OOD images and measure the reconstructed trigger's target-class ASR. Average ASR is 60.00% with 100 images, 89.18% with 500 and 95.28% with 1,000; UCF101 rises from 1.00% to 53.20% to 83.70%, while ImageNet remains 100%, 100% and 99.70%. The experiments therefore use 1,000 OOD images.

### Experiment 6

Set the inversion batch size and test whether in-distribution images are required. Using 1,000 images and one epoch on the ten BadCLIP models, vary batch size among 1, 32 and 64; reconstructed-trigger target-class ASR averages 75.9%, 95.3% and 88.5%, so the experiments use batch size 32, corresponding to about 32 optimization steps. Separately replace Open Images OOD samples with in-distribution samples under the same inversion procedure; average target-class ASR is 98.6% with in-distribution images and 95.3% with OOD images.

### Experiment 7

Test detection against an attacker that narrows the trigger's activation region. Train BadCLIP Adaptive in two phases: first learn and freeze the standard BadCLIP trigger, then train on clean images with clean labels, exact-trigger images with the target label, and Gaussian-perturbed trigger images with their clean labels, using perturbation strength α=0.5 and specificity weight λspec=1. Across ten datasets, the adaptive models average 81.62% Seen accuracy, 64.07% Seen ASR, 69.30% Unseen accuracy and 48.02% Unseen ASR; CLIP-Inspector's reconstructed triggers average 69.77% ASR and anomaly score 9.45. CLIP-Inspector detects 8 of the 10 adaptive backdoored models; EuroSAT has anomaly score 1.66, below the fixed threshold of 2, and UCF101's reconstructed trigger has 30.96% ASR and is treated as invalid because it is below 50%.

### Experiment 8

Test whether the reconstructed target trigger can repair the delivered model. On the same backdoored models, select the class with the largest CLIP-Inspector anomaly score, add its reconstructed trigger to a small labeled clean downstream dataset, retain each image's ground-truth label, and perform CoCoOp-style fine-tuning. Averaged across ten datasets, BadCLIP changes from 81.76% to 81.35% clean accuracy and from 98.48% to 7.61% ASR; BadCLIP Adaptive changes from 81.62% to 81.04% and from 64.35% to 4.23%; Blended changes from 81.69% to 81.69% and from 99.73% to 7.64%; SIBA changes from 75.91% to 77.93% and from 66.30% to 5.76%; and WaNet changes from 82.08% to 82.11% and from 92.99% to 5.27%. The overall averages change from 80.61% to 80.82% clean accuracy and from 84.37% to 6.10% ASR.

### Experiment 9

Compare target-trigger repair with alternative fine-tuning interventions. Average results over all datasets and attack types using clean-only fine-tuning, random ℓ∞-bounded noise, a CLIP-Inspector trigger reconstructed for a non-target class, or the trigger reconstructed for the detected target class. Starting from 80.6% clean accuracy and 84.4% ASR, clean-only fine-tuning ends at 81.4% accuracy and 64.6% ASR, random noise at 81.0% and 52.7%, a wrong-class reconstructed trigger at 80.3% and 47.0%, and the target-class CLIP-Inspector trigger at 80.8% and 6.1%. For the wrong-class control, SIBA falls to 5.4% ASR, while BadCLIP, BadCLIP Adaptive, Blended and WaNet remain at 60.3%, 43.5%, 56.0% and 70.0%.

### Experiment 10

Test whether the detector extends to backdoors stored in the image encoder rather than the prompt meta-net. Fully fine-tune the CLIP image encoder with Blended poisoning and no prompt or meta-net, using Gaussian-noise, triangle and written-text trigger patterns, then run CLIP-Inspector unchanged with the same candidate-set size, OOD pool and ℓ∞ inversion budget. The Gaussian trigger has attacker ASR 99.7%, reconstructed-trigger ASR 98.75% and anomaly score 6.15; the triangle has 86.1%, 96.88% and 5.93; and the text trigger has 94.7%, 62.4% and 4.38. All three anomaly scores exceed the threshold of 2.

### Experiment 11

Measure whether reconstructed perturbations satisfy the intended image-level imperceptibility. Compute SSIM between clean and triggered OOD images for attacker triggers and triggers reconstructed by CLIP-Inspector, Neural Cleanse and Pixel Backdoor. For BadCLIP, the original trigger averages SSIM 0.96309 and the CLIP-Inspector reconstruction averages 0.93663, compared with 0.76556 for Neural Cleanse and 0.63006 for Pixel Backdoor. CLIP-Inspector reconstructions average SSIM 0.93447 for Blended models, 0.93739 for SIBA models and 0.93655 for WaNet models; the corresponding original triggers average 0.5057, 0.99952 and 0.9381.
