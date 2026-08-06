# Reference recipe — problem 11

**Claim.** An Ultralytics YOLO11n detector pretrained on COCO and evaluated on the person class of the COCO val2017 split at 640×640 resolution,retains AP50 ≥ 0.90 (IoU ≥ 0.5), with ≤ 5% relative degradation from clean performance under adversarial perturbations under image modifications under a learned generative illumination-consistent perturbation model. The image manipulations follow a white-box projected gradient descent (PGD) attack over the latent variables of the learned conditional generative model, with perturbations constrained to remain within an L∞ image-space bound of ε = 12/255.
Note that the conditional generator is learned from the COCO data set to describe the perturbation set over a constrained latent space of lighting parameters. The generator g(x,z) produces perturbed images (for original image x) where z lies in a norm-bounded latent space (e.g., \|z\| \le B ); g is a trained conditional variational autoencoder (CVAE).

**Provenance.** llm_generated · gpt-5.6-sol @ xhigh effort — machine-written, unverified by a human

## What is asserted

- The detector is the official Ultralytics YOLO11n object detector with COCO-pretrained weights; it is not fine-tuned or adversarially trained.
- Evaluation is bounding-box detection on the person class only (Ultralytics class 0 / COCO person category) over all 5,000 COCO val2017 images, at Ultralytics imgsz=640 preprocessing.
- Clean person AP50 is the baseline C, computed at IoU=0.50 with the standard confidence-ranked COCO precision-recall protocol.
- For a conditional CVAE g(x,z) learned from COCO data to model illumination changes, the allowed latent variables obey a predeclared norm bound ||z||<=B.
- A white-box PGD adversary differentiates through the frozen CVAE and YOLO11n to select a separate latent code for each validation image that minimizes person detection performance.
- Every attacked image x_adv must be in the generator range for x, within the latent bound, and satisfy ||x_adv-x||_infinity <= 12/255 in the same [0,1] RGB image coordinates before YOLO letterboxing/normalization.
- On the attacked val2017 set, person AP50 A must satisfy both A>=0.90 and relative degradation (C-A)/C<=0.05, equivalently A>=max(0.90,0.95C).

**Underspecified / reading adopted:**
- The statement's “AP50 >=0.90 (IoU >=0.5)” is read as ordinary person AP at exactly IoU 0.50, not AP averaged over IoU>=0.50. The canonical COCO API's AP50 entry uses iouThr=.5.
- “At 640x640” is read as Ultralytics imgsz=640 preprocessing (aspect-ratio-preserving resize plus padding/rectangular batching), not geometric warping of every source to a square. Attack constraints are measured before YOLO preprocessing so padding cannot count as perturbation.
- The claim says the generator is learned “from COCO,” but standard COCO has ordinary single observations and no paired relighting labels. The canonical learned-illumination CVAE protocol requires paired perturbed observations. I therefore adopt the strongest charitable, testable reading: create COCO-derived paired illumination data for generator training without using val2017 (e.g. physically relight/recapture train2017 scenes or attach validated lighting interventions), while retaining the COCO train images as conditions. Merely applying hand-authored synthetic transforms would not establish that a CVAE learned the true illumination set from COCO. If no such paired COCO-derived data can be produced and validated, the named apparatus—and thus the claim as written—fails.
- The norm type for ||z||<=B is omitted. The canonical multi-illumination CVAE uses a pre-reparameterization L2 ball, so the recipe uses L2 and chooses B from a held-out generator-validation encoding-norm quantile before any YOLO val attack. A different preregistered norm with equivalent held-out validation could be valid, but must not be selected using detector robustness.
- The detector attack loss is omitted and AP itself is nondifferentiable. Use detector-native differentiable person-vanishing/objectness and matched-target detection-loss objectives, then keep, per image, the feasible candidate with the greatest preregistered attack loss; use objective/iteration/restart sweeps as a control.
- It is unclear whether AP is Ultralytics' internal interpolated AP or official pycocotools COCO AP. Use exported detections and pycocotools COCOeval restricted to person, IoU=.50, all areas, maxDets=100 as primary; report Ultralytics classwise mAP50 as a cross-check.
- The attack is read as per-image, untargeted worst-case person suppression, not one universal z, since a standard adversarial robustness claim quantifies independently over each input.
- The 5% degradation is interpreted as relative, exactly as written: (C-A)/C, not an absolute five percentage-point drop.
- Pixel values are floating-point RGB in [0,1]. Enforce epsilon before any 8-bit save/reload and evaluate directly; if images must be serialized, re-check the bound afterward.

## Field norms (as established from sources)

**datasets**
- `COCO train2017 / val2017` — The claim names COCO and Ultralytics' official dataset configuration defines the canonical train/validation split and person class. Generator fitting must not use val2017, which is the headline test set.  
  <sub>https://docs.ultralytics.com/datasets/detect/coco; https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml</sub>
- `Paired multi-illumination scenes as the generator-validation norm` — A learned illumination perturbation set needs multiple lighting observations of the same content. The standard primary work uses scene-disjoint multi-illumination pairs and the official resource contains over 1,000 scenes with 25 lights. This is the appropriate held-out apparatus benchmark even though the claim unusually insists on COCO-derived generator training.  
  <sub>arXiv:2007.08450; https://projects.csail.mit.edu/illumination/; arXiv:2007.08450</sub>

**models**
- `Official COCO-pretrained Ultralytics YOLO11n detection checkpoint` — This is the exact named model family/artifact. Official documentation lists YOLO11n at 640 pixels and reports all-class COCO AP50:95=39.5, confirming the artifact but not the claimed person AP50.  
  <sub>https://docs.ultralytics.com/models/yolo11</sub>
- `Conditional VAE learned perturbation set` — The primary learned-perturbation-set work defines exactly a conditional generator over a norm-bounded latent ball and applies it to real illumination variation.  
  <sub>arXiv:2007.08450; arXiv:2007.08450</sub>

**metrics**
- `Person bounding-box AP50` — Ultralytics defines mAP50 as AP at IoU=.50; pycocotools provides the canonical confidence-ranked, 101-recall-point COCO evaluator. Restrict its category list to person and report stats[1]/the corresponding precision tensor slice.  
  <sub>https://docs.ultralytics.com/guides/yolo-performance-metrics; https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py; https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py</sub>
- `CVAE approximation error and expected approximation error on held-out lighting pairs` — The primary learned-set work uses latent-PGD approximation error to test whether real perturbations lie in the learned set, and expected approximation error to test likelihood/coverage. Its multi-illumination results provide comparison values.  
  <sub>arXiv:2007.08450; arXiv:2007.08450</sub>
- `Relative degradation` — This is claim-defined rather than a separately standardized benchmark metric. Compute D=(C-A)/C using full-precision AP values from the identical clean and attacked evaluation pipeline.  
  <sub>TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems</sub>

**protocols**
- `Person-only, 640-pixel Ultralytics validation followed by COCOeval` — Ultralytics validation supports an explicit class list, and its official config maps class 0 to person. Using one frozen preprocessing/postprocessing path for clean and attacked data isolates the perturbation effect.  
  <sub>https://docs.ultralytics.com/modes/val; https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml</sub>
- `Latent white-box PGD with multiple starts` — The learned-set source reduces attacks to optimizing loss over a norm-bounded latent variable. The released illumination config uses 50 PGD iterations; canonical PGD work uses random initialization and repeated runs to expose weak optimization.  
  <sub>https://raw.githubusercontent.com/locuslab/perturbation_learning/master/configs_attack/mi_cvae_pgd_881.json; Towards Deep Learning Models Resistant to Adversarial Attacks</sub>
- `Detector-specific attack-objective control` — Object detection has multiple failure modes, so a robustness conclusion should not depend on a single weak surrogate objective.  
  <sub>TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems</sub>
- `Data-driven latent-radius selection independent of YOLO val2017` — The illumination CVAE paper reports validation encoding-norm distributions, including 75th percentiles, so B should be frozen from held-out generator validation rather than chosen to make detector results favorable.  
  <sub>arXiv:2007.08450</sub>

## Recipe

### G0 — gate  ·  confidence: high

Freeze and hash the official yolo11n.pt checkpoint, the Ultralytics commit/version, pycocotools version, instances_val2017.json and all 5,000 val2017 image IDs. In FP32 eval mode, run the unmodified checkpoint on every val2017 image with the standard Ultralytics imgsz=640 aspect-ratio-preserving preprocessing and one frozen low confidence/NMS setup; export detections. Use COCOeval bbox with catIds=[person], imgIds=all val2017, iouThrs=[.50], area=all and maxDets=100 to obtain clean person AP50 C. Also cross-check the Ultralytics class-0 AP50. Do not filter the dataset down to images containing people: negative images contribute false positives.

- **Pass:** Pass only if C >= 0.90. If C < 0.90, REFUTE immediately: because z=0/no modification is part of the stated perturbation setting and the claim explicitly says AP50>=0.90, the named clean pretrained detector cannot satisfy the claim. Require the pycocotools and Ultralytics person AP50 values to agree within 0.005; larger disagreement is an evaluation-pipeline error that must be repaired before deciding.
- **Failure means:** C<0.90 directly refutes the numerical claim. Evaluator disagreement >0.005 invalidates the apparatus rather than supporting the claim.
- **Resources:** official Ultralytics yolo11n.pt, COCO val2017 images, instances_val2017.json, Ultralytics, pycocotools
- **Why:** This is the cheapest decisive prerequisite and verifies the exact model, split, class and AP semantics before training a generator. Official YOLO11 documentation confirms the 640-pixel COCO artifact but publishes only all-class AP50:95=0.395, so the much stronger person AP50=0.90 assertion must be measured directly.
  - *https://docs.ultralytics.com/models/yolo11*: "See Detection Docs for usage examples with these models trained on COCO , which include 80 pretrained classes.

 Model size
 (pixels) mAP val
50-95 Speed
 CPU ONNX
(ms) Speed
 T4 TensorRT10
(ms) params
 (M) FLOPs
 (B) YOLO11n 640 39.5 56.1 ± 0.8 1.5 ± 0.0 2.6 6.5"
  - *https://docs.ultralytics.com/datasets/detect/coco*: "The COCO dataset is split into three subsets:

 Train2017 : 118,287 images for training object detection, segmentation, and captioning models.
 Val2017 : 5,000 images used for validation during model training."
  - *https://docs.ultralytics.com/modes/val*: "classes list[int] None Specifies a list of class IDs to evaluate. Useful for filtering out and focusing only on certain classes during evaluation."
  - *https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py*: "stats[0] = _summarize(1)
stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])"
- **Caveat:** A clean attack cannot be assumed to monotonically degrade AP for every individual constructed image, but the claim independently requires attacked AP>=.90 and calls it retained performance. Measuring C first is still a direct and very cheap check of the named clean baseline.
- **Caveat:** Do not infer person AP50 from official all-class AP50:95=0.395; they are different metrics.

### A1 — apparatus (after G0)  ·  confidence: medium

Only if G0 passes, construct the exact claimed illumination generator without val leakage. Split COCO train2017 into scene/content-disjoint generator-train and generator-validation subsets before fitting. For each condition image x, obtain paired observations x_tilde of the same scene/content under independently varied real lighting; document capture/intervention and co-registration so boxes and semantics remain unchanged. Train a conditional UNet-style CVAE with prior p(z|x), posterior q(z|x,x_tilde), and decoder g(x,z), maximizing the CVAE ELBO. Do not use YOLO gradients, COCO val2017 images/labels, or AP to select the architecture/checkpoint. Define latent coordinates before reparameterization and freeze B as the 75th percentile of held-out posterior-encoding L2 norms, mirroring the primary illumination protocol. If literal COCO paired relighting cannot be obtained, stop rather than silently substitute the unrelated MI dataset or synthetic brightness jitter.

- **Pass:** Pass only if (i) no COCO val2017 image, annotation, detector gradient or detector score was used in generator fitting/model/B selection; (ii) every training pair has verified same geometry/content and a documented lighting-only intervention; (iii) a finite B is fixed from the held-out 75th-percentile L2 encoding norm before headline attack; and (iv) g is demonstrably conditional: at z representing identity/prior center its mean held-out pixel MSE to x is <=0.004, the published MI CVAE reconstruction-error benchmark. Failure of any item means the specified learned COCO illumination-CVAE apparatus has not been built.
- **Failure means:** The headline result would concern an undefined, leaked, or non-illumination generator rather than the claim. Inability to form paired COCO-derived lighting data exposes a serious internal underspecification of the claim and prevents verification as written.
- **Resources:** COCO train2017, paired COCO-derived multi-lighting observations, conditional UNet CVAE implementation
- **Why:** A CVAE labeled “illumination” is not validated merely by low training loss. Primary work says paired perturbed examples are required and uses same-scene observations under 25 lighting conditions with scene-disjoint holdouts. B must be fixed independently of the detector to avoid choosing a weak threat model after seeing val AP.
  - *arXiv:2007.08450*: "The approach is widely applicable to a range of robustness settings, as we make no assumptions on the type of perturbation being learned: the only requirement is to collect pairs of perturbed examples."
  - *arXiv:2007.08450*: "We use the test set provided by Murmann et al. (2019) which consists of 30 held out scenes and hold out 25 additional “drylab” scenes for validation. Unlike in the CIFAR10 common corruptions setting, there is no such thing as an “unperturbed” example in this dataset so we train on random pairs selec"
  - *arXiv:2007.08450*: "Table 11: Multi-illumination validation set statistics for the ℓ2 norm of the latent space encodings.
Resolution β Mean Std 25% 50% 75% Max
MIP5 (125 × 187) 1 7.42 2.14 5.93 7.35 8.81 16.63
MIP4 (250 × 375) 1 10.88 3.18 8.69 10.75 12.95 24.69
MIP3 (500 × 750) 10 9.11 2.72 7.14 9.00 10.97 20.65"
  - *arXiv:2007.08450*: "Table 1: Condensed evaluation of CVAE perturbation sets trained to produce rotation, translation, ans skew transformations on MNIST (MNIST-RTS), CIFAR10 common corruptions (CIFAR10-C) and multi-illumination perturbations (MI). The approximation error measures the necessary subset property, and the e"
- **Caveat:** This step deliberately takes “learned from COCO” literally. Standard public COCO does not supply relit pairs, and no primary source located establishes a canonical COCO illumination CVAE. Producing new paired observations derived from COCO is therefore a demanding but necessary interpretation, not an off-the-shelf benchmark.
- **Caveat:** The reconstruction-error <=0.004 threshold comes from a lower-resolution MI CVAE, not a published 640-square COCO model. It is an evidence-based reference, but resolution/content shifts may warrant preregistered uncertainty rather than blindly treating it as universal.
- **Caveat:** A 75th-percentile B is standard in the cited apparatus but the claim itself does not specify this quantile. A different detector-independent B with equally strong held-out coverage evidence could be valid.

### A2 — apparatus (after A1)  ·  confidence: medium

Validate the frozen generator and the chosen B on scene/content-disjoint held-out paired lighting data, at the same 640-input pipeline used for the attack. For each held-out pair (x,x_tilde), run multi-start latent PGD over ||z||2<=B to minimize normalized pixel MSE ||g(x,z)-x_tilde||^2. Compute mean PGD approximation error, expected approximation error from prior samples truncated to B, reconstruction error, and 95% bootstrap CIs. Separately sample/interpolate at B and audit for geometric warps, object insertion/removal, texture replacement, clipping artifacts and label invalidation; a blinded human/photometric audit must classify at least 95% as lighting-only and label-preserving.

- **Pass:** Pass only if upper 95% CI of mean PGD approximation error <=0.009, upper 95% CI of mean expected approximation error <=0.049, upper 95% CI of reconstruction error <=0.0055, and >=95% of audited samples are lighting-only/label-preserving. The numeric limits are the worst published values across the cited 125x187, 250x375 and 500x750 multi-illumination CVAEs (PGD AE .006/.008/.009, EAE .049/.049/.048, reconstruction .0040/.0042/.0055).
- **Failure means:** Failure means g and B do not establish the claimed illumination-consistent perturbation set; a detector result under it cannot verify the claim.
- **Resources:** frozen CVAE from A1, scene-disjoint held-out paired lighting set, latent PGD reconstruction evaluator
- **Why:** The primary learned-set framework explicitly warns that a data-learned set lacks mathematical rigor and evaluates whether it contains real perturbations by latent-PGD approximation. This validation separates a genuine illumination threat model from an arbitrary image autoencoder.
  - *arXiv:2007.08450*: "For a perturbation set defined by the generative model from Equation (2), this amounts to finding a latent vector z which best approximates the perturbed example x˜ by solving the following problem:
min d(g(z, x), x˜).
||z||≤ε
This approximation error can be upper bounded with point estimates or can"
  - *arXiv:2007.08450*: "Table 10: Measuring and comparing quality metrics for a multi-illumination CVAE at different resolutions.
Test set quality metrics Test set CVAE metrics
Resolution ε Encoder AE PGD AE OAE EAE Recon. err KL
MIP5 (125 × 187) 17 0.019 0.006 0.049 0.13 0.0040 65.8
MIP4 (250 × 375) 25 0.034 0.008 0.049 0"
- **Caveat:** The 95% human-audit threshold is chosen for this recipe, not found as a field-standard published cutoff. It is necessary because low MSE alone does not prove lighting semantics.
- **Caveat:** Published errors use the source paper's normalization and resolutions. The implementation must exactly document and match its normalized-MSE definition; otherwise those numeric thresholds are not portable.
- **Caveat:** The label-preservation audit should oversample images with people, small people and crowded scenes, since semantic violations there most directly bias the headline AP.

### H3 — headline (after G0, A2)  ·  confidence: high

Run the direct white-box attack on every one of the 5,000 COCO val2017 images. Freeze YOLO and g. For each x, optimize its own latent u by projected gradient ascent inside ||u||2<=B, differentiating through g, the exact imgsz=640 preprocessing and YOLO. Use at least 50 iterations and 20 random starts (plus the identity/prior-center start). Use a detector-specific person-vanishing/objectness loss and retain the feasible candidate with maximal attack loss per image. Enforce BOTH constraints after every step by a feasibility-preserving method: latent projection to B, then solve/reject/backtrack latent steps until the generated x_adv=g(x,u) satisfies ||x_adv-x||infinity<=12/255; do not pixel-clip x_adv to the epsilon box, since that would generally leave the CVAE range. Recheck both norms in FP32 after final generation. Evaluate the complete attacked corpus with exactly G0's frozen inference/export/COCOeval pipeline to get A.

- **Pass:** Pass only if every image is feasible (max_i ||u_i||2<=B plus tolerance and max_i ||x_adv_i-x_i||infinity<=12/255 plus numerical tolerance) AND A>=max(0.90,0.95*C). If any image lacks a feasible attacked candidate use identity u=0 for that image only if A1 established it is an exact feasible identity; otherwise apparatus failure. Report C, A and D=(C-A)/C at full precision. A<0.90 OR D>0.05 REFUTES.
- **Failure means:** A numerical threshold failure directly refutes the retained-AP claim. A constraint violation invalidates the headline experiment and must be repaired, never counted as an attack success or defense success.
- **Resources:** frozen yolo11n.pt, frozen validated CVAE, COCO val2017, pycocotools, white-box latent PGD implementation
- **Why:** This is the only direct test enforcing every stated scope condition simultaneously: exact checkpoint, all val2017, person only, 640 preprocessing, white-box latent PGD, latent bound, CVAE range, L-infinity 12/255, absolute AP floor and relative degradation.
  - *arXiv:2007.08450*: "Then, we can define a learned perturbation set as follows:
S(x) = {g(z, x) : ||z|| ≤ ε} (2)
In other words, we have taken a well-defined norm-bounded ball in the latent space and mapped it to a set of perturbations with a generator g, which perturbs x into x˜ via a latent code z."
  - *https://raw.githubusercontent.com/locuslab/perturbation_learning/master/configs_attack/mi_cvae_pgd_881.json*: ""attack": {
 "type": "cvae_attack", 
 "max_dist": 8.81, 
 "alpha": 0.4405, 
 "niters": 50,"
  - *Towards Deep Learning Models Resistant to Adversarial Attacks*: "During 20 runs of projected gradient descent (PGD). Each run starts at a uniformly random point in the ℓ∞-ball around the same natural example"
  - *TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems*: "The mAP is the average precision of all N object classes. Lower mAP corresponds to better effectiveness of our TOG attacks, which indicates that a smaller portion of the objects are correctly detected."
- **Caveat:** Projection onto the intersection of a latent ball and a generator-induced pixel constraint is nontrivial. Backtracking/rejection preserves membership but may weaken PGD. C4 therefore checks attack convergence and alternative feasible parameterizations.
- **Caveat:** Standard CVAE prior-center output may only approximate x, so exact identity feasibility must be engineered or established; otherwise the natural zero-perturbation fallback is not guaranteed.
- **Caveat:** AP is dataset-level and nondifferentiable, so detector losses are surrogates. Keeping the strongest feasible candidate over multiple objectives/restarts in C4 reduces, but does not eliminate, this limitation.

### C4 — control (after H3)  ·  confidence: medium

Attack-strength and objective control. Repeat H3 under a preregistered grid: iterations {50,100,200}, random starts {1,5,20}, and at least two detector objectives—person-objectness/vanishing and the full matched person detection loss including classification/objectness/localization. For each image form the union of all feasible candidates and select the candidate with the largest common evaluation surrogate fixed in advance, then compute worst-set AP A_strong. Include a gradient sanity test on a stratified subset (finite differences versus autograd directional derivatives, relative error <=1e-2) and log attack loss versus iteration. Also run a random latent-search baseline with the same number of generator evaluations and identical constraints.

- **Pass:** For the claim to survive, the strongest feasible union must still have A_strong>=max(0.90,0.95*C). Additionally, 100/20 and 200/20 AP values must differ by <=0.005, at least 95% of tested finite-difference gradients must have relative error <=1e-2, and optimized PGD must be no weaker than random search (A_PGD<=A_random+0.005). If A_strong violates either claim threshold, REFUTE. If convergence/gradient/PGD-vs-random checks fail, do not verify; repair/strengthen the attack and rerun.
- **Failure means:** A_strong below threshold refutes robustness. Diagnostics failing without an AP refutation mean the apparent robustness may be caused by weak optimization, broken gradients or constraint handling.
- **Resources:** headline attack code, alternative detector losses, finite-difference checker
- **Why:** White-box robustness cannot be established from one arbitrary detector loss or one PGD schedule. Detection literature distinguishes vanishing, fabrication and mislabeling, while canonical PGD evaluation uses repeated random starts. A plateau and objective union make a false robustness conclusion less likely.
  - *TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems*: "We demonstrate three different targeted attacks for real-time object detection: (1) object vanishing attack, (2) object fabrication attack, and (3) object mislabeling attack."
  - *Towards Deep Learning Models Resistant to Adversarial Attacks*: "During 20 runs of projected gradient descent (PGD). Each run starts at a uniformly random point in the ℓ∞-ball around the same natural example"
- **Caveat:** The <=0.005 convergence margin, <=1e-2 finite-difference error, and random-baseline tolerance are recipe choices, not published universal standards.
- **Caveat:** Selecting candidates by a common surrogate rather than AP is unavoidable at per-image generation time; an oracle selection using ground truth can itself alter the threat model and must be preregistered.
- **Caveat:** A specialized exact constrained optimizer or expectation-over-transformation may find still stronger examples; this recipe cannot certify global worst-case robustness.

### C5 — control (after C4)  ·  confidence: medium

Mechanism and budget ablations. Re-evaluate full val2017 under: (a) identity/prior-center output; (b) random feasible latent samples at B; (c) PGD with B/2; (d) PGD at image bounds 0, 4/255, 8/255 and 12/255; and (e) matched-budget direct pixel-space PGD at 12/255 that removes the CVAE/illumination constraint. Keep all detector preprocessing and evaluator settings fixed. The direct-pixel result is a contextual control, not part of the claim. For generator conditions use the strongest converged procedure from C4.

- **Pass:** Final VERIFY requires the exact claimed endpoint (B,12/255) from the strongest attack to meet A>=max(0.90,0.95C), identity output AP to agree with clean C within 0.005, and feasible endpoint AP to be nonincreasing with attack strength up to a tolerance of 0.005 (A_12<=A_8+0.005 and A_B<=A_B/2+0.005). If the exact endpoint fails either numerical claim threshold, REFUTE. If identity differs >0.005 or stronger attacks paradoxically look >0.005 better, do not verify until generator/evaluator/optimization is fixed. No pass/fail requirement is imposed on direct pixel PGD, but report it to delimit what any robustness is attributable to.
- **Failure means:** Endpoint threshold failure refutes. Identity failure indicates the generator changed the clean baseline. Nonmonotonicity indicates attack weakness or feasibility artifacts. A collapse only under direct pixel PGD would show robustness is specific to the learned illumination manifold, exactly the limited mechanism claimed, rather than generic L-infinity robustness.
- **Resources:** strongest attack from C4, random latent sampler, direct pixel PGD control
- **Why:** Sensitivity to both stated budgets checks for cherry-picked weakness; identity verifies attribution; random search verifies adversarial optimization adds strength; direct pixel PGD distinguishes illumination-manifold robustness from general pixel-space robustness.
  - *arXiv:2007.08450*: "Then, we can define a learned perturbation set as follows:
S(x) = {g(z, x) : ||z|| ≤ ε} (2)
In other words, we have taken a well-defined norm-bounded ball in the latent space and mapped it to a set of perturbations with a generator g, which perturbs x into x˜ via a latent code z."
  - *TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems*: "The mAP is the average precision of all N object classes. Lower mAP corresponds to better effectiveness of our TOG attacks, which indicates that a smaller portion of the objects are correctly detected."
- **Caveat:** The monotonic tolerances and intermediate epsilon levels are chosen diagnostics, not claim-specified thresholds.
- **Caveat:** Because B and pixel epsilon interact through a nonlinear generator, some intermediate conditions may have different feasible geometry; nonmonotonicity is a warning, not by itself a direct refutation.
- **Caveat:** Pixel-PGD is deliberately outside the claimed illumination model and must not replace the headline test.

## Decision logic

REFUTE immediately if G0 yields clean person AP50 C<0.90. If G0 passes but the team cannot build and validate the literal COCO-derived paired illumination CVAE in A1-A2, the claim is NOT VERIFIED/ill-posed as written; if the experimental decision system requires binary output, treat failure of this presupposed apparatus as REFUTE of the full conjunction, while clearly distinguishing it from a detector-AP refutation. Once A1-A2 pass, define T=max(0.90,0.95C). H3, C4 and C5 all evaluate the identical exact endpoint; use A*=the lowest AP from the union of all feasible, correctly implemented, converged attacks. REFUTE if A*<T, equivalently if A*<0.90 or (C-A*)/C>0.05. VERIFY only if G0 passes; A1 and A2 validate the nonleaking, paired, illumination-consistent CVAE and fixed B; every attacked image satisfies both latent and 12/255 image bounds while remaining an exact g(x,z) output; H3 has A>=T; C4's strongest objective/restart/iteration union also has A*>=T and all gradient/convergence checks pass; and C5's identity/budget diagnostics pass. Report exact C, A*, D, feasibility maxima, hashes and bootstrap uncertainty. Because this is empirical attack evaluation, VERIFY means “survived this strong reference attack suite,” not a mathematical certificate against every possible latent optimizer.

## Known weaknesses of this reference

- The likely cheapest real-world outcome is that G0 alone refutes the extraordinary 0.90 person-AP50 premise. I found an unofficial issue log suggesting YOLO11n person mAP50 around 0.75, but did not use or cite that number as a field norm because it is not an authoritative benchmark and the recipe should measure it directly.
- The claim's generator clause is internally problematic: public COCO does not provide same-scene relighting pairs, while the closest primary method explicitly requires perturbed pairs and trains illumination CVAEs on a separate Multi-Illumination dataset. My “COCO-derived paired recapture/intervention” interpretation makes the assertion testable but may be stricter than the claimant intended. Training on MI instead would test a nearby, not literal, claim.
- Generator validation thresholds (.009 PGD AE, .049 EAE, .0055 reconstruction) are grounded in the primary paper but at 125x187 to 500x750, not 640x640 COCO. Applying the worst published values at a new domain/resolution is defensible as an apparatus bar but not a universal theorem.
- The >=95% lighting-only human-audit criterion, 0.005 metric/convergence tolerances, finite-difference 1e-2 threshold and attack sweep grid are my design choices, not sourced universal cutoffs. They are diagnostics and should not wrongly exclude an alternative validated implementation with equivalently decisive checks.
- I adopted an L2 latent ball because that is the canonical CVAE construction and its released illumination config. The prompt only says a norm-bounded latent space “e.g. ||z||<=B”; another norm could satisfy the English claim. An evaluator should accept a preregistered alternative norm if B is independently calibrated and the generator passes held-out coverage validation.
- The exact official YOLO11n checkpoint can change behind an unversioned download name. Hashing the weight and package commit is essential; results from another release are not automatically comparable.
- Ultralytics class filtering can change NMS behavior relative to evaluating all predictions then selecting COCO person. The primary recipe should export all-class predictions under one frozen pipeline and restrict COCOeval to person; classes=[0] is a useful cross-check, not permission to alter suppression asymmetrically between clean and attack.
- “640x640” remains semantically ambiguous because Ultralytics defaults to aspect-ratio-preserving/rectangular preprocessing rather than naive square warping. I chose the official imgsz=640 path. A claimant who means hard 640x640 warping must state it and rerun both clean and attack identically.
- Intersection-constrained latent PGD is technically difficult. Pixel clipping after decoding violates generator membership; rejection/backtracking preserves it but can weaken attack. The strength sweep, random baseline and gradient tests reduce this risk but do not prove a global optimum.
- Dataset-level AP is nondifferentiable and nonseparable. Surrogate detector losses may fail to find the AP-worst corpus, so passing cannot certify the claim. Multiple detector-specific objectives make the empirical conclusion stronger but still attack-dependent.
- A relative drop can be negative if attacked AP happens to exceed clean AP. The formula handles that correctly. The absolute 0.90 floor remains independently binding.
- The recipe does not assign statistical hypothesis tests to AP because COCO AP on the fixed complete val2017 split is normally reported deterministically. Bootstrap intervals are useful for generator validation and can be reported for AP sensitivity, but the claim is about this fixed split, not a sampled population.
- Check_resource found COCO mirrors but not yolo11n on the searched hubs; official Ultralytics documentation and release download establish the artifact instead. The Multi-Illumination resource is available through its official project page despite no Hugging Face/Kaggle hit.
- The direct-pixel-PGD control is intentionally not a condition for refuting or verifying the narrow learned-illumination claim. Treating its expected failure as a claim failure would over-scope the assertion.