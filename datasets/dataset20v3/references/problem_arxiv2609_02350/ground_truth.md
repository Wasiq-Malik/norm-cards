# Reference experiments — problem arxiv2609_02350

**Claim.** The question is whether a multimodal-large-language-model agent for continuous vision-language navigation can navigate unseen environments efficiently using only monocular RGB, standard offline expert supervision, and no auxiliary spatial model or extra training data. The proposition is supported only if such an agent exceeds the strongest prior agent subject to the same resource constraints in both unseen-environment success rate and success-weighted path efficiency, while using no more than 24 GB of peak inference memory and lower per-step inference latency than that comparator. It fails if the performance advantage requires additional training data, richer sensors, external spatial tools, or more than the stated memory budget.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2609.02350

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Build the LookStep agent and its structured supervision. Start from Qwen3-VL 8B and train for one epoch with the ms-swift package at a learning rate of 1e-5 using only the R2R-CE and RxR-CE expert trajectories. At every expert step, supervise one autoregressive sequence containing the progress label, candidate-action future-state labels, memory-write decision, memory role, and ground-truth next action; generate the non-action labels automatically from deterministic trajectory rules, including a K=5 future-action window, without additional environment rollouts or manual annotations. Use a FIFO memory queue of length 8. Training runs on eight NVIDIA A100 80GB GPUs for about 1,000 PCIe GPU-hours.

### Experiment 1

Test navigation performance on unseen continuous environments using monocular RGB and no auxiliary spatial module or external data beyond the standard R2R-CE and RxR-CE datasets. Evaluate LookStep in Habitat on the Val-Unseen splits with Navigation Error, Oracle Success Rate, Success Rate, and Success-weighted Path Length. On R2R-CE, LookStep obtains NE 5.34, OS 55.9, SR 49.7, and SPL 45.3; Sim2Real, the listed single-RGB 0K-extra-data comparator without an auxiliary module, obtains NE 5.95, OS 55.8, SR 44.9, and SPL 30.4. On RxR-CE, LookStep obtains NE 6.89, SR 46.9, and SPL 39.9, compared with Sim2Real at NE 8.79, SR 36.7, and SPL 25.5. JanusVLN, which uses the external VGGT visual-spatial tool, obtains R2R SR 52.8 and SPL 49.2 and RxR SR 51.4 and SPL 44.3. StreamVLN* obtains R2R SR 45.5 and SPL 41.6 using 10,033K external training examples, while NaVILA* obtains SR 49.7 and SPL 45.5 using 12,574K external examples. Comparator values are taken from their respective papers rather than rerun by the authors.

### Experiment 2

Measure whether LookStep remains within the stated inference resource range as rolling-memory capacity changes. Run R2R-CE Val-Unseen with memory lengths 4, 6, 8, and 10; the corresponding success rates are 48.9%, 48.3%, 49.7%, and 49.3%, while peak GPU memory is 18.9GB, 19.2GB, 19.7GB, and 20.3GB. Peak memory therefore remains below 24GB at every tested capacity. At the default length 8, per-step inference latency is 59ms. The paper reports JanusVLN at 52.8% SR, 44.3GB peak memory, and 194ms per step.

### Experiment 3

Determine which structured components account for navigation performance. On R2R-CE, compare full LookStep at NE 5.34, OS 55.9, SR 49.7, and SPL 45.3 with LookStep without LFS at 5.39, 52.8, 46.9, and 42.4; an inference-only replacement of EDRM with JanusVLN-style uniform history sampling, without retraining, at 6.64, 70.9, 37.4, and 22.1; a retrained model without EDRM at 5.53, 51.7, 45.2, and 42.7; action-only imitation learning at 6.01, 50.1, 41.7, and 39.5; LookStep without candidate-action outcomes at 5.37, 53.2, 47.8, and 43.9; and LookStep without progress prediction at 5.35, 53.7, 48.3, and 43.6. Every reported removal has lower SR than the full model.

### Experiment 4

Check whether the generated future-state labels correspond to subsequent action behavior. On R2R, collect cases in which the model predicts advance_to_goal and measure whether it subsequently issues STOP successfully: the reported Successful STOP rate is 55.52%, while No STOP is 0.028%. For instructions requiring a turn, test the prediction assigned to the opposite candidate action; the model predicts a negative outcome for that opposite action in 81.17% of cases.

### Experiment 5

Test whether generated memory roles identify the trajectory events used by EDRM. On R2R, compare predicted roles with the deterministic labels and obtain 100% accuracy for start_view, 100% for post_turn_alignment, 99.51% for turn_start, 98.99% for turn_end, 100% for stop_evidence, and 43.64% for goal_approach. After the model emits turn_end, 99.72% of subsequent actions no longer involve turning.

### Experiment 6

Check whether memory-write behavior remains consistent across complete unseen episodes rather than only on isolated role labels. Analyze all 1,839 R2R Val-Unseen episodes comprising 154,481 executed steps, reconstructing a rule-consistency proxy from each executed action sequence with the same deterministic K=5 EDRM rules because executed trajectories cannot always be aligned to expert roles. The model predicts keep on 53.93% and drop on 46.07% of steps; keep precision, recall, and F1 are 99.37%, 99.28%, and 99.32%. Mean and median episode-level keep F1 are 98.79% and 100.00%, and 93.31% of episodes have keep F1 of at least 95%. Across 9,746 multi-step turn segments, both start and end boundaries are identified correctly in 98.26%, interior steps are labeled drop/recent_only in 99.98%, and 94.23% of episodes have every turn pair identified correctly.

### Experiment 7

Measure how much the R2R result depends on including RxR trajectories during training. Train one model on R2R alone and another jointly on R2R and RxR, then evaluate both on R2R. R2R-only training yields NE 6.61, OS 48.3, SR 40.3, and SPL 36.5; joint R2R+RxR training yields NE 5.34, OS 55.9, SR 49.7, and SPL 45.3.

### Experiment 8

Test transfer to a different navigation dataset and task setting. Evaluate LookStep on the HM3D-OVON Val-Unseen subset and obtain SR 38.0 and SPL 26.9. The reported comparison values are SR 35.2 and SPL 19.6 for VLFM and SR 37.1 and SPL 19.8 for DAgRL+OD.

### Experiment 9

Test simulation-trained LookStep on a physical robot in an unseen workspace. Run LookStep on a zsibot-L1 using a front-facing FPV RGB camera and an RTX 4090 24GB GPU. Use 10 instructions involving visually similar objects and behaviors such as entering and exiting rooms, repeat each instruction three times, and count a trial as successful when the robot stops within 1 meter of the target. Across the 30 trials, LookStep achieves a 70% success rate.

### Experiment 10

Characterize observed navigation failures rather than only successful trajectories. Select two representative R2R failures. In one, the agent initially follows the optimal trajectory direction but confuses the required turning landmark with other visually similar objects and fails to recognize it. In the other, the agent misidentifies the stopping position, stops outside the success region, and therefore fails the episode.
