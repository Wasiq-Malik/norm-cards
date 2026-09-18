# Reference experiments — problem arxiv2608_25585

**Claim.** Vision-language-action manipulation policies can adapt at test time to task distributions excluded from adaptation training by using expert demonstrations as context without updating model weights. The property at stake is joint novel-task execution quality and inference scalability in both simulated and physical robotic manipulation. The claim is accepted only if mean whole-task success exceeds that of the strongest prior in-context imitation policy under equal training and expert-context budgets in both settings, while inference-latency growth per additional expert segment is lower than that strongest prior policy’s growth.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2608.25585

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Construct the held-out-task evaluation used for test-time adaptation. Use GR00t N1.5 as the VLA backbone for every method, dual-view RGB observations from third-person and wrist cameras, an action horizon of 16, four denoising steps, and one retrieved expert segment by default. For LIBERO, use LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long, each containing 10 tasks with 50 expert demonstrations per task; train on three suites and reserve the fourth suite entirely for evaluation, populate each held-out task’s buffer with three expert demonstrations, and measure mean success over 50 trials. For the UR5e environment, collect 30 GELLO demonstrations for each of StackBox, ThrowTrash, CloseDrawer, and PressPedal; train on three tasks plus LIBERO as auxiliary data, hold out the fourth task, provide four demonstrations at evaluation, and report success over 12 trials per task. Use the same demonstration buffer for all compared methods.

### Experiment 1

Build the retrieval and grounded-execution components evaluated in the adaptation tests. Slice demonstrations into length-16 action segments with stride 4, align pairs of demonstrations from the same task by Dynamic Time Warping over their action sequences, and train a two-layer retrieval Transformer with contrastive behavioral alignment loss using temperature τ = 0.01. Configure the retriever with hidden size 512, eight attention heads, intermediate size 1024, and two hidden layers. Fine-tune the action head with relevant retrieved segments and randomly sampled irrelevant segments using contextual adherence margin m = 0.01 and loss weight λ = 0.1, and initialize the noisy action chunk with the empirical mean action chunk of the retrieved segments. Compare against Vanilla VLA, RAEA, and RICL variants with λw = 1.0 for RICL_G and λw = 0.2 for RICL_R.

### Experiment 2

Test novel-task adaptation on held-out LIBERO suites. Train on three of the four LIBERO suites, evaluate on the excluded suite with three expert demonstrations per held-out task and 50 trials, and compare Vanilla VLA, RAEA, RICL_G, RICL_R, and RA-VLA. With the off-the-shelf SigLIP 2 retriever, RAEA scores 0.060, 0.106, 0.008, and 0.006 on LIBERO-Spatial, Object, Goal, and Long, averaging 0.0450; RICL_G scores 0.092, 0.156, 0.106, and 0.000, averaging 0.0885; and RICL_R scores 0.048, 0.052, 0.086, and 0.000, averaging 0.0465. With the action-aware retriever, RAEA scores 0.164, 0.182, 0.104, and 0.044, averaging 0.1235; RICL_G scores 0.156, 0.198, 0.116, and 0.048, averaging 0.1295; RICL_R scores 0.210, 0.326, 0.206, and 0.092, averaging 0.2085; and RA-VLA scores 0.320, 0.556, 0.532, and 0.130, averaging 0.3845. Vanilla VLA averages 0.0170. RA-VLA therefore exceeds the strongest baseline average, action-aware RICL_R at 0.2085, by 0.1760 absolute success rate.

### Experiment 3

Test novel-task adaptation on the physical UR5e robot. Under the leave-one-task-out protocol, equip RAEA, RICL_R, and RA-VLA with the action-aware retriever, provide four demonstrations for the held-out task, and run 12 trials for each of StackBox, ThrowTrash, CloseDrawer, and PressPedal. Whole-task success for Vanilla VLA is 0.000, 0.000, 0.333, and 0.000, averaging 0.0833; RAEA obtains 0.000, 0.000, 0.500, and 0.000, averaging 0.1250; RICL_R obtains 0.250, 0.250, 0.583, and 0.333, averaging 0.3542; and RA-VLA obtains 0.417, 0.750, 0.667, and 0.417, averaging 0.5625. The corresponding first-subgoal rates are 0.000, 0.083, 0.500, and 0.000 for Vanilla VLA; 0.000, 0.250, 0.667, and 0.000 for RAEA; 0.583, 0.833, 0.667, and 0.667 for RICL_R; and 0.750, 0.917, 0.750, and 0.583 for RA-VLA. RA-VLA exceeds the strongest baseline whole-task average, RICL_R at 0.3542, by 0.2083 absolute success rate.

### Experiment 4

Measure whether generated actions change when relevant expert context is replaced by random context. On 1,000 samples, compute Relative Contextual Sensitivity as the norm of the difference between actions generated with retrieved and random contexts divided by the norm of the action generated with retrieved context, and pair it with LIBERO-Goal success. RAFT, as named in the reported results, has sensitivity 0.0247 and success 0.104; RICL_R has 0.0788 and 0.206; RA-VLA without the contextual adherence loss has 0.0353 and 0.098; and full RA-VLA has 0.3639 and 0.532. Removing the adherence loss therefore reduces both contextual sensitivity and held-out-task success relative to full RA-VLA.

### Experiment 5

Test whether the learned retriever aligns behavior rather than general visual similarity. For each segment in one expert trajectory, retrieve its nearest segment from a second trajectory and compare the action-aware retriever with off-the-shelf SigLIP 2 and Eagle 2 representations. The off-the-shelf representations retrieve segments from behaviorally different trajectory phases, while the action-aware retriever matches corresponding motion phases. In a quantitative replacement test on LIBERO-Goal, changing RA-VLA from the baseline retriever to the action-aware retriever raises success from 10.2% to 53.2%.

### Experiment 6

Compare reported policy rollouts to inspect how the methods use retrieved guidance. On the held-out LIBERO task of placing a wine bottle on top of a cabinet, RAEA picks the wrong object, RICL_R fails to pick up the target, and RA-VLA completes the pick-and-place. On the physical StackBox task, RAEA picks up trash, RICL_R lifts the correct box but does not place it, and RA-VLA completes the task. On ThrowTrash, Vanilla VLA and RAEA attempt to press the pedal, RICL_R picks up the trash but does not place it in the red bin, and RA-VLA completes the task. On CloseDrawer, Vanilla VLA fails while RAEA, RICL_R, and RA-VLA close the drawer. On PressPedal, Vanilla VLA and RAEA close the drawer, RICL_R fails to press precisely, and RA-VLA presses the pedal. In two additional LIBERO rollouts involving placing moka pots or a bowl on a stove, full RA-VLA succeeds while RA-VLA without the contextual adherence loss fails to follow the retrieved context.

### Experiment 7

Test inference scalability as the number of retrieved expert segments K increases. On a single NVIDIA A100 GPU, generate one action chunk with four denoising steps and average latency over 1,000 inference runs for each context size. RICL latency increases steeply as additional expert segments are concatenated into the multimodal prompt, and RAEA follows a similar increasing trend. RA-VLA remains nearly constant with only negligible additional latency as K grows because retrieved segments are encoded independently and cached. The paper reports this comparison by direction and does not state numerical latency-growth slopes in the text.

### Experiment 8

Benchmark whether retrieval itself becomes a latency bottleneck at large buffer size. Store 512-dimensional retrieval keys and retrieve by multiplying a 1 × 512 query against N × 512 stored keys followed by Top-K selection on an NVIDIA A100. At N = 10^7 stored segments, retrieval takes 0.18 ms, which is 0.36% of total inference latency.

### Experiment 9

Vary the demonstration-buffer size and the number of retrieved segments on LIBERO-Goal. With buffer size N equal to 1, 2, 3, and 4 demonstrations, RA-VLA’s average success rates are 0.482, 0.518, 0.532, and 0.552. With retrieval size K equal to 1, 2, 3, and 4 segments, its success rates are 0.532, 0.540, 0.548, and 0.546. Performance increases across the tested buffer sizes and changes only marginally across the tested retrieval sizes, peaking at K = 3.
