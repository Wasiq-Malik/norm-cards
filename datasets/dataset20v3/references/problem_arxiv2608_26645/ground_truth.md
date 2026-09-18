# Reference experiments — problem arxiv2608_26645

**Claim.** Vision-language-action manipulation policies can achieve autonomous resilience to both robot-pose deviations that leave the environment task-valid and object-state failures that move it outside normal task trajectories. The decisive bar is a strictly higher mean end-to-end task success rate than the strongest prior comparison system and the same VLA without failure-recovery training, together with a strictly higher success rate than that unmodified VLA on every evaluated physical-robot task. Recovery must require no human correction guidance during deployment and must allow the policy to resume and complete the original manipulation task.

**Provenance.** transcribed from the source paper · authored by Claude Opus 5 — automated review only (figure/table-reference guard, schema validation, checked against the cited sections); PENDING HUMAN SIGN-OFF · 2026-09-18 · derived from arXiv:2608.26645

*Drafted by transcription from the source paper. Every experiment must be checked against the cited section before this is promoted with --accept.*

### Experiment 0

Build the retry- and reset-trained policy library used in the evaluations. For each RoboMimic task, begin with 10 human demonstrations and use MimicGen with default parameters to produce 500 augmented demonstrations. Inject perturbation-and-bridging sequences at the trajectory start and between subtasks, with maximum perturbations of 45° rotation and 0.5 m translation; exclude perturbation actions from the training targets and train on the bridging actions followed by the remaining task actions. Use Gemini-2.5-Pro at temperature 0.7 to analyze failure videos and identify object-centric reset skills, collect 20 human demonstrations for each reset skill, and augment each skill to 500 demonstrations with the same procedure. Fine-tune separate task and reset-skill LoRA adapters on the language-model and action-expert modules of π0.5 while freezing all other parameters, using Adam with a constant learning rate of 2.5×10^-4 and no cosine decay. At deployment, an online MLLM monitors execution; ID errors receive no intervention, while an identified OOD error causes the system to load the corresponding reset-skill adapter and prompt, execute the reset, reload the task adapter after the environment is judged valid, and resume the original task.

### Experiment 1

Validate the MLLM failure analyzer used to mine reset data and select reset skills. Manually label 50 execution videos for Coffee and ThreePieceAssembly, then evaluate Gemini-2.5-Pro on retry-versus-reset classification, reset-object identification, and failure-timestamp identification. Retry/reset accuracy is 88% on Coffee and 96% on ThreePieceAssembly; reset-object accuracy is 88% and 78%, respectively; timestamp accuracy, defined by whether the selected frame corresponds to the correct reset object, is 78% and 66%, respectively.

### Experiment 2

Test end-to-end success across the nine RoboMimic simulation tasks. Run OpenVLA, Task-conditioned, Subgoal-conditioned, Motion-conditioned, Subgoal Self-reflection, Phoenix, Phoenix-Human, π0.5, and FLARE for 50 evaluation trials per task. FLARE obtains 96%, 78%, 100%, 100%, 100%, 90%, 72%, 62%, and 58% on Coffee D0, Coffee D1, Stack D0, Stack D1, StackThree D0, StackThree D1, ThreePieceAssembly D0, ThreePieceAssembly D1, and Threading D0, for an 84.0% mean. π0.5 obtains 82%, 56%, 100%, 92%, 90%, 84%, 42%, 58%, and 46%, for a 72.2% mean; FLARE therefore improves the mean by 11.8 percentage points, exceeds π0.5 on seven tasks, and ties it at 100% on Stack D0 and 62% versus 58% on ThreePieceAssembly D1. Phoenix obtains 94%, 48%, 96%, 86%, 50%, 20%, 68%, 52%, and 6%, for a 57.8% mean. Phoenix-Human, which uses human correction of motion instructions, obtains a 78.9% mean. The remaining means are 38.0% for OpenVLA, 41.8% for Task-conditioned, 43.8% for Subgoal-conditioned, 46.9% for Motion-conditioned, and 48.0% for Subgoal Self-reflection. FLARE has the highest result on eight of the nine tasks; on Threading D0 its 58% is below Phoenix-Human’s 40% only in the paper's stated autonomous comparison context, while numerically π0.5 is 46% and Phoenix is 6%.

### Experiment 3

Measure how the retry augmentation depends on perturbation size. On Coffee D1, generate retry demonstrations under different maximum rotation r and translation t settings, fine-tune the VLA on each generated set, and measure task success and valid-demonstration generation rate. Sweep rotation with t fixed at 0.5 and translation with r fixed at 45°. Task performance is highest at r = 30° in the rotation sweep and t = 0.7 in the translation sweep. Increasing perturbation size initially raises task success by increasing demonstration variance but lowers the rate at which valid demonstrations can be generated; excessively large rotations or translations reduce task performance.

### Experiment 4

Test whether learned reset skills account for the gains on tasks with resettable OOD failures. Evaluate FLARE, FLARE without Reset, a Reset-Only variant, and an Oracle variant that replaces MLLM-generated reset instructions with human feedback on Coffee D0, Coffee D1, ThreePieceAssembly D0, and ThreePieceAssembly D1. FLARE scores 96%, 78%, 62%, and 58%. Removing Reset and retaining perturbation-and-bridging training scores 92%, 74%, 60%, and 54%, reducing the four-task average from 73.5% to 70.0%, or 3.5 percentage points. Reset-Only scores 88%, 64%, 60%, and 50%, averaging 65.5%. Human-instruction FLARE-Oracle scores 100%, 90%, 68%, and 64%, averaging 80.5%, seven percentage points above standard FLARE.

### Experiment 5

Measure whether the separately trained object-centric reset adapters can perform their reset operations. Construct dedicated randomized reset environments for the coffee-machine lid and coffee pod in Coffee and for the T-shaped and U-shaped blocks in ThreePieceAssembly, redefining success according to restoration of the reset target. The coffee-machine-lid reset succeeds in 84% of trials and its demonstrations are generated successfully at an 83.7% rate; the coffee-pod reset succeeds in 24% of trials with 11.6% generation efficiency. The T-shaped-block reset succeeds in 88% of trials with 48.6% generation efficiency; the U-shaped-block reset succeeds in 20% of trials with 5.9% generation efficiency.

### Experiment 6

Test FLARE on physical hardware without simulator-state logs. Run π0.5 and FLARE on a Piper arm observed by RealSense D435i top and wrist cameras for 40 trials each on Stack Three Blocks, a long-horizon task, and Insert U-shaped Block, a contact-rich task. Collect 10 human demonstrations and augment them to 50 for each task and reset skill, using Any6D object-pose estimates rather than ground-truth simulator coordinates. On Stack Three Blocks, π0.5 succeeds in 62.5% of trials and FLARE in 75.0%; on Insert U-shaped Block, π0.5 succeeds in 45.0% and FLARE in 55.0%.
