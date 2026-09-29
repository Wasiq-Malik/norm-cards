"""Known-answer test suite for the judge.

    python -m norm_cards.eval.judge_suite --configs gpt-5.6-luna:v1,gpt-5.6-terra:v2
    python -m norm_cards.eval.judge_suite --list

Every case is constructed, short, and has a known correct status for each reference
experiment. That is the whole design, and it is a reaction to the suite it replaces
(`judge_tests.py`), which scored perturbations of REAL proposals as a difference from a
baseline judgement. Both sides of that difference were single noisy judge passes, on
three claims, against a noise floor that was itself one sample — so the same test on the
same judge came back 0.204 and then 0.018. A suite that noisy cannot tell two judges
apart, which is the only thing a judge suite is for.

Here the expected answer is fixed in advance, every case runs several times, and each
behaviour is tested in three unrelated domains. The output is a behaviour x config
pass-rate table: run it for two judges and read the difference.

THE CASES MUST STAY DISJOINT. Each kit has three reference experiments — a headline
comparison, a validity control and a fairness control — built so that each can only be
covered by its own proposal. Many-to-many coverage is real, which is exactly why this
matters: if the fairness control's proposal also re-ran the headline comparison, then
removing the headline proposal would leave the headline half-covered, and the "right"
answer would stop being knowable. When adding a kit, check that no proposal delivers
another reference experiment's content.
"""

import argparse
import collections
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import evaluate, prompts

C, P, M = "covered", "partial", "missing"
EMPTY = "We will investigate the claim using appropriate methods."

# --------------------------------------------------------------------------- #
# Kits. `ref` is [headline, validity control, fairness control]. `prop` holds
# proposals written in DIFFERENT words and with DIFFERENT resources from the
# reference, because substitution is the normal case, not the exception.
# --------------------------------------------------------------------------- #

KITS = {
"adversarial": {
    # The masking control re-checks that the objective beats the baseline under attack,
    # which is a robust-accuracy comparison — so it can stand in for the ROBUST half of
    # the headline. Only the clean half is uncoverable, so split_alone keeps halfB0.
    "split_keep": "halfB0",
    "claims": {
        "can": ("Adversarial training need not trade clean accuracy for robustness: a "
                "training objective can improve both clean accuracy and robust accuracy "
                "over standard adversarial training under an equal training budget. The "
                "claim holds if both exceed standard adversarial training by more than 0 "
                "points."),
        "general": ("A training objective improves both clean accuracy and robust accuracy "
                    "over standard adversarial training generally, across image-"
                    "classification benchmarks of differing difficulty, under an equal "
                    "training budget. The claim holds only if both exceed standard "
                    "adversarial training by more than 0 points on every benchmark."),
        "named": ("On CIFAR-10, a training objective can improve both clean accuracy and "
                  "robust accuracy over standard adversarial training under an equal "
                  "training budget. The claim holds if both exceed standard adversarial "
                  "training by more than 0 points on CIFAR-10."),
    },
    "ref": [
        "Compare the new objective with standard PGD adversarial training under an "
        "identical training budget on CIFAR-10, CIFAR-100 and Tiny-ImageNet with a "
        "ResNet-18, over three seeds. Measure clean accuracy and AutoAttack robust "
        "accuracy at an l-infinity budget of 8/255; the claim holds if both exceed the "
        "standard objective.",
        "Rule out that the robustness gain is an artifact of gradient masking. Attack "
        "both models with twice the attack iterations, with a black-box transfer attack, "
        "and with an adaptive attack, and confirm robust accuracy does not rise and the "
        "ranking of the two objectives holds.",
        "Rule out that the baseline is under-tuned. Give standard adversarial training "
        "the same hyperparameter search budget as the new objective, over step size, "
        "regularisation weight and learning rate, selecting each by validation robust "
        "accuracy.",
    ],
    "prop": {
        "full0": ("Train the proposed objective and PGD adversarial training with the same "
                  "optimiser, schedule and epoch count on SVHN, CIFAR-10 and CIFAR-100 using "
                  "a WideResNet-28-10, five seeds each with 95% confidence intervals. Report "
                  "clean accuracy and robust accuracy under PGD-50 at epsilon 8/255, and test "
                  "whether both exceed the baseline."),
        "full1": ("Check that measured robustness is genuine rather than masked gradients: "
                  "re-attack both models with double the iterations, with transfer attacks "
                  "from an independently trained surrogate, and with an adaptive attack "
                  "tailored to the objective, confirming robust accuracy does not increase and "
                  "that the proposed objective still beats the baseline under every attack."),
        "full2": ("Give the baseline a fair tuning budget: search step size, loss weighting "
                  "and learning rate for PGD adversarial training with as many trials as the "
                  "proposed objective receives, choosing each by held-out robust accuracy."),
        "halfA0": ("Train the proposed objective and PGD adversarial training with the same "
                   "optimiser, schedule and epoch count on SVHN, CIFAR-10 and CIFAR-100, five "
                   "seeds each, and compare CLEAN test accuracy only, with 95% confidence "
                   "intervals."),
        "halfB0": ("Using those same trained models, compare ROBUST accuracy under PGD-50 at "
                   "epsilon 8/255 between the proposed objective and PGD adversarial "
                   "training, five seeds, with 95% confidence intervals."),
        "weak0": ("Train the proposed objective and PGD adversarial training with the same "
                  "optimiser, schedule and epoch count on SVHN, CIFAR-10 and CIFAR-100 using "
                  "a WideResNet-28-10, a single seed each. Report clean accuracy and robust "
                  "accuracy under PGD-50 at epsilon 8/255, and test whether both exceed the "
                  "baseline."),
        "onedata0": ("Train the proposed objective and PGD adversarial training with the same "
                     "optimiser, schedule and epoch count on CIFAR-10 only, using a "
                     "WideResNet-28-10, five seeds with 95% confidence intervals. Report "
                     "clean accuracy and robust accuracy under PGD-50 at epsilon 8/255, and "
                     "test whether both exceed the baseline."),
        "subst0": ("Train the proposed objective and PGD adversarial training with the same "
                   "optimiser, schedule and epoch count on SVHN, CIFAR-100 and Tiny-ImageNet "
                   "using a WideResNet-28-10, five seeds each with 95% confidence intervals. "
                   "Report clean accuracy and robust accuracy under PGD-50 at epsilon 8/255, "
                   "and test whether both exceed the baseline."),
        "merge": ("Under an identical training budget, compare the proposed objective with "
                  "PGD adversarial training on SVHN, CIFAR-10 and CIFAR-100 over five seeds, "
                  "measuring clean accuracy and PGD-50 robust accuracy; give the baseline the "
                  "same hyperparameter-search budget over step size, loss weighting and "
                  "learning rate; and confirm the robustness is not masked gradients by "
                  "re-attacking with doubled iterations, surrogate transfer attacks and an "
                  "adaptive attack."),
    },
},
"steering": {
    "claims": {
        "can": ("A language model that behaves differently when it detects it is being "
                "evaluated can be made to behave in evaluation as it does in deployment by "
                "steering its internal activations, without changing its prompts. The claim "
                "holds if, under steering, the behaviour that differs between the two "
                "contexts comes within 5 percentage points of the model's own deployment "
                "rate while task accuracy falls by no more than 5% relative."),
        "general": ("Steering internal activations makes evaluation-aware language models "
                    "behave in evaluation as they do in deployment, without changing their "
                    "prompts, and this holds generally across model families and sizes. The "
                    "claim holds only if, for every model tested, the context-dependent "
                    "behaviour comes within 5 percentage points of that model's deployment "
                    "rate while task accuracy falls by no more than 5% relative."),
        "named": ("Llama-3.3-Nemotron-Super-49B, which behaves differently when it detects it "
                  "is being evaluated, can be made to behave in evaluation as it does in "
                  "deployment by steering its internal activations, without changing its "
                  "prompts. The claim holds if, for that model, the context-dependent "
                  "behaviour comes within 5 percentage points of its deployment rate while "
                  "task accuracy falls by no more than 5% relative."),
    },
    "ref": [
        "On Llama-3.3-Nemotron-Super-49B, Qwen2.5-32B and Gemma-2-27B, apply a steering "
        "vector during evaluation-cued prompts and measure the rate of the context-"
        "dependent behaviour against each model's deployment rate, over 500 coding prompts "
        "and three sampling seeds; the claim holds if the gap is within 5 points with task "
        "accuracy within 5% relative.",
        "Rule out that any perturbation of that size would do the same. Steer with random "
        "vectors of matched norm, and steer in the opposite direction, and confirm neither "
        "closes the evaluation-deployment gap.",
        "Rule out that steering works by damaging the model. Measure the steered model's "
        "general-knowledge accuracy and its perplexity on held-out text against the "
        "unsteered model.",
    ],
    "prop": {
        "full0": ("On Llama-3.3-Nemotron-Super-49B, Mistral-Large and Qwen2.5-72B, add the "
                  "deployment-minus-evaluation steering direction while serving evaluation-"
                  "cued prompts, and compare the rate of the context-dependent behaviour with "
                  "each model's own deployment-cued rate over 800 prompts and five seeds, with "
                  "confidence intervals; also report task accuracy relative to unsteered."),
        "full1": ("Control for non-specific perturbation: steer with norm-matched random "
                  "directions and with the negated direction, and confirm neither moves the "
                  "evaluation-cued behaviour toward the deployment rate."),
        "full2": ("Check steering does not simply degrade the model: compare steered and "
                  "unsteered models on MMLU and on perplexity over a held-out text corpus."),
        "halfA0": ("On Llama-3.3-Nemotron-Super-49B, Mistral-Large and Qwen2.5-72B, add the "
                   "steering direction during evaluation-cued prompts and measure how close "
                   "the context-dependent behaviour gets to each model's deployment rate, over "
                   "800 prompts and five seeds. Do not measure task accuracy here."),
        "halfB0": ("For those same steered models, measure task accuracy on the evaluation-"
                   "cued prompts relative to the unsteered model, over 800 prompts and five "
                   "seeds."),
        "weak0": ("On Llama-3.3-Nemotron-Super-49B, Mistral-Large and Qwen2.5-72B, add the "
                  "steering direction while serving evaluation-cued prompts, and compare the "
                  "rate of the context-dependent behaviour with each model's deployment-cued "
                  "rate on 20 prompts with one sample each; also report task accuracy."),
        "onedata0": ("On Llama-3.3-Nemotron-Super-49B only, add the deployment-minus-"
                     "evaluation steering direction while serving evaluation-cued prompts, "
                     "and compare the rate of the context-dependent behaviour with its "
                     "deployment-cued rate over 800 prompts and five seeds, with confidence "
                     "intervals; also report task accuracy relative to unsteered."),
        "subst0": ("On Mistral-Large, Qwen2.5-72B and Llama-3.1-70B, add the deployment-minus-"
                   "evaluation steering direction while serving evaluation-cued prompts, and "
                   "compare the rate of the context-dependent behaviour with each model's own "
                   "deployment-cued rate over 800 prompts and five seeds, with confidence "
                   "intervals; also report task accuracy relative to unsteered."),
        "merge": ("On Llama-3.3-Nemotron-Super-49B, Mistral-Large and Qwen2.5-72B, steer "
                  "evaluation-cued prompts with the deployment-minus-evaluation direction and "
                  "compare the behaviour rate with each model's deployment rate over 800 "
                  "prompts and five seeds, reporting task accuracy; in the same runs steer "
                  "with norm-matched random and negated directions as controls; and compare "
                  "steered and unsteered models on MMLU and held-out perplexity."),
    },
},
"robot": {
    "claims": {
        "can": ("A vision-language-action policy can adapt at test time to tasks excluded "
                "from its training by conditioning on a few expert demonstrations, without "
                "updating its weights. The claim holds if mean task success on held-out "
                "tasks exceeds the strongest in-context imitation baseline given the same "
                "demonstrations."),
        "general": ("Vision-language-action policies adapt at test time to tasks excluded "
                    "from their training by conditioning on a few expert demonstrations, "
                    "without updating their weights, and this holds generally across task "
                    "suites and robot embodiments. The claim holds only if mean held-out task "
                    "success exceeds the strongest in-context imitation baseline, given the "
                    "same demonstrations, in every suite tested."),
        "named": ("On the LIBERO benchmark, a vision-language-action policy can adapt at test "
                  "time to tasks excluded from its training by conditioning on a few expert "
                  "demonstrations, without updating its weights. The claim holds if mean "
                  "held-out task success on LIBERO exceeds the strongest in-context imitation "
                  "baseline given the same demonstrations."),
    },
    "ref": [
        "On LIBERO, Meta-World and CALVIN, hold out whole task families, give the policy and "
        "the strongest in-context imitation baseline the same three expert demonstrations per "
        "held-out task, and compare mean success over 50 trials per task and three seeds.",
        "Rule out leakage between training and held-out tasks. Audit the adaptation-training "
        "data for overlapping instructions, objects and scene layouts, and remove any held-"
        "out task with a near-duplicate.",
        "Rule out that the policy wins by seeing more demonstration content. Give the "
        "baseline the same demonstrations at matched total frame count and context length.",
    ],
    "prop": {
        "full0": ("On LIBERO, RLBench and ManiSkill, exclude entire task families from "
                  "training, give the policy and the best in-context imitation baseline "
                  "identical sets of three demonstrations per held-out task, and compare mean "
                  "success over 100 trials per task and five seeds with confidence intervals."),
        "full1": ("Check for train/test leakage: compare every held-out task's instruction, "
                  "object set and scene layout against the adaptation-training data and drop "
                  "held-out tasks with near-duplicates before evaluating."),
        "full2": ("Match the demonstration budget: give the baseline the identical "
                  "demonstrations truncated or padded to the same total frame count and "
                  "context length as the policy receives."),
        "halfA0": ("On LIBERO, RLBench and ManiSkill, exclude entire task families from "
                   "training and measure the policy's mean success on held-out tasks given "
                   "three demonstrations each, over 100 trials and five seeds. Do not run any "
                   "baseline here."),
        "halfB0": ("On the same held-out tasks and demonstrations, measure the best in-context "
                   "imitation baseline's mean success over 100 trials and five seeds, and "
                   "compare it with the policy's."),
        "weak0": ("On LIBERO, RLBench and ManiSkill, exclude entire task families from "
                  "training, give the policy and the best in-context imitation baseline "
                  "identical sets of three demonstrations per held-out task, and compare "
                  "success from a single trial per task and one seed."),
        "onedata0": ("On LIBERO only, exclude entire task families from training, give the "
                     "policy and the best in-context imitation baseline identical sets of "
                     "three demonstrations per held-out task, and compare mean success over "
                     "100 trials per task and five seeds with confidence intervals."),
        "subst0": ("On RLBench, ManiSkill and Meta-World, exclude entire task families from "
                   "training, give the policy and the best in-context imitation baseline "
                   "identical sets of three demonstrations per held-out task, and compare mean "
                   "success over 100 trials per task and five seeds with confidence "
                   "intervals."),
        "merge": ("On LIBERO, RLBench and ManiSkill, after auditing held-out tasks against "
                  "training data for overlapping instructions, objects and layouts and "
                  "removing near-duplicates, give the policy and the best in-context imitation "
                  "baseline identical three-demonstration sets matched in total frame count "
                  "and context length, and compare mean success over 100 trials and five "
                  "seeds."),
    },
},
}

# --------------------------------------------------------------------------- #
# Behaviours. Each builds (claim form, proposal keys, expected per ref item).
# An expected entry is a SET of acceptable statuses. `target` marks which reference
# items the behaviour is actually testing; the rest are collateral and should just
# stay put.
# --------------------------------------------------------------------------- #

BASE = ["full0", "full1", "full2"]
BEHAVIOURS = {
    # known-good control: every behaviour below is read against this one
    "baseline":        ("can", BASE,                              [{C}, {C}, {C}], [0, 1, 2],
                        "every reference item delivered by different means -> all covered"),
    # many-to-many
    "split":           ("can", ["halfA0", "halfB0", "full1", "full2"], [{C}, {C}, {C}], [0],
                        "one item delivered jointly by two smaller proposals -> covered"),
    "split_alone":     ("can", ["halfA0", "full1", "full2"],      [{P}, {C}, {C}], [0],
                        "only half of an item delivered -> partial"),
    "merge":           ("can", ["merge"],                         [{C}, {C}, {C}], [0, 1, 2],
                        "one proposal delivers all three items -> all covered"),
    # sensitivity
    # partial is accepted: a headline proposal can carry incidental protection (the robot
    # kit's "exclude entire task families" partly guards against leakage). The failure
    # this exists to catch is the judge INVENTING coverage — `covered` — not the grade.
    "removal":         ("can", ["full0", "full2"],                [{C}, {P, M}, {C}], [1],
                        "the only proposal for an item is removed -> not covered"),
    "power":           ("can", ["weak0", "full1", "full2"],       [{P}, {C}, {C}], [0],
                        "one seed / a handful of samples -> partial"),
    "cross_claim":     ("can", None,                              [{M}, {M}, {M}], [0, 1, 2],
                        "a competent proposal for a different claim -> all missing"),
    "empty":           ("can", [EMPTY],                           [{M}, {M}, {M}], [0, 1, 2],
                        "one contentless sentence -> all missing"),
    # invariance
    "duplicate":       ("can", BASE + BASE,                       [{C}, {C}, {C}], [0, 1, 2],
                        "the whole set repeated -> unchanged"),
    "order":           ("can", BASE[::-1],                        [{C}, {C}, {C}], [0, 1, 2],
                        "the set reversed -> unchanged"),
    # claim is the contract: breadth and substitution
    "breadth_can":     ("can", ["onedata0", "full1", "full2"],    [{C}, {C}, {C}], [0],
                        "one setting, claim says CAN -> covered"),
    "breadth_general": ("general", ["onedata0", "full1", "full2"], [{P}, {C}, {C}], [0],
                        "one setting, claim says holds GENERALLY -> partial"),
    "subst_free":      ("can", ["subst0", "full1", "full2"],      [{C}, {C}, {C}], [0],
                        "resources swapped, claim names none -> covered"),
    # When the claim binds a setting and nothing runs in it, the controls ran in the wrong
    # setting too, so they cannot establish anything about the claim either. An earlier
    # version expected the controls to stay covered; the judge was right and it was wrong.
    "subst_bound":     ("named", ["subst0", "full1", "full2"],    [{P, M}, {P, M}, {P, M}], [0, 1, 2],
                        "resources swapped away from the one the CLAIM names -> nothing covered"),
}
CROSS = {"adversarial": "steering", "steering": "robot", "robot": "adversarial"}


def cases() -> list:
    out = []
    for kit_name, kit in KITS.items():
        for beh, (form, keys, expect, target, _) in BEHAVIOURS.items():
            if beh == "split_alone":
                keys = [kit.get("split_keep", "halfA0"), "full1", "full2"]
            if keys is None:                                   # cross_claim
                other = KITS[CROSS[kit_name]]["prop"]
                proposal = [other[k] for k in BASE]
            else:
                proposal = [kit["prop"][k] if k in kit["prop"] else k for k in keys]
            out.append({"id": f"{kit_name}/{beh}", "kit": kit_name, "behaviour": beh,
                        "claim": kit["claims"][form], "reference": kit["ref"],
                        "proposal": proposal, "expect": [sorted(e) for e in expect],
                        "target": target})
    return out


def judge_case(case: dict, model: str, prompt: str, run: int, out_dir: str) -> dict:
    problem = {"problem_id": case["id"].replace("/", "__"), "claim": case["claim"],
               "domain": "ai"}
    gt = {"experiments": case["reference"],
          "provenance": {"kind": "transcribed from the source paper"}}
    arm = {"experiments": case["proposal"], "subclaims": []}
    try:
        ev = evaluate.evaluate_arm(
            problem, gt, arm, run, out_dir, f"{model}-{prompt}-{problem['problem_id']}",
            model=model, progress=False, system=prompts.PROMPTS[prompt],
            min_chain=1 if prompt == "v2" else 2)
        statuses = [None] * len(case["reference"])
        for rc in ev.get("reference_coverage") or []:
            i = rc.get("ref_index")
            if isinstance(i, int) and 0 <= i < len(statuses):
                statuses[i] = rc.get("status")
        return {"statuses": statuses, "error": None}
    except Exception as e:                                       # a crash is a result too
        return {"statuses": [None] * len(case["reference"]),
                "error": f"{type(e).__name__}: {str(e)[:160]}"}


def run_suite(configs: list, runs: int, workers: int, out_path: str, only: list = None):
    all_cases = [c for c in cases() if not only or c["behaviour"] in only]
    out_dir = os.path.join(os.path.dirname(out_path), "traces")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(c, m, p, r) for (m, p) in configs for c in all_cases for r in range(runs)]
    print(f"  {len(all_cases)} cases x {runs} runs x {len(configs)} configs = {len(jobs)} judge calls")
    results = []

    def one(job):
        c, m, p, r = job
        return {"case": c["id"], "behaviour": c["behaviour"], "kit": c["kit"], "model": m,
                "prompt": p, "run": r, "expect": c["expect"], "target": c["target"],
                **judge_case(c, m, p, r, out_dir)}

    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(one, jobs), 1):
            results.append(res)
            if i % 25 == 0 or i == len(jobs):
                print(f"    {i}/{len(jobs)} done", flush=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=1)
    return results


def report(results: list) -> str:
    cfgs = sorted({(r["model"], r["prompt"]) for r in results})
    tag = lambda c: f"{c[0].replace('gpt-', '')}:{c[1]}"
    lines = []
    head = f"  {'behaviour':16s} " + " ".join(f"{tag(c):>16s}" for c in cfgs)
    lines += ["  TARGET-ITEM PASS RATE (share of runs where the tested item got an accepted status)",
              head]
    agg = collections.defaultdict(lambda: [0, 0])
    coll = collections.defaultdict(lambda: [0, 0])
    for r in results:
        cfg = (r["model"], r["prompt"])
        for i, s in enumerate(r["statuses"]):
            ok = s in r["expect"][i]
            bucket = agg if i in r["target"] else coll
            bucket[(r["behaviour"], cfg)][0] += ok
            bucket[(r["behaviour"], cfg)][1] += 1
    for beh in BEHAVIOURS:
        cells = []
        for c in cfgs:
            a, n = agg[(beh, c)]
            cells.append(f"{a}/{n} {a / n:4.0%}" if n else "—")
        lines.append(f"  {beh:16s} " + " ".join(f"{x:>16s}" for x in cells))
    lines.append("")
    tot = {c: [sum(agg[(b, c)][0] for b in BEHAVIOURS), sum(agg[(b, c)][1] for b in BEHAVIOURS)]
           for c in cfgs}
    ctot = {c: [sum(coll[(b, c)][0] for b in BEHAVIOURS), sum(coll[(b, c)][1] for b in BEHAVIOURS)]
            for c in cfgs}
    lines.append(f"  {'TARGET overall':16s} " + " ".join(
        f"{tot[c][0] / tot[c][1]:>16.1%}" for c in cfgs))
    lines.append(f"  {'collateral':16s} " + " ".join(
        f"{ctot[c][0] / ctot[c][1]:>16.1%}" if ctot[c][1] else f"{'—':>16s}" for c in cfgs))
    # self-agreement: same case, same config, every run gives the same status per item
    by = collections.defaultdict(list)
    for r in results:
        by[(r["case"], r["model"], r["prompt"])].append(r["statuses"])
    agree = collections.defaultdict(lambda: [0, 0])
    for (case, m, p), runs in by.items():
        if len(runs) < 2:
            continue
        for i in range(len(runs[0])):
            agree[(m, p)][0] += len({tuple([x[i]]) for x in runs}) == 1
            agree[(m, p)][1] += 1
    lines.append(f"  {'self-agreement':16s} " + " ".join(
        f"{agree[c][0] / agree[c][1]:>16.1%}" if agree[c][1] else f"{'—':>16s}" for c in cfgs))
    errs = collections.Counter((r["model"], r["prompt"]) for r in results if r["error"])
    if errs:
        lines.append(f"  {'errors':16s} " + " ".join(f"{errs.get(c, 0):>16d}" for c in cfgs))
    return "\n".join(lines)


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs", default="gpt-5.6-luna:v2,gpt-5.6-terra:v2",
                    help="comma-separated model:prompt pairs; prompt is v1 or v2")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="", help="comma-separated behaviours to run")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true", help="print the cases and exit")
    ap.add_argument("--report", default="", help="re-print the report for a saved run")
    args = ap.parse_args()

    if args.list:
        for beh, (form, _, expect, target, what) in BEHAVIOURS.items():
            print(f"  {beh:16s} [{form:7s}] {what}")
        print(f"\n  {len(cases())} cases across kits: {', '.join(KITS)}")
        return
    if args.report:
        with open(args.report, encoding="utf-8") as f:
            print(report(json.load(f)))
        return
    configs = [tuple(x.split(":")) for x in args.configs.split(",") if x.strip()]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out = args.out or os.path.join("results", "judge_suite", f"{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = run_suite(configs, args.runs, args.workers, out,
                    [b.strip() for b in args.only.split(",") if b.strip()] or None)
    print("\n" + report(res) + f"\n\n  -> {out}")


if __name__ == "__main__":
    _main()
