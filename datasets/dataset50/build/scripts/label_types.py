"""Label every dataset50 reference experiment with one experiment type, K samples per claim
with gpt-6-luna, majority vote per item. The six drafting roles stay in ground_truth.json;
the types go to datasets/dataset50/build/types.json.

    python datasets/dataset50/build/scripts/label_types.py [K] [workers]

Types, asked in order (the first yes wins, so every item gets exactly one):
  setup     builds data, models, metrics or the method, with no outcome of its own
  headline  the claim's own test (by main purpose), including settings and analyses the bar names
  threat    a threat to validity: controls and confounds, robustness and sensitivity checks,
            measurement checks, and caveat notes about how reliable the evidence is; stays a
            threat when the claim's bar names it, unless the claim's own statement asserts it
  analysis  ablations and mechanism checks the claim does not require
  other     whatever is left: new settings the claim does not name, pure scope notes
Scored recall uses headline, threat and analysis; setup and other are not scored.
"""
import collections
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from norm_cards import llm

MODEL = "gpt-6-luna"
TYPES = ("setup", "threat", "analysis", "other", "headline")
GT = "results/dataset50run_fn/eval/ground_truth"
OUT = "datasets/dataset50/build/types.json"

PROMPT = """You are sorting the reference experiments of a scientific claim into types.

CLAIM (with the bar it must pass):
{claim}

REFERENCE EXPERIMENTS:
{refs}

Give each reference experiment exactly one type, judged by its main purpose, not by every detail it includes (a main comparison that also reports seeds or includes a random baseline is still a main comparison). The CLAIM has two parts: its statement (before "Supported only if") and its bar (the conditions after it). Ask these questions in order and take the first that answers yes:

1. setup: Does it only build or fix the data, models, metrics, benchmark, training recipe or the method itself, with no outcome of its own? (Defining a procedure that later experiments use is setup, even if it reports a small sanity number.)
2. headline: Is its main purpose the claim's own test? That is: measuring the claimed effect or the comparison the claim makes (including against existing or published methods and any alternative the claim's statement names); a new setting (models, sizes, datasets, tasks, hardware) the claim names; an ablation or mechanism measurement the claim names as something it must show; cost when the claim is about cost. If its main purpose is one of the checks in question 3, answer no here even when the claim's bar names it.
3. threat: Is its main purpose to address a threat to the validity of the result: something other than the claimed cause that could produce it, or a reason the measured result could be unreliable? This covers:
   - controls: random, shuffled, sham, null or no-treatment versions in which the claimed cause is absent; a setting where the effect should be absent or smaller;
   - confounds: baselines matched for compute, steps, data or tuning; inputs that remove a confounding cue; checks that the gain does not come from extra capacity, more updates, a different backbone or a weaker baseline implementation;
   - measurement: leakage, contamination or train/test overlap; validating a judge, annotator or metric; checking an assumption the analysis relies on; side effects such as false alarms or harm to unrelated behaviour;
   - robustness: seeds and variance, sensitivity to hyperparameters, thresholds or other chosen settings ("not one lucky setting"), prompt paraphrases or sampling settings;
   - caveat notes that point to a weakness of the evidence (unequal tuning, single runs, test-set selection, re-implemented or copied baselines, weak proxies, gaps between theory and experiment), even with no test of their own.
   Removing one of the method's own components is an ablation (question 4), not a control.
4. analysis: Does it remove or replace a component of the proposed method to see which part does the work (an ablation), or measure the method's internals, training dynamics or failure cases to explain why it works?
5. other: Anything else, such as rerunning the test in a new setting the claim does not name, or a note about scope that does not question the evidence.

Return JSON: {{"types": [{{"index": 1, "type": "<setup|threat|analysis|other|headline>"}}, ...]}} with one entry per reference experiment, in order."""


def one(job):
    pid, r = job
    gt = json.load(open(f"{GT}/problem_{pid}/ground_truth.json"))
    refs = "\n\n".join(f"{i}. {e}" for i, e in enumerate(gt["experiments"], 1))
    for _ in range(3):
        try:
            d = llm.complete_json(PROMPT.format(claim=gt["claim"], refs=refs), model=MODEL)
            t = [x["type"] for x in sorted(d["types"], key=lambda x: x["index"])]
            if len(t) == len(gt["experiments"]) and all(x in TYPES for x in t):
                return pid, t
        except Exception as e:
            print(f"  {pid} pass {r}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    return pid, None


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    pids = sorted(p.split("problem_")[1] for p in glob.glob(f"{GT}/problem_*"))
    votes = collections.defaultdict(list)
    with ThreadPoolExecutor(workers) as ex:
        for pid, t in ex.map(one, [(p, r) for p in pids for r in range(K)]):
            if t:
                votes[pid].append(t)
    out = {}
    for pid in pids:
        gt = json.load(open(f"{GT}/problem_{pid}/ground_truth.json"))
        items = []
        for i, role in enumerate(gt["roles"]):
            v = [t[i] for t in votes[pid]]
            top, n = collections.Counter(v).most_common(1)[0] if v else (None, 0)
            items.append({"role": role, "type": top if n * 2 > len(v) else "TIE", "votes": v})
        out[pid] = items
    json.dump(out, open(OUT, "w"), indent=1)
    c = collections.Counter(x["type"] for v in out.values() for x in v)
    print(f"{sum(map(len, votes.values()))}/{len(pids) * K} passes; types: {dict(c)}")
