"""Decisions API (gpt-6-luna, no generated text) as a second coverage judge on every
field-norms arm, to compare with the generative gpt-6-luna judge.

    python datasets/dataset50/build/scripts/decisions_judge.py <arm|item> [workers]

arm   the 2026-10-13 pilot: the v2 judge prompt without its writing instructions, one call
      per proposal, one choice question per reference experiment.
item  one call per reference experiment, with only that experiment as the reference, so
      each verdict is made on its own rather than alongside the others.

Every call is cached under results/dataset50run_fn/decisions/<variant>/cache, so a rerun
only sends what is missing. Answers are deterministic (identical on repeated calls), so
there are no passes to vote over; each item keeps its choice and its probabilities.
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from norm_cards import config
from norm_cards.eval import load_claims, prompts, reference
from norm_cards.eval.evaluate import load_arm

R = "results/dataset50run_fn"
SOURCES = {"qwen3.8-27b": f"{R}/proposals_qwen", "gpt-5.6-terra": f"{R}/proposals_g56",
           "gpt-5.6-luna": f"{R}/proposals_g56"}
CLAIMS = load_claims("norm_cards/data/dataset50-claims-v1.jsonl")
KEY = config.openai_key()

INTRO = prompts.EVALUATION_SYSTEM_V2.split("For partial or missing")[0].strip()
TAIL = "For each reference experiment, decide whether a team running the whole proposed set would learn what it establishes."
CHOICES = [
    {"value": "covered", "description": "They would learn it, with comparable reliability. A different dataset, model, metric or procedure counts, unless the claim itself names it."},
    {"value": "partial", "description": "They would address it, but the answer would stay unreliable: no rule for deciding the outcome, a measurement that cannot tell the answers apart, a competing explanation left open, or too few runs to trust."},
    {"value": "missing", "description": "Nothing in the proposed set addresses it."}]


def call(path, body, tries=8):
    if os.path.exists(path):
        return json.load(open(path))
    err = None
    for t in range(tries):
        try:
            r = httpx.post("https://api.openai.com/v1/decisions", json=body, timeout=180,
                           headers={"Authorization": f"Bearer {KEY}"})
            if r.status_code == 200:
                d = r.json()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(d, open(path, "w"))
                return d
            err = f"{r.status_code} {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(err)
        except httpx.HTTPError as e:
            err = repr(e)
        time.sleep(min(60, 2 ** t))
    raise RuntimeError(err)


def user_text(claim, subclaims, refs, props, gt):
    u = prompts.EVALUATION_USER.format(
        domain=claim.get("domain", "ai"), problem_id=claim["problem_id"], claim=claim["claim"],
        subclaims="\n".join(f"{i}. {s}" for i, s in enumerate(subclaims, 1)) or "(none provided)",
        provenance=reference.provenance_line(gt), n_ref=len(refs),
        reference=reference.experiments_text(refs, "REFERENCE EXPERIMENT"),
        n=len(props), experiments=reference.experiments_text(props, "PROPOSED EXPERIMENT"))
    return u.split("For each reference experiment: state")[0].strip()


def question(name, i):
    return {"type": "choice", "name": name, "choices": CHOICES,
            "instructions": f"REFERENCE EXPERIMENT {i}: would a team running the whole proposed set learn what it establishes?"}


def judge(variant, pid, arm, model):
    claim = CLAIMS[pid]
    gt = json.load(open(f"{R}/eval/ground_truth/problem_{pid}/ground_truth.json"))
    refs = reference.experiments(gt)
    a = load_arm(SOURCES.get(model, f"{R}/proposals"), pid, arm)
    cache = f"{R}/decisions/{variant}/cache/{pid}/{arm}"
    if variant == "arm":
        body = {"model": "gpt-6-luna",
                "input": f"{INTRO}\n\n{user_text(claim, a['subclaims'], refs, a['experiments'], gt)}\n\n{TAIL}",
                "questions": [question(f"ref_{i}", i) for i in range(len(refs))]}
        ans = call(f"{cache}.json", body)["answers"]
    else:
        ans = []
        for i, ref in enumerate(refs):
            body = {"model": "gpt-6-luna",
                    "input": f"{INTRO}\n\n{user_text(claim, a['subclaims'], [ref], a['experiments'], gt)}\n\n{TAIL}",
                    "questions": [question("ref_0", 0)]}
            ans += call(f"{cache}.{i}.json", body)["answers"]
    assert len(ans) == len(refs), (pid, arm, len(ans), len(refs))
    out = {"problem_id": pid, "arm": arm, "proposer_model": model, "variant": variant,
           "n_proposed": len(a["experiments"]),
           "reference_coverage": [{"ref_index": i, "status": x["choice"],
                                   "probabilities": {p["value"]: p["probability"] for p in x["probabilities"]},
                                   "confidence": x.get("confidence")} for i, x in enumerate(ans)]}
    p = f"{R}/decisions/{variant}/judgments/problem_{pid}/{arm}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    return p


if __name__ == "__main__":
    variant, workers = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 48
    assert variant in ("arm", "item")
    jobs = []
    for f in sorted(glob.glob(f"{R}/eval/judgments/problem_*/*.json")):
        if ".trace" in f:
            continue
        j = json.load(open(f))
        if j.get("scores", {}).get("n_proposed"):
            jobs.append((j["problem_id"], j["arm"], j["proposer_model"]))
    print(f"{variant}: {len(jobs)} arms, {workers} workers", flush=True)
    done = fail = 0
    t0 = time.time()

    def one(job):
        try:
            return judge(variant, *job)
        except Exception as e:
            print(f"FAILED {job}: {type(e).__name__}: {e}", flush=True)
            return None

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, jobs):
            done += r is not None
            fail += r is None
            if (done + fail) % 100 == 0:
                print(f"  {done + fail}/{len(jobs)} ({fail} failed) {time.time() - t0:.0f}s", flush=True)
    print(f"{variant}: {done} judged, {fail} failed, {time.time() - t0:.0f}s")
