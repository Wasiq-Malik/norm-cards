"""Results data for the report's ACT 4, from the dataset50 judgments.

    D50_RUN=results/dataset50run_fn D50_PROPOSALS=proposals,proposals_qwen,proposals_g56 \
        python datasets/dataset50/build/scripts/report_data.py <out.json>

One VIEW per proposer plus "all" (pooled), so every per-claim, per-card, per-field and per-role
figure can be read for a single model: averaging over models hides a model the card does not
help. Each view has the same keys (summary / claims / per_card / metrics / by_field / by_role /
up_down). Static per-claim facts (title, venue, references, cards) sit once in `claims`.

Conventions as in every earlier revision: strict recall per (claim, model, condition); paired
bootstrap over (claim, model) pairs, 10,000 resamples, 95% percentile interval, seed 0. A
proposal the provider refused (no experiments) is left out, with its pair, not scored 0.
A proposer is reported once it has judged proposals on (nearly) all 100 claims.
"""
import collections
import glob
import json
import os
import random
import statistics as st
import sys

R = os.environ.get("D50_RUN", "results/dataset50run")         # results/dataset50run_fn: Field Norms prompt
SOURCES = os.environ.get("D50_PROPOSALS", "proposals,proposals_qwen,proposals_older").split(",")
JUDGE_RUNS = int(os.environ.get("D50_JUDGE_RUNS", "3"))
V = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
ORDER = ["gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra", "qwen3.8-27b", "gpt-5.6-terra", "gpt-5.6-luna"]
ARMS = ("nocard", "rag", "card")
OBLIGED = {"headline", "apparatus", "control", "confound"}
RIGOUR = {"control", "confound"}
FIELDS = ["robotics", "interpretability", "vision", "nlp", "math_mltheory"]
ROLES = ["headline", "apparatus", "control", "confound", "mechanism", "external"]

claims = {json.loads(l)["problem_id"]: json.loads(l) for l in open("norm_cards/data/dataset50-claims-v1.jsonl")}
sub = json.load(open("datasets/dataset50/subfields.json"))
roles = {p: json.load(open(f"{R}/eval/ground_truth/problem_{p}/ground_truth.json"))["roles"] for p in claims}

J = {}
for p in glob.glob(f"{R}/eval/judgments/problem_*/*.json"):
    if ".trace" in p or os.path.basename(p).startswith("_"):
        continue
    j = json.load(open(p))
    c = j["arm"].split("+", 1)[1]
    if c not in ARMS or not j.get("scores", {}).get("n_proposed"):
        continue                         # ablation arms, and proposals the provider refused
    J[(j["problem_id"], j["proposer_model"], c)] = j
have = collections.Counter(m for _, m, _ in J)
MODELS = [m for m in ORDER if have[m] >= 290]          # 100 claims x 3 conditions, a few refusals allowed


def items(j, keep=None):
    rs = roles[j["problem_id"]]
    return [c for i, c in enumerate(j["reference_coverage"]) if keep is None or rs[i] in keep]


def strict(j, keep=None):
    s = items(j, keep)
    return st.fmean(c["status"] == "covered" for c in s) if s else None


def recall(j, keep=None):
    # as scoring.py: an item the three passes split on (CONTESTED) scores the mean of its votes
    s = items(j, keep)
    return st.fmean((c.get("mean_status_score") or 0.0) if c["status"] == "CONTESTED" else V[c["status"]]
                    for c in s) if s else None


def boot(pairs, n=10000, seed=0):
    rnd = random.Random(seed)
    d = [b - a for a, b in pairs]
    if not d:
        return None
    bs = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(n))
    return {"d": round(st.fmean(d), 4), "lo": round(bs[int(n * .025)], 4),
            "hi": round(bs[int(n * .975)], 4), "n": len(d)}


def pairs(f, a, b, models, pids):
    out = []
    for p in pids:
        for m in models:
            x, y = J.get((p, m, a)), J.get((p, m, b))
            if x and y:
                vx, vy = f(x), f(y)
                if vx is not None and vy is not None:
                    out.append((vx, vy))
    return out


def mean_of(f, models, pids, arm):
    v = [f(j) for (p, m, a), j in J.items() if a == arm and m in models and p in pids]
    v = [x for x in v if x is not None]
    return round(st.fmean(v), 4) if v else None


def summ(f, models, pids):
    return {"means": {a: mean_of(f, models, pids, a) for a in ARMS},
            "deltas": {"card_vs_nocard": boot(pairs(f, "nocard", "card", models, pids)),
                       "card_vs_rag": boot(pairs(f, "rag", "card", models, pids)),
                       "rag_vs_nocard": boot(pairs(f, "nocard", "rag", models, pids))}}


def view(models):
    pids = list(claims)
    S = lambda j: strict(j)
    v = {"summary": {"full": summ(S, models, pids),
                     "obliged": summ(lambda j: strict(j, OBLIGED), models, pids),
                     "rigour": summ(lambda j: strict(j, RIGOUR), models, pids)}}
    cl = {}
    for p in pids:
        vals = {a: [S(J[(p, m, a)]) for m in models if (p, m, a) in J] for a in ARMS}
        ds = [S(J[(p, m, "card")]) - S(J[(p, m, "nocard")]) for m in models
              if (p, m, "card") in J and (p, m, "nocard") in J]
        cl[p] = {"v": {a: (round(st.fmean(x), 4) if x else None) for a, x in vals.items()},
                 "d": round(st.fmean(ds), 4) if ds else None}
    v["claims"] = cl
    cards = collections.defaultdict(list)
    for p, c in cl.items():
        for s in sub[p]["subfields"]:
            if c["d"] is not None:
                cards[s].append({"id": p, "d": c["d"]})
    v["per_card"] = sorted(({"name": s, "n": len(x), "mean": round(st.fmean(y["d"] for y in x), 4),
                             "lo": min(y["d"] for y in x), "hi": max(y["d"] for y in x), "claims": x}
                            for s, x in cards.items()), key=lambda r: -r["mean"])
    v["metrics"] = {
        "strict recall": {"sub": "a reference experiment counts only when the judge marks it fully covered",
                          "means": v["summary"]["full"]["means"], "delta": v["summary"]["full"]["deltas"]["card_vs_nocard"]},
        "recall": {"sub": "a partly addressed experiment counts as half",
                   "means": {a: mean_of(recall, models, pids, a) for a in ARMS},
                   "delta": boot(pairs(recall, "nocard", "card", models, pids))}}
    v["by_field"] = {}
    for f in FIELDS:
        fp_ = [p for p in pids if claims[p]["provenance"]["field"] == f]
        v["by_field"][f] = {"n": len(fp_), "means": {a: mean_of(S, models, fp_, a) for a in ARMS},
                            "card_vs_nocard": boot(pairs(S, "nocard", "card", models, fp_)),
                            "rag_vs_nocard": boot(pairs(S, "nocard", "rag", models, fp_))}
    v["by_role"] = {}
    for r in ROLES:
        cov = collections.defaultdict(list)
        for (p, m, a), j in J.items():
            if m in models:
                cov[a] += [c["status"] == "covered" for i, c in enumerate(j["reference_coverage"]) if roles[p][i] == r]
        v["by_role"][r] = {"n": len(cov["card"]), **{a: round(st.fmean(x), 4) for a, x in cov.items() if x},
                           "card_vs_nocard": boot(pairs(lambda j, r=r: strict(j, {r}), "nocard", "card", models, pids))}
    ds = [c["d"] for c in cl.values() if c["d"] is not None]
    v["up_down"] = {"up": sum(d > .0005 for d in ds), "down": sum(d < -.0005 for d in ds), "n": len(ds)}
    mx = max(cl, key=lambda p: cl[p]["d"] if cl[p]["d"] is not None else -9)
    mn = min(cl, key=lambda p: cl[p]["d"] if cl[p]["d"] is not None else 9)
    v["extremes"] = {"max": [mx, cl[mx]["d"]], "min": [mn, cl[mn]["d"]]}
    n_prop = [len(e) for d in PROPS for k, e in d.items() if k[0] in models]
    return v


PROPS = []
for src in SOURCES:
    d = {}
    for f in glob.glob(f"{R}/{src}/problem_*/scify_proposer.json"):
        for arm, a in json.load(open(f))["arms"].items():
            if a.get("experiments"):
                d[(arm.split("+")[0], arm.split("+", 1)[1], f)] = a["experiments"]
    PROPS.append(d)

D = {"models": MODELS, "judge_model": ["gpt-6-luna"], "judge_runs": [JUDGE_RUNS], "run": R,
     "views": {"all": view(MODELS), **{m: view([m]) for m in MODELS}}}
short = lambda v: "—" if "none" in v else v.split(" ")[0] + (" Findings" if "Findings" in v else " workshop" if "Workshop" in v else "")
D["claims"] = [{"id": p, "title": c["provenance"]["title"], "venue": short(c["provenance"]["venue"]),
                "field": c["provenance"]["field"], "n_ref": len(roles[p]), "subfields": sub[p]["subfields"]}
               for p, c in sorted(claims.items())]
D["mean_ref"] = round(st.fmean(len(r) for r in roles.values()), 2)
shape = {}
for m in MODELS:
    ex = [e for d in PROPS for k, e in d.items() if k[0] == m]
    shape[m] = {"mean_exp": round(st.fmean(len(e) for e in ex), 2),
                "full_budget": round(st.fmean(len(e) >= 9 for e in ex), 3),
                "chars": round(st.fmean(len(x) for e in ex for x in e)),
                **{f"exp_{c}": round(st.fmean(len(e) for k, e in [(k, e) for d in PROPS for k, e in d.items()]
                                              if k[0] == m and k[1] == c), 2) for c in ARMS}}
D["shape"] = shape
runs_ = [[r["reference_coverage"] for r in j.get("runs") or []] for (p, m, a), j in J.items() if m in MODELS]
items_ = [[run[i]["status"] for run in rs] for rs in runs_ if len(rs) >= 2 for i in range(min(len(x) for x in rs))]
if items_:
    D["agreement"] = {"unanimous": round(st.fmean(len(set(x)) == 1 for x in items_), 4), "n_items": len(items_),
                      "contested": sum(c["status"] == "CONTESTED" for j in J.values() if j["proposer_model"] in MODELS
                                       for c in j["reference_coverage"])}
D["refused"] = sorted(f"{p} {m}+{a}" for p in claims for m in MODELS for a in ARMS
                      if os.path.exists(f"{R}/eval/judgments/problem_{p}/{m}+{a}.json") and (p, m, a) not in J)
json.dump(D, open(sys.argv[1], "w"), indent=1)
for k in ["all"] + MODELS:
    f = D["views"][k]["summary"]
    print(f"{k:14s} means {f['full']['means']} | card-nocard {f['full']['deltas']['card_vs_nocard']} "
          f"| ctl+conf {f['rigour']['deltas']['card_vs_nocard']}")
