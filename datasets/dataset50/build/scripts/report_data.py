"""Results data for the report's ACT 4, from the dataset50 judgments.

    python datasets/dataset50/build/scripts/report_data.py <out.json>

Keys and shapes match the report's existing charts (summary / claims / per_card / metrics /
shape), plus dataset50-only keys: by_field, by_role and the template-search card ablation.
Conventions as in every earlier revision: strict recall per (claim, model, condition); paired
bootstrap over (claim, model) pairs, 10,000 resamples, 95% percentile interval, seed 0. A
proposal the provider refused (no experiments) is left out, with its pair, not scored 0.
"""
import collections
import glob
import json
import os
import random
import statistics as st
import sys

R = "results/dataset50run"
V = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
MODELS = ["gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra"]
OBLIGED = {"headline", "apparatus", "control", "confound"}
RIGOUR = {"control", "confound"}


def cond(arm: str) -> str:
    c = arm.split("+", 1)[1]
    return "tmpl" if c == "card:template" else c


claims = {json.loads(l)["problem_id"]: json.loads(l) for l in open("norm_cards/data/dataset50-claims-v1.jsonl")}
sub = json.load(open("datasets/dataset50/subfields.json"))
roles = {p: json.load(open(f"{R}/eval/ground_truth/problem_{p}/ground_truth.json"))["roles"] for p in claims}

J = {}
for p in glob.glob(f"{R}/eval/judgments/problem_*/*.json"):
    if ".trace" in p or os.path.basename(p).startswith("_"):
        continue
    j = json.load(open(p))
    if not j.get("scores", {}).get("n_proposed"):
        continue                                  # refused by the provider: no proposal to score
    J[(j["problem_id"], j["proposer_model"], cond(j["arm"]))] = j


def items(j, keep=None):
    rs = roles[j["problem_id"]]
    return [c for i, c in enumerate(j["reference_coverage"]) if keep is None or rs[i] in keep]


def strict(j, keep=None):
    s = items(j, keep)
    return st.fmean(c["status"] == "covered" for c in s) if s else None


def recall(j, keep=None):
    # as scoring.py: an item the three passes split on (CONTESTED) scores the mean of its votes
    s = items(j, keep)
    return st.fmean(c.get("mean_status_score") or 0.0 if c["status"] == "CONTESTED" else V[c["status"]]
                    for c in s) if s else None


def boot(pairs, n=10000, seed=0):
    rnd = random.Random(seed)
    d = [b - a for a, b in pairs]
    if not d:
        return None
    bs = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(n))
    return {"d": round(st.fmean(d), 4), "lo": round(bs[int(n * .025)], 4),
            "hi": round(bs[int(n * .975)], 4), "n": len(d)}


def pairs(f, a, b, models=MODELS, pids=None):
    out = []
    for p in (pids or claims):
        for m in models:
            x, y = J.get((p, m, a)), J.get((p, m, b))
            if x and y:
                vx, vy = f(x), f(y)
                if vx is not None and vy is not None:
                    out.append((vx, vy))
    return out


def summ(f, arms=("nocard", "rag", "card", "tmpl"), pids=None):
    pids = pids or list(claims)
    means = {a: round(st.fmean(v for k, j in J.items() if k[2] == a and k[0] in pids
                               for v in [f(j)] if v is not None), 4)
             for a in arms if any(k[2] == a for k in J)}
    per_model = {m: {a: round(st.fmean(v for k, j in J.items() if k[1] == m and k[2] == a and k[0] in pids
                                       for v in [f(j)] if v is not None), 4)
                     for a in arms if any(k[2] == a and k[1] == m for k in J)} for m in MODELS}
    deltas = {"card_vs_nocard": boot(pairs(f, "nocard", "card", pids=pids)),
              "card_vs_rag": boot(pairs(f, "rag", "card", pids=pids)),
              "rag_vs_nocard": boot(pairs(f, "nocard", "rag", pids=pids)),
              "card_vs_tmpl": boot(pairs(f, "tmpl", "card", pids=pids)),
              "tmpl_vs_nocard": boot(pairs(f, "nocard", "tmpl", pids=pids))}
    for m in MODELS:
        deltas[f"card_vs_nocard__{m}"] = boot(pairs(f, "nocard", "card", [m], pids))
        deltas[f"card_vs_rag__{m}"] = boot(pairs(f, "rag", "card", [m], pids))
        deltas[f"card_vs_tmpl__{m}"] = boot(pairs(f, "tmpl", "card", [m], pids))
    return {"means": means, "per_model": per_model, "deltas": deltas}


def per_claim(f, p, a):
    v = [f(J[(p, m, a)]) for m in MODELS if (p, m, a) in J]
    v = [x for x in v if x is not None]
    return round(st.fmean(v), 4) if v else None


def claim_delta(f, p, a="nocard", b="card"):
    d = [f(J[(p, m, b)]) - f(J[(p, m, a)]) for m in MODELS if (p, m, a) in J and (p, m, b) in J]
    return round(st.fmean(d), 4) if d else None


S = lambda j: strict(j)
D = {"models": MODELS, "judge_model": ["gpt-6-luna"], "judge_runs": [3], "n_judged": len(J),
     "summary": {"full": summ(S), "obliged": summ(lambda j: strict(j, OBLIGED)),
                 "rigour": summ(lambda j: strict(j, RIGOUR))}}

short = lambda v: v.split(" ")[0] + (" Findings" if "Findings" in v else " workshop" if "Workshop" in v else "") if "none" not in v else "—"
D["claims"] = [{"id": p, "title": c["provenance"]["title"], "venue": short(c["provenance"]["venue"]),
                "field": c["provenance"]["field"], "n_ref": len(roles[p]), "subfields": sub[p]["subfields"],
                "v": {a: per_claim(S, p, a) for a in ("nocard", "rag", "card", "tmpl")},
                "d": claim_delta(S, p)} for p, c in sorted(claims.items())]
D["mean_ref"] = round(st.fmean(len(r) for r in roles.values()), 2)

cards = collections.defaultdict(list)
for c in D["claims"]:
    for s in c["subfields"]:
        if c["d"] is not None:
            cards[s].append({"id": c["id"], "d": c["d"]})
D["per_card"] = sorted(({"name": s, "n": len(v), "mean": round(st.fmean(x["d"] for x in v), 4),
                         "lo": min(x["d"] for x in v), "hi": max(x["d"] for x in v), "claims": v, "new": True}
                        for s, v in cards.items()), key=lambda r: -r["mean"])

D["metrics"] = {
    "strict recall": {"sub": "a reference experiment counts only when the judge marks it fully covered",
                      "means": D["summary"]["full"]["means"], "delta": D["summary"]["full"]["deltas"]["card_vs_nocard"]},
    "recall": {"sub": "a partly addressed experiment counts as half",
               "means": summ(recall)["means"], "delta": boot(pairs(recall, "nocard", "card"))}}

FIELDS = ["robotics", "interpretability", "vision", "nlp", "math_mltheory"]
D["by_field"] = {f: {"n": sum(c["field"] == f for c in D["claims"]),
                     **summ(S, pids=[c["id"] for c in D["claims"] if c["field"] == f])} for f in FIELDS}

rc = collections.defaultdict(lambda: collections.defaultdict(list))
for (p, m, a), j in J.items():
    for i, c in enumerate(j["reference_coverage"]):
        rc[roles[p][i]][a].append(c["status"] == "covered")
D["by_role"] = {r: {"n": len(v.get("card", [])), **{a: round(st.fmean(x), 4) for a, x in v.items()}} for r, v in rc.items()}
D["by_role_delta"] = {r: boot(pairs(lambda j, r=r: strict(j, {r}), "nocard", "card")) for r in rc}

shape = {}
props = {}
for f in glob.glob(f"{R}/proposals/problem_*/scify_proposer.json"):
    for arm, v in json.load(open(f))["arms"].items():
        props.setdefault(arm.split("+")[0], []).append(v.get("experiments") or [])
for m in MODELS:
    ex = [e for e in props.get(m, []) if e]
    shape[m] = {"mean_exp": round(st.fmean(len(e) for e in ex), 2),
                "full_budget": round(st.fmean(len(e) >= 9 for e in ex), 3),
                "chars": round(st.fmean(len(x) for e in ex for x in e))}
D["shape"] = shape
D["up_down"] = {"up": sum(c["d"] is not None and c["d"] > .0005 for c in D["claims"]),
                "down": sum(c["d"] is not None and c["d"] < -.0005 for c in D["claims"])}
D["refused"] = sorted(f"{p} {m}+{a}" for p in claims for m in MODELS for a in ("nocard", "rag", "card")
                      if os.path.exists(f"{R}/eval/judgments/problem_{p}/{m}+{a}.json") and (p, m, a) not in J)
json.dump(D, open(sys.argv[1], "w"), indent=1)
f = D["summary"]["full"]
print(f"{len(J)} judged | means {f['means']}")
for k in ("card_vs_nocard", "card_vs_rag", "rag_vs_nocard", "card_vs_tmpl", "tmpl_vs_nocard"):
    print(f"  {k:16s} {f['deltas'][k]}")
