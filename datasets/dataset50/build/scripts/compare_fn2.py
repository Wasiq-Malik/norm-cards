"""fieldnorms vs fieldnorms2 proposer prompt on subset30 (or all claims with --all), k=3 judgments, per proposer: strict recall
per experiment type (gpt-6-luna labels, types.json) for each arm, card - no card under each prompt
(paired bootstrap over claims, 10,000 resamples, seed 0), and how often proposed experiments are
anchored on 'evidence' (the misuse fieldnorms2 targets: there is no evidence in any arm).

    python datasets/dataset50/build/scripts/compare_fn2.py [--all] [--b5]

--b5 compares budget 9 with budget 5 under fieldnorms2 instead of the two prompts.
"""
import collections
import glob
import json
import random
import re
import sys

from norm_cards.eval.evaluate import load_arm

RUNS = ({"b9": "results/dataset50run_fn2", "b5": "results/dataset50run_fn2_b5"} if "--b5" in sys.argv
        else {"fn": "results/dataset50run_fn", "fn2": "results/dataset50run_fn2"})
SOURCES = {"qwen3.8-27b": "proposals_qwen", "gpt-5.6-terra": "proposals_g56", "gpt-5.6-luna": "proposals_g56"}
SUB = (sorted(json.load(open("datasets/dataset50/build/types.json"))) if "--all" in sys.argv
       else open("datasets/dataset50/build/subset30.txt").read().strip().split(","))
T = json.load(open("datasets/dataset50/build/types.json"))
GROUPS = {"setup": {"setup"}, "headline": {"headline"}, "threat": {"threat"}, "analysis": {"analysis"},
          "other": {"other"},
          "scored": {"headline", "threat", "analysis"}}
SUBCLAIM = re.compile(r"^\W*(to address|addressing|for) subclaims?", re.I)
EVIDENCE = re.compile(r"evidence piece|with respect to (the )?(field norms?|norm card|evidence|relevant literature)", re.I)

cov = {}                                   # (prompt, model, arm) -> pid -> [(type, covered)]
for name, R in RUNS.items():
    for f in glob.glob(f"{R}/eval/judgments/problem_*/*.json"):
        if ".trace" in f:
            continue
        j = json.load(open(f))
        p = j["problem_id"]
        if p not in SUB or len(j.get("runs") or []) < 3 or not j["scores"].get("n_proposed"):
            continue
        m, a = j["arm"].split("+")
        cov.setdefault((name, m, a), {})[p] = [(T[p][i]["type"], c["status"] == "covered")
                                               for i, c in enumerate(j["reference_coverage"])]


def rows(name, m, g):
    c, n = cov.get((name, m, "card"), {}), cov.get((name, m, "nocard"), {})
    out = []
    for p in sorted(set(c) & set(n)):
        a = [x for t, x in c[p] if t in GROUPS[g]]
        b = [x for t, x in n[p] if t in GROUPS[g]]
        if a:
            out.append((sum(a), sum(b), len(a)))
    return out


def diff(rs, n=10000):
    est = lambda r: (sum(x[0] for x in r) - sum(x[1] for x in r)) / sum(x[2] for x in r)
    rnd = random.Random(0)
    bs = sorted(est(rnd.choices(rs, k=len(rs))) for _ in range(n))
    lo, hi = bs[int(n * .025)], bs[int(n * .975)]
    return f"{est(rs):+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if lo > 0 or hi < 0 else " ")


def level(name, m, a, g):
    x = [c for items in cov.get((name, m, a), {}).values() for t, c in items if t in GROUPS[g]]
    return sum(x) / len(x) if x else float("nan")


models = sorted({m for _, m, _ in cov})
for g in GROUPS:
    n = sum(T[p][i]["type"] in GROUPS[g] for p in SUB for i in range(len(T[p])))
    print(f"\n{g}   {n} items   (nocard / rag / card; card - nocard)")
    for m in models:
        cells = []
        for name in RUNS:
            rs = rows(name, m, g)
            lv = " / ".join(f"{level(name, m, a, g):.2f}" for a in ("nocard", "rag", "card"))
            cells.append(f"{lv}  {diff(rs) if rs else '-':26s}")
        print(f"  {m:14s} " + "  ".join(f"{k}: {c}" for k, c in zip(RUNS, cells)))

print("\nshare of proposed experiments anchored on a subclaim / on 'evidence', and experiments per proposal")
for m in models:
    cells = []
    for name, R in RUNS.items():
        src = f"{R}/{SOURCES.get(m, 'proposals')}"
        for a in ("nocard", "rag", "card"):
            ex = []
            for p in SUB:
                try:
                    ex.append(load_arm(src, p, f"{m}+{a}")["experiments"])
                except Exception:
                    pass
            flat = [e for x in ex for e in x]
            sub = sum(bool(SUBCLAIM.search(e)) for e in flat) / len(flat) if flat else float("nan")
            evd = sum(bool(EVIDENCE.search(e[:300])) for e in flat) / len(flat) if flat else float("nan")
            cells.append(f"{a} {sub:.2f}/{evd:.2f} ({len(flat) / max(len(ex), 1):.1f})")
    a, b = RUNS
    print(f"  {m:14s} {a}: {'  '.join(cells[:3])}   {b}: {'  '.join(cells[3:])}")
