"""Decisions API judges against the generative gpt-6-luna k=3 judge, on the field-norms arms
judged by all three. Item agreement (with luna's own pass-vs-pass agreement as the ceiling),
the label mix, and card - no card per proposer under each judge (paired bootstrap, seed 0)."""
import collections
import glob
import json
import random
import statistics as st

R = "results/dataset50run_fn"
ARMS = ("nocard", "rag", "card")
L = {}
for f in glob.glob(f"{R}/eval/judgments/problem_*/*.json"):
    if ".trace" in f:
        continue
    j = json.load(open(f))
    if len(j.get("runs") or []) >= 3 and j["scores"].get("n_proposed"):
        L[(j["problem_id"], j["arm"])] = j
DEC = {}
for v in ("arm", "item"):
    DEC[v] = {}
    for f in glob.glob(f"{R}/decisions/{v}/judgments/problem_*/*.json"):
        j = json.load(open(f))
        DEC[v][(j["problem_id"], j["arm"])] = [c["status"] for c in j["reference_coverage"]]
keys = sorted(k for k in L if all(k in DEC[v] for v in DEC))
print(f"{len(keys)} arms judged by luna k=3 and both Decisions variants\n")

luna = {k: [c["status"] for c in L[k]["reference_coverage"]] for k in keys}
passes = {k: [[c["status"] for c in r["reference_coverage"]] for r in L[k]["runs"]] for k in keys}


def agree(a, b):
    x = [s == t for k in keys for s, t in zip(a[k], b[k])]
    return st.fmean(x), len(x)


pp = [p[i][n] == p[j][n] for k in keys for p in [passes[k]] for i in range(3) for j in range(i + 1, 3)
      for n in range(min(len(p[i]), len(p[j])))]
print(f"luna pass vs luna pass (ceiling)  {st.fmean(pp):.3f}")
for v in DEC:
    a, n = agree(luna, DEC[v])
    sp = [luna[k][i] == DEC[v][k][i] for k in keys for i in range(len(luna[k])) if luna[k][i] != "CONTESTED"]
    print(f"luna majority vs decisions[{v:4s}]  {a:.3f}  (on items luna's passes don't split: {st.fmean(sp):.3f}, n={n})")
a, _ = agree(DEC["arm"], DEC["item"])
print(f"decisions[arm] vs decisions[item]  {a:.3f}")

print("\nlabel mix")
for name, J in (("luna k=3", luna), ("decisions[arm]", DEC["arm"]), ("decisions[item]", DEC["item"])):
    c = collections.Counter(s for k in keys for s in J[k]); t = sum(c.values())
    print(f"  {name:16s} " + "  ".join(f"{s} {c[s] / t:.2f}" for s in ("covered", "partial", "missing", "CONTESTED") if c[s]))
print("\nconfusion, luna majority (rows) x decisions[arm] / decisions[item] (cols: covered partial missing)")
for s in ("covered", "partial", "missing", "CONTESTED"):
    row = []
    for v in DEC:
        c = collections.Counter(DEC[v][k][i] for k in keys for i in range(len(luna[k])) if luna[k][i] == s)
        row.append(" ".join(f"{c[t]:4d}" for t in ("covered", "partial", "missing")))
    print(f"  {s:10s} {row[0]}   |  {row[1]}")


def strict(sts):
    return st.fmean(s == "covered" for s in sts)


def boot(d, n=10000):
    r = random.Random(0)
    bs = sorted(st.fmean(r.choices(d, k=len(d))) for _ in range(n))
    return f"{st.fmean(d):+.3f} [{bs[int(n * .025)]:+.3f}, {bs[int(n * .975)]:+.3f}]"


models = sorted({k[1].split("+")[0] for k in keys})
pids = sorted({k[0] for k in keys})
print("\ncard - no card, strict recall, per proposer")
print(f"  {'':15s} {'luna k=3':26s} {'decisions[arm]':26s} {'decisions[item]':26s}")
for m in models + ["all"]:
    ms = models if m == "all" else [m]
    cells = []
    for J in (luna, DEC["arm"], DEC["item"]):
        d = [strict(J[(p, f"{x}+card")]) - strict(J[(p, f"{x}+nocard")]) for p in pids for x in ms
             if (p, f"{x}+card") in J and (p, f"{x}+nocard") in J]
        cells.append(f"{boot(d):26s}")
    print(f"  {m:15s} " + "".join(cells) + f" n={len(d)}")
print("\nmean strict recall (nocard / rag / card)")
for m in models:
    row = []
    for J in (luna, DEC["arm"], DEC["item"]):
        row.append(" / ".join(f"{st.fmean(strict(J[(p, f'{m}+{a}')]) for p in pids if (p, f'{m}+{a}') in J):.3f}" for a in ARMS))
    print(f"  {m:15s} " + "   |   ".join(row))
