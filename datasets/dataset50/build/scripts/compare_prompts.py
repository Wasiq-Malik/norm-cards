"""Old (SciFy) vs new (fieldnorms) prompt, single vote, on the claims judged in both.
Old single vote = pass 0 of each k=3 judgment (run_statuses[0])."""
import json, glob, statistics as st, random, sys
roles = {p.split("problem_")[1].split("/")[0]: json.load(open(p))["roles"]
         for p in glob.glob("results/dataset50run/eval/ground_truth/problem_*/ground_truth.json")}
MODELS = sys.argv[1].split(",")
RIG = {"control", "confound"}; CORE = {"headline", "apparatus", "mechanism"}

def load(root, first_vote):
    J = {}
    for p in glob.glob(f"{root}/eval/judgments/problem_*/*.json"):
        if ".trace" in p: continue
        j = json.load(open(p)); c = j["arm"].split("+", 1)[1]
        if c not in ("nocard", "rag", "card") or not j["scores"].get("n_proposed") or j["proposer_model"] not in MODELS: continue
        st_ = [(c_.get("run_statuses") or [c_["status"]])[0] if first_vote else c_["status"] for c_ in j["reference_coverage"]]
        J[(j["problem_id"], j["proposer_model"], c)] = st_
    return J
OLD, NEW = load("results/dataset50run", True), load("results/dataset50run_fn", False)
def s(p, v, keep=None):
    x = [st_ == "covered" for i, st_ in enumerate(v) if keep is None or roles[p][i] in keep]
    return st.fmean(x) if x else None
both = sorted({p for p, m, c in NEW} & {p for p, m, c in OLD})
both = [p for p in both if all((p, m, c) in NEW and (p, m, c) in OLD for m in MODELS for c in ("nocard", "card", "rag"))]
def boot(J, m, a, b, keep=None):
    d = [s(p, J[(p, m, b)], keep) - s(p, J[(p, m, a)], keep) for p in both
         if s(p, J[(p, m, a)], keep) is not None and s(p, J[(p, m, b)], keep) is not None]
    if not d: return "—"
    r = random.Random(0); bs = sorted(st.fmean(r.choices(d, k=len(d))) for _ in range(5000))
    return f"{st.fmean(d):+.3f} [{bs[125]:+.3f}, {bs[4875]:+.3f}]"
print(f"{len(both)} claims judged in both runs (single vote)")
for m in MODELS:
    for name, J in (("old SciFy prompt ", OLD), ("new Field Norms  ", NEW)):
        mu = {c: round(st.fmean(s(p, J[(p, m, c)]) for p in both), 3) for c in ("nocard", "rag", "card")}
        print(f"  {m:13s} {name} {mu} | card-nocard {boot(J, m, 'nocard', 'card')} | rag-nocard {boot(J, m, 'nocard', 'rag')}"
              f" | ctl+conf {boot(J, m, 'nocard', 'card', RIG)} | core {boot(J, m, 'nocard', 'card', CORE)}")
