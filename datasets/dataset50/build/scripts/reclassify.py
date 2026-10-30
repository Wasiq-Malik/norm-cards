"""Stage-0 subfield prompt on every dataset50 draft claim, K samples each (no taxonomy writes)."""
import json, sys, glob, collections
from concurrent.futures import ThreadPoolExecutor
from norm_cards.taxonomy import Taxonomy
from norm_cards import classifier, llm
MODEL, K, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tax = Taxonomy.load()
vocab = classifier._vocab_by_area(tax)
claims = []
for p in sorted(glob.glob("datasets/dataset50/build/drafts/*.json") + glob.glob("datasets/dataset50/build/drafts_orig20/*.json")):
    d = json.load(open(p)); base = "arxiv" + d["paper"]["id"].replace(".", "_")
    for c in d["claims"]:
        claims.append({"problem_id": base if c["cid"] == "c0" else f"{base}_{c['cid']}", "claim": c["claim"]})
def canon(s):
    h = tax.get(s)
    return h["task"] if h else (tax.find_near_duplicate(s) or s)
def one(args):
    c, r = args
    m = ", ".join(x["task"] for x in tax.heuristic_match(c["claim"], top_k=8)) or "(none)"
    for attempt in range(3):
        try:
            d = llm.complete_json(classifier._PROMPT.format(claim=c["claim"], vocab=vocab, matches=m), model=MODEL)
            return c["problem_id"], r, [canon(s) for s in d.get("subfields") or []], d.get("new_subfields") or []
        except Exception as e:
            err = repr(e)
    return c["problem_id"], r, None, err
jobs = [(c, r) for c in claims for r in range(K)]
res = collections.defaultdict(list); errs = []
with ThreadPoolExecutor(12) as ex:
    for pid, r, subs, new in ex.map(one, jobs):
        if subs is None: errs.append((pid, new)); continue
        res[pid].append({"subfields": subs, "new": new})
json.dump(dict(sorted(res.items())), open(OUT, "w"), indent=1)
print("claims", len(claims), "done", len(res), "errors", len(errs), errs[:3])
