"""Drop the dataset's own source papers from the shared subfield bundles.

build_cards' leakage check is keyed by claim id, and shared bundles are keyed
"subfield_<name>", so it never fired on a shared card. This checks every bundle against
every source paper, by arXiv id and by normalized title, and rewrites the bundles file
without them (the original is kept as subfield_bundles.raw.jsonl).

    python datasets/dataset50/build/scripts/leakage.py <bundles.jsonl> [--write]
"""
import json, glob, re, sys, shutil
from norm_cards import fulltext

def norm(t): return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

src = {}
for p in glob.glob("datasets/dataset50/build/drafts*/*.json"):
    d = json.load(open(p))["paper"]; src[d["id"]] = norm(d["title"])
path = sys.argv[1]
L = [json.loads(l) for l in open(path) if l.strip()]
hits = 0
for b in L:
    keep = []
    for p in b["papers"]:
        ax, t = fulltext.arxiv_id_of(p), norm(p.get("title"))
        hit = next((i for i, st in src.items() if ax == i or (len(t) > 25 and (t[:60] in st or st[:60] in t))), None)
        if hit:
            hits += 1; print(f"  {b['subfield']}: drops {hit} {p.get('title','')[:70]}")
        else:
            keep.append(p)
    b["papers"] = keep
print(f"{hits} source-paper hits across {len(L)} bundles")
if "--write" in sys.argv and hits:
    shutil.copy(path, path.replace(".jsonl", ".raw.jsonl"))
    with open(path, "w") as f:
        f.writelines(json.dumps(b) + "\n" for b in L)
    print("rewrote", path)
