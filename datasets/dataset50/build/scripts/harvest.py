"""arXiv harvest for the dataset50 expansion: per field x venue, papers whose arXiv
comment names a 2026 venue (or 2025 at worst) and whose v1 is on/after 2026-02-01.
Saves after every query; arXiv asks for >= 3 s between calls."""
import json, os, sys, time, re, requests, xml.etree.ElementTree as ET
OUT = os.path.join(os.path.dirname(__file__), "candidates.json")
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
VENUES = ["NeurIPS 2026", "ICML 2026", "ICLR 2026", "CVPR 2026", "ECCV 2026", "ACL 2026",
          "EMNLP 2026", "NAACL 2026", "CoRL 2026", "RSS 2026", "ICRA 2026", "AAAI 2026"]
FIELDS = {
 "robotics": 'abs:(robot OR robotic OR "vision-language-action" OR manipulation OR humanoid OR embodied OR locomotion)',
 "interpretability": 'abs:("mechanistic interpretability" OR "sparse autoencoder" OR "knowledge editing" OR "model editing" OR "activation steering" OR "steering vector" OR circuit OR "attention heads" OR "linear probe")',
 "vision": 'abs:(backdoor OR "adversarial robustness" OR "adversarial attack" OR poisoning OR deepfake OR forgery OR "AI-generated image" OR watermark OR "out-of-distribution detection")',
 "nlp": 'abs:("constrained decoding" OR "structured output" OR "instruction following" OR jailbreak OR "red teaming" OR "safety alignment" OR "reward model" OR hallucination OR "long context")',
 "nlp_cd": 'abs:("constrained decoding" OR "constrained generation" OR "grammar-constrained" OR "structured generation" OR "JSON schema" OR "logical constraints" OR "lexically constrained" OR "controllable generation")',
 "sci_math": 'abs:("theorem proving" OR autoformalization OR Lean OR molecular OR molecule OR protein OR "drug discovery" OR "materials discovery" OR "PDE")',
 # 2026-10-29: fifth field changed to math + ML theory/optimization; science is held out for a later test set
 "ml_theory": '(cat:cs.LG OR cat:stat.ML) AND abs:(optimizer OR "learning rate" OR "training dynamics" OR "implicit bias" OR "loss landscape" OR "generalization bound" OR "sample complexity" OR "scaling law" OR grokking OR "gradient descent" OR "weight decay" OR "batch size")',
}
CUTOFF = "2026-02-01"
def load(): return json.load(open(OUT)) if os.path.exists(OUT) else {}
def query(q, n=200):
    for a in range(6):
        try:
            r = requests.get("https://export.arxiv.org/api/query", timeout=90, params={
                "search_query": q, "sortBy": "submittedDate", "sortOrder": "descending",
                "max_results": n})
            if r.status_code == 200: return r.text
            print(f"   http {r.status_code}", flush=True)
        except Exception as e:
            print(f"   {type(e).__name__}", flush=True)
        time.sleep(10 + 20 * a)
    return None
rows = load(); done = {(r["field"], v) for r in rows.values() for v in r.get("_q", [])}
for f, fq in FIELDS.items():
    for v in VENUES:
        if (f, v) in done: continue
        xml = query(f'co:"{v}" AND {fq}')
        time.sleep(3.5)
        if xml is None: print(f"[{f}/{v}] gave up", flush=True); continue
        root = ET.fromstring(xml); n_new = n_hit = 0
        for e in root.findall("a:entry", NS):
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[1])
            pub = e.find("a:published", NS).text[:10]
            if pub < CUTOFF: continue
            n_hit += 1
            r = rows.get(aid)
            if r is None:
                n_new += 1
                c = e.find("x:comment", NS)
                r = rows[aid] = {"id": aid, "v1": pub, "field": f, "venue": v,
                    "title": " ".join(e.find("a:title", NS).text.split()),
                    "abstract": " ".join(e.find("a:summary", NS).text.split()),
                    "comment": c.text.strip() if c is not None else "",
                    "cats": [t.get("term") for t in e.findall("a:category", NS)], "_q": []}
            r["_q"].append(v)
        json.dump(rows, open(OUT, "w"), indent=1)
        print(f"[{f:16s}/{v:12s}] {n_hit:3d} post-cutoff, {n_new:3d} new  (total {len(rows)})", flush=True)
print("DONE", len(rows))
