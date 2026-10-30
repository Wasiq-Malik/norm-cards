"""Consolidate the gpt-6-luna v2 labels (subfields_luna_v2.json) into shared cards.

Rules, applied in order; every claim's before/after goes to consolidation.json.
  1. Merge: labels that ride on the same claims become one card.
  2. Hand removals: labels that point at a different literature than the claim's (each
     with its reason).
  3. Drop catch-all and incidental labels: a broad parent or a benchmark's format/domain
     when the claim already has a more specific label.
  4. A card needs claims from >= 2 papers; labels below that are dropped.
  5. A claim left with nothing inherits its sibling claims' labels.
"""
import json, collections, glob

D = "datasets/dataset50/build/subfield_labels"
raw = json.load(open(f"{D}/subfields_luna_v2.json"))

MERGE = {  # Deep Learning Optimization (10 claims) sits entirely inside Training Dynamics (13),
           # and 3 of Learning Theory's 4 claims also carry it
    "Deep Learning Optimization": "Optimization and Training Dynamics",
    "Training Dynamics": "Optimization and Training Dynamics",
    "Learning Theory and Generalization": "Optimization and Training Dynamics",
    # 2/2 claims inside Mechanistic Interpretability; SAE norms fold into that card
    "Sparse Autoencoders": "Mechanistic Interpretability",
}
CATCHALL = {  # label -> dropped when the claim also has any of these
    "Language Modeling": None,                       # any other label
    "Question Answering": None,                      # benchmark format, not the subfield
    "Code and Structured Generation": {"Constrained Decoding"},
    "Text-to-SQL": {"Constrained Decoding"},         # evaluation domain of the decoding papers
    "Image Classification": {"Adversarial Robustness", "Backdoor Attacks and Defenses"},
    "Object Detection": {"Adversarial Robustness"},
    "3D object detection": {"Adversarial Robustness"},
    "Math": {"Theorem Proving and Autoformalization"},
    "Robotics": {"Vision-Language-Action Models", "Robot Manipulation", "Embodied Navigation and Planning"},
    "Agents": {"Mechanistic Interpretability"},      # agents are the object of study, not the method
    "OCR": None, "Image Understanding": None, "Text classification": None, "Coding Agents": None,
}
HAND = {  # (claim, label): reason
    ("arxiv2605_28030", "Adversarial Robustness"): "harmful fine-tuning of LLMs; the card's papers are vision robustness",
    ("arxiv2605_28030_c1", "Adversarial Robustness"): "same",
    ("arxiv2608_23830_c1", "Optimization and Training Dynamics"): "RL for instruction following, not optimization theory",
    ("arxiv2603_18532_c1", "Optimization and Training Dynamics"): "sim-to-real generalization, not learning theory",
    ("arxiv2604_17132", "Constrained Decoding"): "contrastive decoding for over-refusal; no output constraint",
    ("arxiv2609_06718", "Robot Manipulation"): "humanoid whole-body control, not manipulation",
    ("arxiv2605_08612_c1", "Mechanistic Interpretability"): "isolating backdoor parameters, not interpreting them",
}

def step(labels, pid):
    out = [s for s in dict.fromkeys(MERGE.get(s, s) for s in labels) if (pid, s) not in HAND]
    return [s for s in out if not (s in CATCHALL and (
        (CATCHALL[s] is None and len(out) > 1) or (CATCHALL[s] and set(out) & CATCHALL[s])))]

lab = {p: step(v, p) for p, v in raw.items()}
for _ in range(2):  # rule 4 can empty a claim, rule 5 refills it; repeat once for stability
    papers = collections.defaultdict(set)
    for p, v in lab.items():
        for s in v: papers[s].add(p.split("_c")[0])
    lab = {p: [s for s in v if len(papers[s]) >= 2] for p, v in lab.items()}
    for p, v in lab.items():
        if not v:
            sib = [s for q, w in lab.items() if q != p and q.split("_c")[0] == p.split("_c")[0] for s in w]
            lab[p] = list(dict.fromkeys(sib))

field = {}
for f in glob.glob("datasets/dataset50/build/drafts*/*.json"):
    d = json.load(open(f)); b = "arxiv" + d["paper"]["id"].replace(".", "_")
    for c in d["claims"]: field[b if c["cid"] == "c0" else f"{b}_{c['cid']}"] = d["paper"]["field"]
json.dump({"rules": __doc__, "merge": MERGE, "catchall": {k: sorted(v) if v else "any" for k, v in CATCHALL.items()},
           "hand": {f"{p} | {s}": r for (p, s), r in HAND.items()},
           "claims": {p: {"field": field[p], "subfields": lab[p], "luna_v2": raw[p]} for p in sorted(lab)}},
          open(f"{D}/consolidation.json", "w"), indent=1)

cl, pa = collections.Counter(), collections.defaultdict(set)
for p, v in lab.items():
    for s in v: cl[s] += 1; pa[s].add(p.split("_c")[0])
fields = collections.defaultdict(collections.Counter)
for p, v in lab.items():
    for s in v: fields[s][field[p]] += 1
print(f"cards {len(cl)} | labels/claim {sum(map(len, lab.values()))/len(lab):.2f} | empty {[p for p,v in lab.items() if not v]}")
for s, n in cl.most_common():
    print(f"  {s:40s} {n:2d} claims {len(pa[s]):2d} papers  " + ", ".join(f"{f} {k}" for f, k in fields[s].most_common()))
