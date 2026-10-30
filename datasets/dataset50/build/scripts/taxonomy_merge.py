"""Step 2 of the 2026-10-29 taxonomy audit (after taxonomy_audit.py): fold entries that
consolidate.py merges into one card, so subfield_gather searches under the card's name."""
import json

P = "norm_cards/data/taxonomy.json"
tasks = json.load(open(P))["tasks"]
by = {t["task"]: t for t in tasks}
FOLD = {"Optimization and Training Dynamics": ["Deep Learning Optimization", "Training Dynamics",
                                               "Learning Theory and Generalization"],
        "Mechanistic Interpretability": ["Sparse Autoencoders"]}
gone = set()
for target, parts in FOLD.items():
    if target not in by:
        by[target] = {"task": target, "area": by[parts[0]]["area"], "keywords": [], "datasets": [],
                      "metrics": [], "card_type": "empirical", "source": by[parts[0]]["source"]}
        tasks.append(by[target])
    kws = by[target]["keywords"]
    # interleave so the first three keywords (the gather queries) come from each part
    for i in range(max(len(by[p]["keywords"]) for p in parts)):
        kws += [by[p]["keywords"][i] for p in parts if i < len(by[p]["keywords"])]
    by[target]["keywords"] = list(dict.fromkeys(kws))
    gone |= set(parts)
out = [t for t in tasks if t["task"] not in gone]
json.dump({"tasks": out}, open(P, "w"), indent=2)
print(f"{len(tasks)} -> {len(out)} entries;", {t: by[t]["keywords"][:6] for t in FOLD})
