"""Turn the 50 dataset50 drafts into the files the eval harness reads.

    python datasets/dataset50/build/scripts/make_dataset.py

Writes
  norm_cards/data/dataset50-claims-v1.jsonl            one problem per claim (NORM_CARDS_CLAIMS)
  datasets/dataset50/references/problem_<id>/ground_truth.json   its reference experiments + roles
  datasets/dataset50/dataset.json                      what the dataset is and how it was made

Claim ids follow dataset34: arxiv<id> for a paper's c0, arxiv<id>_c<n> for the rest, which is
also how subfields.json is keyed. An experiment shared by two claims goes into both references.
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
DRAFTS = sorted(glob.glob(os.path.join(ROOT, "datasets/dataset50/build/drafts/*.json"))
                + glob.glob(os.path.join(ROOT, "datasets/dataset50/build/drafts_orig20/*.json")))
CLAIMS = os.path.join(ROOT, "norm_cards/data/dataset50-claims-v1.jsonl")
DS = os.path.join(ROOT, "datasets/dataset50")
DRAFTER = "claude-opus-5.5"


def pid_of(paper_id: str, cid: str) -> str:
    base = "arxiv" + paper_id.replace(".", "_")
    return base if cid == "c0" else f"{base}_{cid}"


def main():
    sub = json.load(open(os.path.join(DS, "subfields.json")))
    problems, refs = [], {}
    for path in DRAFTS:
        d = json.load(open(path))
        p = d["paper"]
        for c in d["claims"]:
            pid = pid_of(p["id"], c["cid"])
            exps = [e for e in d["experiments"] if c["cid"] in e["claims"]]
            problems.append({
                "type": "problem", "format_version": "1.0", "problem_version": "1.0",
                "domain": "ai", "problem_id": pid, "claim": c["claim"], "artifacts": [],
                "needs_review": True,
                "provenance": {"arxiv_id": p["id"], "arxiv_v1_date": p["v1"], "venue": p["venue"],
                               "field": p["field"], "title": p["title"], "drafted_from": d["read"],
                               "drafter_model": DRAFTER, "draft": os.path.relpath(path, ROOT)},
                "_decision_rule": c["decision_rule"],
                "_subfields_hint": c["subfields_hint"],
            })
            refs[pid] = {
                "type": "ground_truth", "format_version": "3.0", "problem_id": pid,
                "claim": c["claim"],
                "provenance": {"kind": "transcribed from the source paper",
                               "sources": [f"arXiv:{p['id']}"],
                               "notes": "Drafted for dataset50 from the arXiv HTML; bars checked "
                                        "against the paper's own numbers. Not human-reviewed.",
                               "author": f"{DRAFTER} — automated drafting; PENDING HUMAN SIGN-OFF",
                               "drafter_model": DRAFTER, "drafted_from": d["read"],
                               "draft": os.path.relpath(path, ROOT)},
                "experiments": [e["text"] for e in exps],
                "roles": [e["role"] for e in exps],
            }

    ids = [x["problem_id"] for x in problems]
    assert len(ids) == len(set(ids)), "duplicate problem ids"
    missing = sorted(set(ids) ^ set(sub))
    assert not missing, f"claims and subfields.json disagree: {missing}"
    thin = [pid for pid, r in refs.items() if len(r["experiments"]) < 2]
    assert not thin, f"claims with fewer than 2 reference experiments: {thin}"

    with open(CLAIMS, "w", encoding="utf-8") as f:
        for x in sorted(problems, key=lambda x: x["problem_id"]):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for pid, r in refs.items():
        out = os.path.join(DS, "references", f"problem_{pid}")
        os.makedirs(out, exist_ok=True)
        json.dump(r, open(os.path.join(out, "ground_truth.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)

    fields = {}
    for x in problems:
        fields.setdefault(x["provenance"]["field"], set()).add(x["provenance"]["arxiv_id"])
    meta = {
        "name": "dataset50", "claims": "norm_cards/data/dataset50-claims-v1.jsonl", "built": "2026-10-29",
        "description": "50 source papers, 10 per field (robotics, interpretability, vision, NLP, "
                       "math + ML theory), 100 claims; "
                       "format-2 shared subfield cards.",
        "claims_model": DRAFTER, "references_model": DRAFTER, "references_reviewed": False,
        "card_format": "2", "card_models": {"map": "gpt-6-luna", "reduce": "gpt-6-luna"},
        "card_text": "arXiv HTML main body first, PDF main body fallback",
        "subfield_classifier": "gpt-6-luna, classifier._PROMPT on each claim, 3 samples, labels kept "
                               "when >= 2 agree; then datasets/dataset50/build/scripts/consolidate.py (merges, "
                               "catch-all drops, 7 hand removals, cards need >= 2 papers)",
        "gather": "subfield_gather: gpt-6-luna queries from subfield name + keywords, gpt-6-luna "
                  "relevance screen, full-text-only candidates, source papers removed (leakage.py)",
        "papers_per_field": {k: len(v) for k, v in sorted(fields.items())},
        "notes": ["Selection: main-track 2026 venue in the arXiv comment, arXiv v1 >= 2026-02-01; "
                  "exceptions kept from the original 20 are listed in datasets/dataset50/build/drafts/SKIPPED.md "
                  "and the dataset50 memory (2510.20487 pre-cutoff, 2608.09928 workshop, Findings papers).",
                  "Science papers are held out for a later test set (datasets/dataset50/build/science_heldout.json).",
                  "References are NOT human-reviewed."],
    }
    json.dump(meta, open(os.path.join(DS, "dataset.json"), "w"), indent=1)
    n_exp = sum(len(r["experiments"]) for r in refs.values())
    print(f"{len(problems)} claims, {n_exp} reference experiments "
          f"({n_exp / len(problems):.1f}/claim), papers per field {meta['papers_per_field']}")


if __name__ == "__main__":
    main()
