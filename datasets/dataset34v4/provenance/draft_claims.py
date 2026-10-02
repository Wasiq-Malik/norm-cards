"""Draft the extra per-paper claims from split.json, in the house claim style.

    ./.venv/bin/python datasets/dataset20v4_draft/draft_claims.py

Uses make_claims' rules verbatim (bar yes, finding no, subject yes, apparatus and method no),
anchored to the experiments the hand split assigned to each claim, and shown the paper's
existing claim so the new one does not restate it. Same drafter model as v3 (gpt-5.6-sol).
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())
from norm_cards import llm                                        # noqa: E402
from norm_cards.eval.make_claims import DRAFT_PROMPT, SELF_REF    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gpt-5.6-sol"
RULES = DRAFT_PROMPT.split("Also return `subfields`")[0].replace(
    "Below is a paper. Write the CLAIM its experiments were designed to decide.",
    "Below is a paper. It makes several claims. One is already written down (EXISTING CLAIM). "
    "Write ANOTHER claim from the same paper: the one that the TARGET EXPERIMENTS below were "
    "designed to decide.")

ANCHOR = """
EXISTING CLAIM (already in the dataset - do not restate it or fold it in):
{existing}

THE PROPOSITION TO WRITE UP, in rough form (sharpen it; keep its scope):
{rough}

TARGET EXPERIMENTS - the experiments this claim must be decidable by. They contain the authors'
RESULTS; those are findings and must NOT appear in the claim. Use them only to see what question
was asked and what comparison or threshold decided it:
{experiments}

PAPER TITLE: {title}

PAPER TEXT:
{text}

Return JSON: {{"claim": "...",
               "decision_rule": "the sentence in your claim that states the bar, quoted verbatim",
               "leaked_method_check": "quote any phrase naming the authors' own technique or its
                                       mechanism, or 'none'",
               "named_apparatus": ["every dataset, benchmark or baseline named; usually empty"]}}"""


def draft(pid, c, claims, refs):
    base = claims[pid]
    text = open(os.path.join(".pdfcache", hashlib.md5(pid.encode()).hexdigest()[:12] + ".p18.txt")).read()
    exps = refs[pid]["experiments"]
    idx = sorted(set(c["experiments"]))
    prompt = RULES + ANCHOR.format(
        existing=base["claim"], rough=c["draft"],
        experiments="\n\n".join(f"[{i}] {exps[i]}" for i in idx),
        title=base["provenance"].get("title", ""), text=text[:150000])
    out = llm.complete_json(prompt, model=MODEL)
    claim = (out.get("claim") or "").strip()
    rec = {"type": "problem", "format_version": "1.0", "problem_version": "1.0", "domain": "ai",
           "problem_id": f"{pid}_{c['id'].lower()}", "claim": claim, "artifacts": [],
           "needs_review": True,
           "provenance": {**base["provenance"], "drafted_from": "full text + hand split",
                          "drafter_model": MODEL, "parent_claim": pid, "split_id": c["id"]},
           "_decision_rule": out.get("decision_rule", ""),
           "_leaked_method_check": out.get("leaked_method_check", ""),
           "_named_apparatus": out.get("named_apparatus") or [],
           "_self_reference": bool(SELF_REF.search(claim))}
    return rec


def main():
    split = json.load(open(os.path.join(HERE, "split.json")))["papers"]
    claims = {json.loads(l)["problem_id"]: json.loads(l)
              for l in open("norm_cards/data/dataset20-claims-v3.jsonl")}
    refs = {pid: json.load(open(f"datasets/dataset20v3/references/problem_{pid}/ground_truth.json"))
            for pid in split}
    jobs = [(pid, c) for pid, p in split.items() for c in p["claims"] if c["verdict"] == "keep"]
    with ThreadPoolExecutor(7) as ex:
        out = list(ex.map(lambda j: draft(j[0], j[1], claims, refs), jobs))
    with open(os.path.join(HERE, "new_claims.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in out:
        print(f"\n== {r['problem_id']}  method:{r['_leaked_method_check']!r}  "
              f"apparatus:{r['_named_apparatus']}  selfref:{r['_self_reference']}\n{r['claim']}")


if __name__ == "__main__":
    main()
