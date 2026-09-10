"""Draft claim records from source papers, for a person to review.

    python -m norm_cards.eval.make_claims --papers candidates.json --out claims.jsonl

The missing stage between "we picked these papers" and `transcribe.py`, which needs
a claim to already exist. One record per paper, in the claims-file format.

WHAT A CLAIM IS HERE, AND WHAT IT IS NOT
----------------------------------------
A claim states what the authors set out to ESTABLISH — the proposition their
experiments were designed to decide — not what they found. Two failure modes have
already cost this project a rewrite each:

  results as claim   "Editing degrades reasoning by 27% on AIME" hands the answer
                     to the proposer. It then designs experiments to confirm a
                     number rather than to decide a question, and recall measures
                     reading comprehension.

  method as claim    "Take the SVD of the hidden-state matrix and compute the
                     directional relative change rate" spoon-feeds the design. The
                     whole point is to see whether a system can arrive at an
                     adequate instrument on its own, so naming the instrument in
                     the claim destroys the measurement.

The claim carries the requirements; the reference is one instance satisfying them.
A proposer that meets the requirements by another adequate route is correct, and
the judge is built to score it that way.

Every record is stamped `needs_review: true` and carries the paper's arXiv v1 date,
which is the contamination filter that matters — a paper accepted at a 2026 venue
may have been public since 2024, in which case a proposer may well have read it.
Conference date tells you nothing; v1 does.
"""

import argparse
import json
import os
import re

from .. import llm, fulltext

DRAFT_PROMPT = """Below is a paper. Write the CLAIM its experiments were designed to decide.

A claim is a proposition that could turn out false. Someone handed only this claim
should understand what question is at stake and what would settle it, without being
told the answer or the method.

HARD CONSTRAINTS — each of these has broken this dataset before:

1. NO RESULTS. Do not state what the authors found, no headline numbers, no "X
   degrades by 27%". State what was in question before they ran anything. If the
   paper's contribution is a negative finding, the claim is the proposition they
   set out to test, phrased so that either outcome is possible.

2. NO METHOD. Do not name the specific instrument, statistic, architecture or
   procedure the authors invented to answer it. If the claim names the technique,
   a system reading it has been handed the experimental design. Name the artifact
   under test and the property at issue; leave HOW to establish it open.

3. KEEP WHAT CONSTRAINS. Do keep the things that make the claim decidable and
   scoped: the class of system, the setting, the regime, and any threshold the
   authors themselves treat as the bar. A claim with no commitments cannot be
   refuted and is useless here.

4. SELF-CONTAINED. No "this paper", no citations, no reference to the authors. It
   should read like a proposition someone wrote down before the work existed.

Three to six sentences. Plain prose, no bullets, no headings.

Also return `subfields`: 2-3 research areas this sits in, as a working scientist
would name them (e.g. "Vision-Language-Action Models", "Knowledge Editing").

PAPER TITLE: {title}

PAPER TEXT:
{text}

Return JSON: {{"claim": "...", "subfields": ["...", "..."],
               "what_is_at_stake": "one line: what a reader learns from the answer",
               "leaked_method_check": "quote any phrase in your claim that names the
                                       authors' specific technique, or 'none'"}}"""


def _pid(arxiv_id: str) -> str:
    return "arxiv" + re.sub(r"[^0-9]", "_", arxiv_id)


def draft_one(paper: dict, cache_dir: str, model: str) -> dict:
    text, _ = fulltext.fetch_fulltext(
        {"arxiv_id": paper["id"], "title": paper.get("title", "")}, cache_dir)
    if not text:
        text = paper.get("abstract") or ""
        source = "abstract only — full text unavailable"
    else:
        source = "full text"
    out = llm.complete_json(
        DRAFT_PROMPT.format(title=paper.get("title", ""), text=text[:150000]), model=model)
    return {
        "type": "problem", "format_version": "1.0", "problem_version": "1.0",
        "domain": "ai", "problem_id": _pid(paper["id"]),
        "claim": (out.get("claim") or "").strip(),
        "artifacts": [],
        "needs_review": True,
        "provenance": {
            "arxiv_id": paper["id"],
            # The contamination filter. NOT the venue date: an ICML-2026 paper on
            # arXiv since 2024 is not unseen, whatever the proceedings say.
            "arxiv_v1_date": paper.get("v1", ""),
            "venue": paper.get("venue", ""),
            "theme": paper.get("theme", ""),
            "title": paper.get("title", ""),
            "drafted_from": source, "drafter_model": model,
        },
        "_subfields_hint": out.get("subfields") or [],
        "_what_is_at_stake": out.get("what_is_at_stake", ""),
        "_leaked_method_check": out.get("leaked_method_check", ""),
    }


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--papers", required=True,
                    help="JSON list of {id, title, v1, venue, theme, abstract}")
    ap.add_argument("--out", required=True, help="claims JSONL to write")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--cache_dir", default=".pdfcache")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    papers = json.load(open(args.papers, encoding="utf-8"))
    if args.limit:
        papers = papers[:args.limit]
    existing = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                existing[r["problem_id"]] = r

    for i, p in enumerate(papers, 1):
        pid = _pid(p["id"])
        if pid in existing:
            print(f"[{i}/{len(papers)}] {pid} exists — skipping")
            continue
        try:
            rec = draft_one(p, args.cache_dir, args.model)
        except Exception as e:
            print(f"[{i}/{len(papers)}] {pid} FAILED: {type(e).__name__}: {e}")
            continue
        existing[pid] = rec
        leak = (rec.get("_leaked_method_check") or "none").strip().lower()
        flag = "" if leak in ("none", "", "n/a") else f"   ⚠ possible method leak: {leak[:70]}"
        print(f"[{i}/{len(papers)}] {pid}  {rec['provenance']['venue']:7s} "
              f"v1={rec['provenance']['arxiv_v1_date']}  {rec['provenance']['theme']}{flag}")
        print(f"        {rec['claim'][:150]}")
        with open(args.out, "w", encoding="utf-8") as f:
            for r in existing.values():
                f.write(json.dumps(r) + "\n")
    print(f"\n-> {args.out}   {len(existing)} claim(s). Every one is needs_review: "
          f"read it against the paper before it becomes a fixture.")


if __name__ == "__main__":
    _main()
