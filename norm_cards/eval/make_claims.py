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

A claim is a proposition that could turn out false. Someone handed only this claim should
understand what question is at stake AND what would settle it, without being told the answer.

THE BAR IS PART OF THE CLAIM. THE FINDING IS NOT.
This distinction is the whole task, and getting it wrong in either direction ruins the claim.

  A BAR is the threshold that decides the question. It belongs in the claim, WITH ITS NUMBER:
      "...is considered robust if the post-training attack success rate remains within 5% of
       a model trained on clean data only"
      "...mediated by fewer than 50 attention heads in the final 10 layers"
      "...retains AP50 >= 0.90 with <= 5% relative degradation from clean performance"

  A FINDING is what the authors got when they ran it. It must NOT appear:
      "editing degrades reasoning by 27% on AIME"        <- the answer, omit
      "SWYB achieves 100.0% syntactic validity"          <- the answer, omit

  Write the bar even when the paper states it only implicitly, by reporting a comparison: if the
  authors treat "beats the strongest prior baseline under an equal training budget" as the test,
  say exactly that. NEVER substitute a vague word for a bar. "reliable", "comparable",
  "competitive", "substantially intact", "high accuracy", "robust" are not bars: each one hides
  the number that decides the claim, and a claim that cannot be decided is useless here.

NAME THE SUBJECT. DO NOT NAME THE APPARATUS.
The claim names what it is ABOUT. It does not name the equipment the authors happened to use to
look at it. This is the third failure mode, and it is the one that quietly destroys the
measurement:

  SUBJECT - belongs in the claim. The system, model family or phenomenon whose behaviour is at
  stake: "a vision-language-action manipulation policy", "OpenVLA-7B", "parameter-modifying
  knowledge-editing methods". Where the claim really is about one specific artifact, name it.

  APPARATUS - must NOT appear. Which datasets it was measured on, which baselines it was compared
  against, which prompts, seeds, splits, sampling settings or protocol were used. Those are
  DESIGN DECISIONS, and designing them is exactly what is being measured here. Handing them over
  leaves the proposer nothing to design:
      "...on 256 BigCodeBench problems, using identical datasets, sampling settings and
       prompt templates"                                  <- apparatus, omit
      "...against PGD-AT, TRADES, MART and Cons-AT"       <- apparatus, omit
      "...trained and evaluated on CelebA-HQ at 128x128 and 256x256"   <- apparatus, omit

  KEEP THE BAR, DROP THE APPARATUS. They are different things and the claim needs the first
  without the second:
      "...the behavioural marker that differs between evaluation and deployment contexts comes
       within 5 percentage points of the model's own deployment rate, while task accuracy
       degrades by no more than 5% relative"
  That states a decidable bar with numbers and names no dataset. That is the target shape.

  TEST: if a competent researcher in the field could have picked a different dataset, baseline
  set or protocol and still tested the same proposition, it is apparatus - leave it out. Name at
  most two or three specific artifacts in the whole claim, and only where they are the subject.

WHAT STILL MUST NOT APPEAR: the authors' own METHOD — the technique they invented to answer the
question, and its name. If the paper's contribution is called SWYB or HYVE or EUCLEAN, the claim
says what property is at stake, never that name or its mechanism. Naming it hands over the
experimental design, which is the thing being measured.

FORM: three to six sentences, plain prose, no bullets, self-contained, no "this paper", no
citations. Aim for the length of the examples above rather than an exhaustive specification.

Also return `subfields`: 2-3 research areas this sits in, as a working scientist would name them.

PAPER TITLE: {title}

PAPER TEXT:
{text}

Return JSON: {{"claim": "...", "subfields": ["...", "..."],
               "decision_rule": "the sentence in your claim that states the bar, quoted verbatim",
               "leaked_method_check": "quote any phrase in your claim naming the authors' own
                                       technique or its mechanism, or 'none'",
               "named_apparatus": ["every dataset, benchmark or baseline named in your claim;
                                    empty list if none, which is the usual correct answer"]}}"""


# A claim must stand on its own. "the authors' system", "this paper", "our method"
# all smuggle the paper back in — and worse, they point at the very technique the
# claim is supposed to withhold, so a proposer reads "whatever they built" and the
# measurement is gone.
SELF_REF = re.compile(r"\bthe authors'?\b|\bthis (?:paper|work|study)\b|"
                      r"\bwe (?:propose|introduce|present)\b|\bour (?:method|approach|system|model)\b",
                      re.I)


def _pid(arxiv_id: str) -> str:
    return "arxiv" + re.sub(r"[^0-9]", "_", arxiv_id)


def draft_one(paper: dict, cache_dir: str, model: str) -> dict:
    text, _ = fulltext.fetch_fulltext(
        {"arxiv_id": paper["id"], "title": paper.get("title", "")}, cache_dir, max_pages=fulltext.FULL_PAPER)
    if not text:
        text = paper.get("abstract") or ""
        source = "abstract only — full text unavailable"
    else:
        source = "full text"
    out = llm.complete_json(
        DRAFT_PROMPT.format(title=paper.get("title", ""), text=text[:400000]), model=model)
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
        "_decision_rule": out.get("decision_rule", ""),
        "_named_apparatus": out.get("named_apparatus") or [],
    }


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--papers", required=True,
                    help="JSON list of {id, title, v1, venue, theme, abstract}")
    ap.add_argument("--out", required=True, help="claims JSONL to write")
    ap.add_argument("--model", default="gpt-5.6-sol")
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
        sr = SELF_REF.search(rec["claim"])
        if sr:
            flag += f"   ⚠ NOT self-contained: {sr.group(0)!r}"
        print(f"[{i}/{len(papers)}] {pid}  {rec['provenance']['venue']:7s} "
              f"v1={rec['provenance']['arxiv_v1_date']}  {rec['provenance']['theme']}{flag}")
        print(f"        {rec['claim'][:150]}")
        with open(args.out, "w", encoding="utf-8") as f:
            for r in existing.values():
                f.write(json.dumps(r) + "\n")
    weak = [r for r in existing.values()
            if not re.search(r"\d", r.get("_decision_rule") or "")]
    if weak:
        print(f"\n{len(weak)} claim(s) whose stated decision rule carries NO number — check these "
              f"first, a claim without a bar cannot be decided:")
        for r in weak:
            print(f"   {r['problem_id']}: {(r.get('_decision_rule') or '(none given)')[:90]}")
    print(f"\n-> {args.out}   {len(existing)} claim(s). Every one is needs_review: "
          f"read it against the paper before it becomes a fixture.")


if __name__ == "__main__":
    _main()
