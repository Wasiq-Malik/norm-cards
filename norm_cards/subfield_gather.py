"""Gather papers for a SUBFIELD, not for a claim.

    python -m norm_cards.subfield_gather --subfields results/dataset20v2/subfields.json \
        --out results/dataset20v2/shared

A norm card is supposed to describe what a FIELD does. Retrieving with the claim's
own text — which is what the per-claim pipeline does — makes the card partly a
function of the question being asked, and invites the obvious objection that the
card helped because it was built for that claim. This module removes the claim
from the loop entirely: queries come from the subfield's own name and taxonomy
keywords, so one card is built per subfield and served unchanged to every claim
routed there.

It is also cheaper. Twenty claims over twenty-six subfields collapse to eleven
retrievals, because a subfield shared by six claims is retrieved once.
"""

import argparse
import collections
import json
import os
import time

from concurrent.futures import ThreadPoolExecutor

from . import pipeline, llm
from .sources import get_sources
from .models import ClaimAnalysis
from .taxonomy import Taxonomy


# What a norm card needs from a field: the standard evaluations, the protocols, and
# the things that make a result hold up. Deliberately generic — no claim enters here.
QUERY_SHAPES = [
    "{sf} standard benchmarks and evaluation protocol",
    "{sf} experimental setup ablation study",
    "{sf} baselines and controls comparison",
    "{sf} reproducibility evaluation methodology",
    "{sf} survey of evaluation practice",
]


def queries_for(subfield: str, taxonomy: Taxonomy, per_keyword: int = 3):
    """Queries built from the subfield and its taxonomy keywords, never from a claim."""
    entry = taxonomy.get(subfield) or {}
    kws = (entry.get("keywords") or [])[:per_keyword]
    out = collections.defaultdict(list)
    for shape in QUERY_SHAPES:
        out["norms"].append(shape.format(sf=subfield))
    for k in kws:
        out["keyword"].append(f"{k} benchmark evaluation")
        out["keyword"].append(f"{k} experimental protocol")
    return dict(out)


# The templates above put the subfield's name into generic shapes, and the generic
# shapes retrieve generic papers: through dataset50's first gather only 68% of bundle
# papers were on-topic (dataset34v8: 71%; Activation Steering 9/30 in both), and the
# same off-topic 2025-26 evaluation papers filled the recent-paper quota of up to 8
# bundles. So from dataset50 an LLM writes the queries from the subfield's name and
# keywords — still never from a claim — and screens the ranked candidates.
QUERY_MODEL = "gpt-6-luna"

_QUERY_PROMPT = """You are building a literature search for the machine-learning subfield
"{sf}". Its taxonomy keywords: {kws}.

Write {n} search queries (3-8 words each) for an academic search engine (OpenAlex,
arXiv) that together retrieve the papers whose EXPERIMENTS show how this subfield
evaluates work:
  - its standard benchmarks and evaluation suites, and the papers that introduced them
  - its canonical methods, the ones new work compares against
  - papers that criticise or fix how it evaluates (weak baselines, flawed metrics,
    leakage, unreliable judges or attacks)
  - strong recent empirical papers

Every query must use vocabulary specific to this subfield as it is studied in machine
learning. Where a word is shared with another field, disambiguate it ("decoding" also
means brain decoding and channel decoding; "optimization" also means operations
research and numerical solvers; "survey" also means opinion surveys). Generic queries
such as "evaluation methodology" or "reproducibility" retrieve other fields' papers:
do not write them.

Return JSON {{"queries": ["...", "..."]}}"""

_SCREEN_PROMPT = """Subfield: {sf}
Paper title: {t}
Abstract: {a}

Would a researcher in this subfield (as it is studied in machine learning / AI) treat
this paper as part of its literature, such that its experiments show how the subfield
runs and evaluates experiments?
Answer JSON {{"on_topic": true|false, "reason": "<= 12 words"}}"""


def llm_queries(subfield: str, taxonomy: Taxonomy, n: int = 12, model: str = QUERY_MODEL) -> dict:
    kws = ", ".join((taxonomy.get(subfield) or {}).get("keywords") or []) or "(none)"
    d = llm.complete_json(_QUERY_PROMPT.format(sf=subfield, kws=kws, n=n), model=model)
    return {"llm": [q.strip() for q in d.get("queries") or [] if q and q.strip()][:n]}


def relevance_screen(subfield: str, model: str = QUERY_MODEL, workers: int = 16):
    """A `screen` for pipeline.gather_from_analysis: keep candidates the model calls
    on-topic for the subfield from title and abstract. A failed call keeps the paper."""
    def keep(papers):
        def one(p):
            try:
                d = llm.complete_json(_SCREEN_PROMPT.format(
                    sf=subfield, t=p.title, a=(p.abstract or "")[:1500]), model=model)
                return d.get("on_topic") is not False
            except Exception:
                return True
        with ThreadPoolExecutor(workers) as ex:
            ok = list(ex.map(one, papers))
        return [p for p, k in zip(papers, ok) if k]
    return keep


def gather(subfield: str, taxonomy: Taxonomy, sources, top_k: int = 30,
           per_query: int = 8, require_fulltext: bool = False,
           queries: str = "llm", screen: bool = True) -> dict:
    """`queries="template", screen=False` reproduces dataset34v8's gather."""
    an = ClaimAnalysis(subfields=[subfield], method="subfield-direct", confidence=1.0)
    qs = llm_queries(subfield, taxonomy) if queries == "llm" else queries_for(subfield, taxonomy)
    bundle = pipeline.gather_from_analysis(an, qs, taxonomy, sources,
                                           per_query=per_query, top_k=top_k,
                                           require_fulltext=require_fulltext,
                                           screen=relevance_screen(subfield) if screen else None)
    bundle.update({"type": "norm_card_paper_bundle", "format_version": "0.3-subfield",
                   "problem_id": "subfield_" + subfield.replace("/", "_").replace(" ", "_"),
                   "subfield": subfield,
                   # `claim` is what normcard's reduce/recipe stages read. A subfield
                   # bundle has none, so it gets the field's name — enough for the
                   # reduce step's relevance judgement, and carrying no question.
                   "claim": f"Experimental norms and standard practice in {subfield}."})
    return bundle


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subfields", required=True,
                    help="JSON of problem_id -> {subfields: [...]}, from the classifier")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_claims", type=int, default=2,
                    help="only retrieve subfields serving at least this many claims; "
                         "a singleton subfield cannot be reused and is not worth a card")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--sources", default="openalex,arxiv")
    ap.add_argument("--require_fulltext", action="store_true",
                    help="choose the top_k only from candidates whose full text is fetchable")
    ap.add_argument("--queries", default="llm", choices=["llm", "template"],
                    help="llm: gpt-6-luna writes the queries from the subfield name and "
                         "keywords; template: QUERY_SHAPES, as through dataset34v8")
    ap.add_argument("--no_screen", action="store_true",
                    help="skip the gpt-6-luna relevance screen of ranked candidates")
    ap.add_argument("--only", default="", help="comma-separated subfields to gather")
    args = ap.parse_args()

    sf_map = json.load(open(args.subfields, encoding="utf-8"))
    cnt = collections.Counter(s for v in sf_map.values() for s in v["subfields"])
    wanted = [s for s, n in cnt.most_common() if n >= args.min_claims]
    if args.only:
        wanted = [s for s in wanted if s in {x.strip() for x in args.only.split(",")}]
    print(f"{len(wanted)} subfield(s) serving >= {args.min_claims} claims "
          f"(of {len(cnt)} total)\n")

    tax = Taxonomy.load()
    sources = get_sources(names=[x.strip() for x in args.sources.split(",") if x.strip()])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "subfield_bundles.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["subfield"] for l in open(path, encoding="utf-8") if l.strip()}

    for i, sf in enumerate(wanted, 1):
        if sf in done:
            print(f"[{i}/{len(wanted)}] {sf}: already gathered")
            continue
        print(f"\n=== [{i}/{len(wanted)}] {sf}  (serves {cnt[sf]} claims) ===", flush=True)
        try:
            b = gather(sf, tax, sources, top_k=args.top_k,
                       require_fulltext=args.require_fulltext,
                       queries=args.queries, screen=not args.no_screen)
        except Exception as e:
            print(f"  FAILED, continuing: {type(e).__name__}: {str(e)[:140]}", flush=True)
            continue
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(b) + "\n")
        print(f"  -> {len(b.get('papers') or [])} papers", flush=True)
        time.sleep(2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    _main()
