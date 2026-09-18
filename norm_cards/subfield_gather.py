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

from . import pipeline
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


def gather(subfield: str, taxonomy: Taxonomy, sources, top_k: int = 30,
           per_query: int = 8) -> dict:
    an = ClaimAnalysis(subfields=[subfield], method="subfield-direct", confidence=1.0)
    qs = queries_for(subfield, taxonomy)
    bundle = pipeline.gather_from_analysis(an, qs, taxonomy, sources,
                                           per_query=per_query, top_k=top_k)
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
    args = ap.parse_args()

    sf_map = json.load(open(args.subfields, encoding="utf-8"))
    cnt = collections.Counter(s for v in sf_map.values() for s in v["subfields"])
    wanted = [s for s, n in cnt.most_common() if n >= args.min_claims]
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
            b = gather(sf, tax, sources, top_k=args.top_k)
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
