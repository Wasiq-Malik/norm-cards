"""Turn a batch paper-gathering run into per-problem norm cards.

    python -m norm_cards.eval.build_cards --run results/eval_icml2026/pipeline

`norm_cards.subfield_gather` writes every subfield's bundle into one
`paper_bundles.jsonl`, while `normcard.py` works on a single bundle. This splits the
former into the `problem_<id>/` layout the rest of the harness expects and generates
a card for each, so the pipeline — subfields -> papers -> norms -> card — runs end to
end with one command per stage.

Also reports whether a claim's own source paper turned up in its bundle. For claims
derived from a specific paper that is leakage worth knowing about: the pipeline would
be handing the proposer the very experiment section the reference was transcribed
from, and the evaluation would measure copying rather than experimental design.
"""

import argparse
import json
import os

from .. import normcard


def source_papers(ref_dir: str) -> dict:
    """problem_id -> the identifiers of the paper its reference was taken from."""
    out = {}
    if not os.path.isdir(ref_dir):
        return out
    for d in sorted(os.listdir(ref_dir)):
        path = os.path.join(ref_dir, d, "ground_truth.json")
        if not d.startswith("problem_") or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            ref = json.load(f)
        blob = " ".join((ref.get("provenance") or {}).get("sources") or []).lower()
        out[d.replace("problem_", "")] = blob
    return out


def leakage(bundle: dict, source_blob: str) -> list:
    """Papers in the bundle that look like the claim's own source paper."""
    if not source_blob:
        return []
    hits = []
    for p in bundle.get("papers") or []:
        ax = str(p.get("arxiv_id") or "").strip().lower()
        if ax and ax.split("v")[0] in source_blob:
            hits.append(p.get("title", ax))
            continue
        title = (p.get("title") or "").strip().lower()
        if len(title) > 25 and title[:60] in source_blob:
            hits.append(p.get("title"))
    return hits


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True,
                    help="dir holding paper_bundles.jsonl from norm_cards.subfield_gather")
    ap.add_argument("--references", default="",
                    help="ground_truth dir, for the source-paper leakage check")
    ap.add_argument("--cache_dir", default=".pdfcache")
    ap.add_argument("--map_model", default="gpt-5-mini")
    ap.add_argument("--reduce_model", default="gpt-5")
    ap.add_argument("--recipe_model", default="gpt-5")
    ap.add_argument("--min_evidence", type=int, default=2,
                    help="papers a RESOURCE needs before it counts as a norm rather "
                         "than one group's choice (design items are exempt: they are "
                         "scarce and are the part that transfers)")
    ap.add_argument("--resource_cap", type=int, default=8)
    ap.add_argument("--design_cap", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="build at most this many cards, then exit. Cards are "
                         "expensive and long-lived processes accumulate the "
                         "full text of every paper they have read; a bounded "
                         "limit keeps memory flat when a driver re-runs this "
                         "until the run is complete (already-built cards are "
                         "skipped, so re-running resumes).")
    args = ap.parse_args()

    path = os.path.join(args.run, "paper_bundles.jsonl")
    with open(path, encoding="utf-8") as f:
        bundles = [json.loads(l) for l in f if l.strip()]
    print(f"{len(bundles)} bundles in {path}")

    srcs = source_papers(args.references) if args.references else {}
    built = 0

    for b in bundles:
        pid = str(b.get("problem_id") or (b.get("analysis") or {}).get("problem_id") or "")
        if not pid:
            print("  bundle with no problem_id — skipping")
            continue
        out_dir = os.path.join(args.run, f"problem_{pid}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as f:
            json.dump(b, f, indent=2)

        hits = leakage(b, srcs.get(pid, ""))
        flag = (f"  ⚠️  LEAKAGE: the claim's own source paper is in the bundle "
                f"({len(hits)}): {hits}" if hits else "  no source-paper leakage")

        card_path = os.path.join(out_dir, "norm_card.json")
        if os.path.exists(card_path) and not args.force:
            print(f"[{pid}] card exists; --force to redo.{flag}")
            continue

        print(f"\n=== [{pid}] {len(b.get('papers') or [])} papers, "
              f"subfields={(b.get('analysis') or {}).get('subfields')} ===\n{flag}")
        card = normcard.generate_card(b, args.cache_dir, map_model=args.map_model,
                                      reduce_model=args.reduce_model,
                                      recipe_model=args.recipe_model,
                                      min_evidence=args.min_evidence,
                                      resource_cap=args.resource_cap,
                                      design_cap=args.design_cap)
        card.setdefault("problem_id", pid)
        with open(card_path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        sizes = {k: len((card.get("card") or {}).get(k) or []) for k in normcard.MENU_KEYS}
        print(f"    -> {card_path}  curated card {sizes}", flush=True)
        built += 1
        if args.limit and built >= args.limit:
            print(f"\nreached --limit {args.limit}; exiting (re-run to continue)")
            return


if __name__ == "__main__":
    _main()
