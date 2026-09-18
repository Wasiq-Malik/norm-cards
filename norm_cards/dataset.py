"""Materialise a tracked dataset into a runnable eval root.

    python -m norm_cards.dataset bootstrap --dataset dataset20v3 --run results/myrun
    python -m norm_cards.dataset assemble  --dataset dataset20v3 --run results/myrun

`datasets/<name>/` holds the INPUTS to an experiment — claims, references, the
shared subfield cards and the papers they were built from. `results/` holds the
OUTPUTS, is git-ignored, and is disposable: anything under it can be rebuilt from
a dataset plus API calls. Keeping the split explicit is what lets a new machine
reproduce a run without re-paying for card construction.

Two steps, because they answer different questions:

`bootstrap` lays out the eval root the harness expects — references where
`NORM_CARDS_EVAL_ROOT` will look for them.

`assemble` builds each claim's two proposer inputs from the SHARED subfield cards:

  norm_card.json   the claim's subfield cards merged through the same curation the
                   per-subfield path uses, so a claim in three fields gets one card
                   under one item cap rather than three cards' worth of items.

  bundle.json      the union of those subfields' retrieved papers. The `rag` arm
                   reads full text from here, so rag and card draw on exactly the
                   same source material and differ only in whether it was curated.

No claim text enters either one. The cards were retrieved with queries built from
the subfield name alone, which is the whole point: one card per field, reused
unchanged by every claim routed there.
"""

import argparse
import collections
import json
import os
import re
import shutil

from . import curate


def dataset_dir(name: str) -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "datasets", name)
    if not os.path.isdir(d):
        raise SystemExit(f"no dataset at {d}. Available: "
                         f"{sorted(os.listdir(os.path.join(here, 'datasets')))}")
    return d


def manifest(ds: str) -> dict:
    with open(os.path.join(ds, "dataset.json"), encoding="utf-8") as f:
        return json.load(f)


def claims_path(ds: str) -> str:
    """Claims live in the shared claim library, not in the dataset — one copy."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, manifest(ds)["claims"])


def load_subfields(ds: str) -> dict:
    with open(os.path.join(ds, "subfields.json"), encoding="utf-8") as f:
        return json.load(f)


def shared_menus(ds: str) -> dict:
    """subfield name -> its extracted menu, from the per-subfield cards."""
    out = {}
    root = os.path.join(ds, "cards")
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d, "norm_card.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
        for sub in card.get("norm_cards") or []:
            out.setdefault(sub["subfield"], sub["menu"])
        # fall back to the directory name if the reduce stage renamed the subfield
        name = d.replace("problem_subfield_", "").replace("_", " ")
        if name not in out and (card.get("norm_cards") or []):
            out[name] = card["norm_cards"][0]["menu"]
    return out


def bootstrap(ds: str, run: str) -> None:
    gt = os.path.join(run, "eval", "ground_truth")
    os.makedirs(gt, exist_ok=True)
    n = 0
    for d in sorted(os.listdir(os.path.join(ds, "references"))):
        src = os.path.join(ds, "references", d)
        dst = os.path.join(gt, d)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        n += 1
    print(f"  {n} references -> {gt}")
    m = manifest(ds)
    print(f"\n  {m['name']} — claims drafted with {m.get('claims_model','?')}, "
          f"references {'human-reviewed' if m.get('references_reviewed') else 'NOT human-reviewed'}")
    print(f"\n  export NORM_CARDS_EVAL_ROOT={os.path.join(run, 'eval')}")
    print(f"  export NORM_CARDS_CLAIMS={m['claims']}")


def assemble(ds: str, run: str, resource_cap: int, design_cap: int) -> None:
    sfmap = load_subfields(ds)
    menus = shared_menus(ds)
    bundles = {}
    with open(os.path.join(ds, "subfield_bundles.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                b = json.loads(line)
                bundles[b["subfield"]] = b
    print(f"  {len(menus)} shared subfield cards, {len(bundles)} paper bundles")

    out_root = os.path.join(run, "cards")
    served = skipped = 0
    for pid, rec in sorted(sfmap.items()):
        subs = [s for s in rec["subfields"] if s in menus]
        if not subs:
            print(f"  {pid:20s} none of {rec['subfields']} has a card — no card arm")
            skipped += 1
            continue
        out_dir = os.path.join(out_root, f"problem_{pid}")
        os.makedirs(out_dir, exist_ok=True)

        cards = [{"subfield": s, "menu": menus[s]} for s in subs]
        card, stats = curate.curate(cards, subfield_order=subs,
                                    resource_cap=resource_cap, design_cap=design_cap)
        with open(os.path.join(out_dir, "norm_card.json"), "w", encoding="utf-8") as f:
            json.dump({"type": "scientific_norm_card_set", "format_version": "0.3-shared",
                       "problem_id": pid, "subfields": subs, "claim": "",
                       "card": card, "curation": stats, "norm_cards": []}, f, indent=2)

        seen, papers = set(), []
        for s in subs:
            for p in (bundles.get(s) or {}).get("papers") or []:
                key = (p.get("title") or "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    papers.append(p)
        with open(os.path.join(out_dir, "bundle.json"), "w", encoding="utf-8") as f:
            json.dump({"type": "norm_card_paper_bundle", "format_version": "0.3-shared",
                       "problem_id": pid, "subfields": subs,
                       "claim": f"Experimental norms in {', '.join(subs)}.",
                       "papers": papers}, f, indent=2)

        print(f"  {pid:20s} {stats['totals']['out']:3d} items, {len(papers):3d} papers "
              f"from {len(subs)} card(s): {', '.join(subs)}")
        served += 1

    print(f"\n  served {served} claims, {skipped} without a card -> {out_root}")
    leaks = leakage(out_root)
    print(f"  source-paper leakage: {len(leaks)}/{served}"
          + ("" if not leaks else f"  !! {leaks}"))


def leakage(out_root: str) -> list:
    """Claims whose own source paper turned up in the papers their card was built from.

    Shared cards are retrieved per field, so a claim's own paper can be pulled in by
    a neighbour's query. That would let the proposer read the very experiment section
    the reference was transcribed from, and the run would measure copying.
    """
    bad = []
    for d in sorted(os.listdir(out_root)):
        pid = d.replace("problem_", "")
        m = re.match(r"arxiv(\d{4})_(\d+)$", pid)
        if not m:
            continue
        want = f"{m.group(1)}.{m.group(2)}"
        with open(os.path.join(out_root, d, "bundle.json"), encoding="utf-8") as f:
            b = json.load(f)
        if any(want in str(p.get("arxiv_id") or "") for p in b["papers"]):
            bad.append(pid)
    return bad


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["bootstrap", "assemble"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="eval root to build under, e.g. results/myrun")
    ap.add_argument("--resource_cap", type=int, default=8)
    ap.add_argument("--design_cap", type=int, default=14)
    args = ap.parse_args()
    ds = dataset_dir(args.dataset)
    if args.cmd == "bootstrap":
        bootstrap(ds, args.run)
    else:
        assemble(ds, args.run, args.resource_cap, args.design_cap)


if __name__ == "__main__":
    _main()
