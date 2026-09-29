"""Score a finished sweep: arm means, paired intervals, and where the gain lives.

    python -m norm_cards.eval.analyze --run results/myrun

Reads the judgments under `<run>/eval/judgments` and reports four things, in the
order they should be believed:

1. **Per-model arm means.** The ordering of nocard / rag / card, model by model.
   A result that only holds for one proposer is a result about that proposer.

2. **Paired bootstrap intervals.** Resampling is over CLAIMS, paired within claim,
   because between-claim variance dwarfs the effect — an unpaired interval on this
   data is wide enough to hide anything.

3. **The role split.** Recall over the whole reference answers "would this proposal
   have reproduced the paper?", which is not the question. A claim obliges its
   headline, apparatus, controls and confounds; a paper's mechanism and external
   follow-ups usually serve its OTHER claims. Scoring both ways separates "designed
   a good test of the claim" from "guessed more of what the authors did".

4. **recall@k.** The proposer is instructed to list experiments in priority order
   (`scify_proposer.py`), so the ranking is real and a prefix is meaningful: recall@k
   is what a team would recover if it could only run the first k. This is the metric
   that matters operationally, because SciFy's production budget is 3 and this
   harness runs at 9 — an effect that only appears at 9 is not yet an effect you can
   ship.

5. **Precision and strict recall**, as robustness. Precision is the share of
   proposals bearing on any reference item; it is NOT correctness, because under
   claim-is-the-contract an unmatched proposal may be an adequate route the authors
   did not take. Strict recall drops partial credit: if an effect needs halves to
   survive, say so.

6. **Per-card attribution.** Mean effect for each shared subfield card. Most claims
   are served by two or three cards, so a claim's change is credited to each of
   them — presence, not isolation. Read it as which cards to keep building.
"""

import argparse
import collections
import glob
import json
import os
import random
import statistics as st

ARMS = ("nocard", "rag", "card")
VALUE = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
BUDGETS = (3, 6, 9)
OBLIGED = {"headline", "apparatus", "control", "confound"}
BEYOND = {"mechanism", "external"}
RIGOUR = {"control", "confound"}


def arm_base(arm: str) -> str:
    """Arms are named "<model>+<condition>". Test nocard FIRST — it ends in "card"."""
    a = (arm or "").lower()
    if "nocard" in a:
        return "nocard"
    if "rag" in a:
        return "rag"
    if "retr" in a:
        return "retrieval"
    if "card" in a:
        return "card"
    return a


def boot(pairs, n=10000, seed=0):
    rng = random.Random(seed)
    d = [b - a for a, b in pairs]
    bs = sorted(st.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return st.fmean(d), bs[int(.025 * n)], bs[int(.975 * n)]


def load(run: str):
    gt_root = os.path.join(run, "eval", "ground_truth")
    roles = {}
    for p in glob.glob(os.path.join(gt_root, "problem_*", "ground_truth.json")):
        pid = os.path.basename(os.path.dirname(p)).replace("problem_", "")
        with open(p, encoding="utf-8") as f:
            roles[pid] = json.load(f).get("roles") or []
    full, sliced = {}, collections.defaultdict(dict)
    cov, nprop = {}, {}
    for p in glob.glob(os.path.join(run, "eval", "judgments", "problem_*", "*.json")):
        if os.path.basename(p).startswith("_"):      # _selftest and friends
            continue
        with open(p, encoding="utf-8") as f:
            j = json.load(f)
        key = (j["problem_id"], j.get("proposer_model", "?"), arm_base(j.get("arm", "")))
        full[key] = j["scores"]["recall"]
        # `covered_by` holds the proposal indices carrying each reference item, which
        # is what makes a prefix — and therefore recall@k — computable.
        cov[key] = [(VALUE.get(c.get("status"), 0.0), sorted(c.get("covered_by") or []))
                    for c in (j.get("reference_coverage") or [])]
        nprop[key] = j["scores"].get("n_proposed") or 0
        rl = roles.get(j["problem_id"], [])
        statuses = j["scores"].get("coverage_statuses") or []
        for sel, tag in ((OBLIGED, "obliged"), (BEYOND, "beyond"), (RIGOUR, "rigour")):
            vals = [{"covered": 1.0, "partial": 0.5, "missing": 0.0}.get(s, 0.0)
                    for i, s in enumerate(statuses) if i < len(rl) and rl[i] in sel]
            if vals:
                sliced[tag][key] = st.fmean(vals)
    return full, sliced, cov, nprop


def recall_at(items, k):
    """Value recovered if only the first k proposals, in priority order, were run."""
    if not items:
        return 0.0
    return sum(v for v, by in items if by and min(by) < k) / len(items)


def strict(items):
    """Recall with no partial credit — does the effect survive without halves?"""
    return st.fmean(1.0 if v == 1.0 else 0.0 for v, _ in items) if items else 0.0


def precision(items, n_prop):
    """Share of proposals bearing on any reference item. Overlap, NOT correctness."""
    used = {i for _, by in items for i in by}
    return len(used) / n_prop if n_prop else 0.0


def block(title, tbl, models, pids):
    print(f"\n  === {title} ===")
    print(f"  {'model':16s} " + " ".join(f"{a:>8s}" for a in ARMS))
    for m in models:
        cells = []
        for a in ARMS:
            v = [tbl[(p, m, a)] for p in pids if (p, m, a) in tbl]
            cells.append(f"{st.fmean(v):8.3f}" if v else f"{'—':>8s}")
        print(f"  {m:16s} " + " ".join(cells))
    for b in ("nocard", "rag"):
        pr = [(tbl[(p, m, b)], tbl[(p, m, "card")]) for p in pids for m in models
              if (p, m, b) in tbl and (p, m, "card") in tbl]
        if not pr:
            continue
        d, lo, hi = boot(pr)
        print(f"    pooled card − {b:7s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={len(pr)}"
              + ("  *" if (lo > 0 or hi < 0) else ""))


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default="", help="dataset name, for per-card attribution")
    args = ap.parse_args()

    full, sliced, cov, nprop = load(args.run)
    if not full:
        raise SystemExit(f"no judgments under {os.path.join(args.run, 'eval', 'judgments')}")
    models = sorted({m for _, m, _ in full})
    pids = sorted({p for p, _, _ in full})

    print("=" * 72)
    print(f"  {len(pids)} claims × {len(models)} models × {len(ARMS)} arms "
          f"= {len(full)} judgements")
    print("=" * 72)
    block("FULL RECALL", full, models, pids)
    for tag, title in (("obliged", "CLAIM-OBLIGED (headline+apparatus+control+confound)"),
                       ("beyond", "BEYOND-CLAIM (mechanism+external)"),
                       ("rigour", "CONTROLS + CONFOUNDS ONLY")):
        if sliced.get(tag):
            block(title, sliced[tag], models, pids)

    if cov:
        print("\n  === recall@k — value recovered within a k-experiment budget ===")
        print(f"  {'budget':>8s} " + " ".join(f"{a:>8s}" for a in ARMS) + "   Δcard−nocard")
        for k in BUDGETS:
            ms = []
            for a in ARMS:
                v = [recall_at(cov[(p, m, a)], k) for p in pids for m in models
                     if (p, m, a) in cov]
                ms.append(st.fmean(v) if v else 0.0)
            print(f"  {k:>8d} " + " ".join(f"{x:8.3f}" for x in ms)
                  + f"   {ms[2] - ms[0]:+.3f}")
        ms = [st.fmean([full[(p, m, a)] for p in pids for m in models if (p, m, a) in full])
              for a in ARMS]
        print(f"  {'full':>8s} " + " ".join(f"{x:8.3f}" for x in ms)
              + f"   {ms[2] - ms[0]:+.3f}")
        print("\n  === robustness ===")
        for name, fn in (("strict recall", lambda k: strict(cov[k])),
                         ("precision", lambda k: precision(cov[k], nprop[k]))):
            ms = [st.fmean([fn((p, m, a)) for p in pids for m in models if (p, m, a) in cov])
                  for a in ARMS]
            pr = [(fn((p, m, "nocard")), fn((p, m, "card"))) for p in pids for m in models
                  if (p, m, "nocard") in cov and (p, m, "card") in cov]
            d, lo, hi = boot(pr)
            print(f"  {name:16s} " + " ".join(f"{x:8.3f}" for x in ms)
                  + f"   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  + ("  *" if (lo > 0 or hi < 0) else ""))

    eff = {}
    for p in pids:
        d = [full[(p, m, "card")] - full[(p, m, "nocard")] for m in models
             if (p, m, "card") in full and (p, m, "nocard") in full]
        if d:
            eff[p] = st.fmean(d)

    if args.dataset:
        from .. import dataset as ds_mod
        sfmap = ds_mod.load_subfields(ds_mod.dataset_dir(args.dataset))
        per = collections.defaultdict(list)
        for p, e in eff.items():
            for s in sfmap.get(p, {}).get("subfields", []):
                per[s].append(e)
        print("\n  === PER SHARED CARD (attribution: a claim counts for each card serving it) ===")
        print(f"  {'card':40s} {'claims':>6s} {'mean Δ':>8s}")
        for s, v in sorted(per.items(), key=lambda kv: -st.fmean(kv[1])):
            print(f"  {s:40s} {len(v):6d} {st.fmean(v):+8.3f}")

    print("\n  === PER CLAIM (mean over models) ===")
    print(f"  {'claim':20s} " + " ".join(f"{a:>8s}" for a in ARMS) + "   Δcard−nocard")
    up = dn = 0
    for p in pids:
        cells = []
        for a in ARMS:
            v = [full[(p, m, a)] for m in models if (p, m, a) in full]
            cells.append(f"{st.fmean(v):8.3f}" if v else f"{'—':>8s}")
        d = eff.get(p)
        up += d is not None and d > 5e-4
        dn += d is not None and d < -5e-4
        print(f"  {p:20s} " + " ".join(cells) + (f"   {d:+.3f}" if d is not None else "   —"))
    print(f"\n  card beats nocard on {up} claims, loses on {dn}, "
          f"ties on {len(pids) - up - dn}")


if __name__ == "__main__":
    _main()
