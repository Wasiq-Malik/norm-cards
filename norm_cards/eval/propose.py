"""Generate proposer output for a claim set, one arm per (model, condition).

    python -m norm_cards.eval.propose --cards results/.../pipeline --arms nocard,retrieval,card

Writes `<out>/problem_<id>/scify_proposer.json` in the shape evaluate.py reads, with
an `arms` dict keyed by "<model>+<condition>".

The claim is decomposed ONCE, by a fixed decomposer, and the resulting subclaims are
handed to every arm. Decomposition is a separate pipeline stage; letting each model
decompose its own claim would fold two capabilities into one number and leave the
arms working from different problems.

CONDITIONS (`--arms`), each differing only in `current_evidence`:

  nocard      {} — the claim and its subclaims alone. What the model already knows.
  retrieval   the top-ranked retrieved papers, titles + truncated abstracts. The
              control that "card beats nocard" does not provide: a norm card is a
              synthesis OVER retrieved papers, so beating the empty condition only
              shows that relevant literature helps. Beating this one shows the
              synthesis does.
  card        the curated norm card.

`--sections` narrows what the card condition shows (resources / design / a
comma-separated subset of the six keys), which turns "what should we keep in the
card?" into a measurement: run an arm per subset and read off which sections carry
the effect. Such arms are named "<model>+card:<label>".

`--budget` varies SciFy's own "propose up to 3 experiments" cap. Leave it at 3 for
any headline number — 3 is what SciFy runs — and vary it only as a labelled
diagnostic, to check how much of a null result is the cap rather than the card.

Every run is stamped `ablation: true`, and evaluate.py reads that stamp and hides the
evidence from the judge — showing the judge the treatment would hand the carded arm
an automatic match on norm alignment and quietly invalidate the contrast.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from . import load_claims
from .. import scify_proposer

DEFAULT_MODELS = ("gpt-5.4", "gpt-5.5", "gpt-5.6")


RESOURCE_SECTIONS = ("datasets", "models", "metrics")
DESIGN_SECTIONS = ("protocols", "controls", "confounds")


def resolve_sections(spec: str) -> tuple:
    """--sections resources | design | all | a comma-separated subset."""
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return scify_proposer.MENU_KEYS
    if spec == "resources":
        return RESOURCE_SECTIONS
    if spec == "design":
        return DESIGN_SECTIONS
    picked = tuple(x.strip() for x in spec.split(",") if x.strip())
    bad = [x for x in picked if x not in scify_proposer.MENU_KEYS]
    if bad:
        raise SystemExit(f"unknown card section(s) {bad}; "
                         f"choose from {list(scify_proposer.MENU_KEYS)}")
    return picked


def build_conditions(names, card: dict, bundle: dict, sections: tuple,
                     label: str, claim: str = "", cache_dir: str = ".pdfcache") -> dict:
    """condition name -> the `current_evidence` dict that defines it."""
    out = {}
    # The card arm is built first so the retrieval arms can be trimmed to exactly
    # its size: a contrast between conditions of different context length measures
    # length as much as content.
    card_ev = (scify_proposer.norm_card_evidence(card, sections=sections)
               if card else {})
    budget = sum(len(v) for v in card_ev.values()) or 9500
    for n in names:
        if n == "nocard":
            out["nocard"] = {}
        elif n == "retrieval":
            ev = scify_proposer.retrieval_evidence(bundle) if bundle else {}
            if not ev:
                print("    retrieval arm needs a bundle.json with papers — skipping")
                continue
            out["retrieval"] = ev
        elif n == "rag":
            # Full text of the SAME papers the card was built from, top BM25
            # passages, trimmed to the card's own budget. The `retrieval` arm above
            # sees only abstracts, which the extraction pipeline's own notes record
            # as naming zero datasets — beating it proves little.
            ev = (scify_proposer.fulltext_retrieval_evidence(
                      bundle, claim, cache_dir=cache_dir, budget=budget)
                  if bundle else {})
            if not ev:
                print("    rag arm needs a bundle with fetchable full text — skipping")
                continue
            out["rag"] = ev
        elif n == "card":
            if not card:
                print("    card arm needs a norm_card.json — skipping")
                continue
            out[f"card:{label}" if label else "card"] = card_ev
        else:
            raise SystemExit(f"unknown arm {n!r}; choose from nocard, retrieval, card")
    return out


def propose_all(problem: dict, models, subclaims: list, conditions: dict,
                workers: int = 3, budget: int = scify_proposer.DEFAULT_BUDGET,
                repeats: int = 1) -> dict:
    """One proposer call per (model, condition, repeat), on identical inputs.

    `repeats` > 1 draws the same arm several times. The proposer runs at
    temperature 1 for the gpt-5 family, so a single draw per arm leaves proposer
    variance entirely uncontrolled while judge variance is controlled at k=3 —
    and resampling the same no-card condition was observed to move recall by up to
    0.17, which is larger than any card effect measured so far. Repeats are what
    turn that into a number instead of a worry. Extra draws are named
    "<model>+<condition>#<i>"."""
    claim = problem["claim"]
    jobs = [(f"{m}+{cond}" + (f"#{r}" if r else ""), m, ev)
            for m in models for cond, ev in conditions.items()
            for r in range(repeats)]

    def one(job):
        arm, model, ev = job
        try:
            return arm, {"model": model,
                         "condition": arm.split("+", 1)[1].split("#", 1)[0],
                         "repeat": int(arm.split("#")[1]) if "#" in arm else 0,
                         "evidence_chars": sum(len(v) for v in ev.values()),
                         "experiments": scify_proposer.propose(claim, subclaims, ev,
                                                               model=model,
                                                               budget=budget)}
        except Exception as e:                       # one bad arm must not sink the run
            return arm, {"model": model,
                         "condition": arm.split("+", 1)[1].split("#", 1)[0],
                         "repeat": int(arm.split("#")[1]) if "#" in arm else 0,
                         "experiments": [], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return dict(ex.map(one, jobs))


def load_json(cards_dir: str, pid: str, name: str) -> dict:
    path = os.path.join(cards_dir, f"problem_{pid}", name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated proposer models")
    ap.add_argument("--problems", default="all",
                    help="comma-separated problem ids, or 'all'")
    ap.add_argument("--decomposer", default="gpt-5.5",
                    help="model that produces the subclaims shared by every arm")
    ap.add_argument("--arms", default="nocard,retrieval,card",
                    help="conditions to run per model: nocard, retrieval (abstracts), "
                         "rag (full-text passages at the card's budget), card")
    ap.add_argument("--cards", default="",
                    help="dir of problem_<id>/{norm_card,bundle}.json — the card and "
                         "retrieval arms both read from here")
    ap.add_argument("--sections", default="all",
                    help="what the card arm shows: all | resources | design | a "
                         "comma-separated subset of "
                         "datasets,models,metrics,protocols,controls,confounds")
    ap.add_argument("--repeats", type=int, default=1,
                    help="draws per arm. >1 measures PROPOSER variance, which a "
                         "single draw leaves uncontrolled — the proposer runs at "
                         "temperature 1, and resampling one arm moved recall by up "
                         "to 0.17, more than any card effect measured so far")
    ap.add_argument("--budget", type=int, default=scify_proposer.DEFAULT_BUDGET,
                    help="SciFy's 'propose up to N experiments' cap (default 3, "
                         "which is what SciFy runs; vary only as a diagnostic)")
    ap.add_argument("--cache_dir", default=".pdfcache",
                    help="PDF cache the rag arm reads full text from")
    ap.add_argument("--out", default=os.path.join("results", "proposals"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    sections = resolve_sections(args.sections)
    label = "" if args.sections.strip().lower() == "all" else args.sections.strip().lower()
    claims = load_claims()
    ids = (sorted(claims) if args.problems == "all"
           else [p.strip() for p in args.problems.split(",") if p.strip()])

    for pid in ids:
        if pid not in claims:
            print(f"[{pid}] not in the claims file — skipping")
            continue
        out_dir = os.path.join(args.out, f"problem_{pid}")
        out_path = os.path.join(out_dir, "scify_proposer.json")
        if os.path.exists(out_path) and not args.force:
            print(f"[{pid}] proposals exist; --force to redo")
            continue

        problem = claims[pid]
        print(f"\n=== problem {pid} · decomposing with {args.decomposer} ===")
        subclaims = scify_proposer.decompose(problem["claim"], model=args.decomposer)
        print(f"    {len(subclaims)} subclaims")

        card = load_json(args.cards, pid, "norm_card.json") if args.cards else {}
        bundle = load_json(args.cards, pid, "bundle.json") if args.cards else {}
        conditions = build_conditions(names, card, bundle, sections, label,
                                      claim=problem["claim"], cache_dir=args.cache_dir)
        if not conditions:
            print(f"[{pid}] no runnable condition — skipping")
            continue
        for c, ev in conditions.items():
            print(f"    condition {c:16s} evidence {sum(len(v) for v in ev.values()):6d} chars")

        print(f"    proposing: {len(models)} model(s) x {len(conditions)} condition(s), "
              f"budget {args.budget}")
        arms = propose_all(problem, models, subclaims, conditions,
                           workers=args.workers, budget=args.budget,
                           repeats=args.repeats)

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"problem_id": pid, "claim": problem["claim"],
                       "decomposer_model": args.decomposer, "subclaims": subclaims,
                       "conditions": sorted(conditions), "budget": args.budget,
                       "repeats": args.repeats,
                       "card_sections": list(sections),
                       "card_format": card.get("format_version", ""),
                       "norm_card_used": bool(card),
                       # evaluate.py reads this stamp and hides the evidence from the
                       # judge; a contrast the judge can see is not a contrast.
                       "ablation": True,
                       "arms": arms}, f, indent=2)
        for m, a in sorted(arms.items()):
            print(f"      {m:26s} " + (a.get("error") or f"{len(a['experiments'])} experiments"))
        print(f"    -> {out_path}")


if __name__ == "__main__":
    _main()
