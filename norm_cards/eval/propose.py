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

`--budget` is SciFy's "propose up to N experiments" cap. SciFy runs 3; our default is
5 (RUN_BUDGET): on dataset50, 5 lost no recall against 9 and the card's gain held.

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

# gpt-6 family only from dataset50 (2026-10-29); dataset34v4-v8 ran gpt-5.4, gpt-5.5 and
# gpt-5.6-sol as well. gpt-6-luna is the judge and card builder, gpt-6-astra too costly.
DEFAULT_MODELS = ("gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra")


RESOURCE_SECTIONS = ("datasets", "models", "metrics")
DESIGN_SECTIONS = ("protocols", "controls", "confounds")


def resolve_sections(spec: str) -> tuple:
    """--sections resources | design | all | a comma-separated subset."""
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None          # every section the card's own format has (0.3 or 0.4)
    if spec == "resources":
        return RESOURCE_SECTIONS
    if spec == "design":
        return DESIGN_SECTIONS
    picked = tuple(x.strip() for x in spec.split(",") if x.strip())
    known = scify_proposer.MENU_KEYS + scify_proposer.MENU_KEYS_V04
    bad = [x for x in picked if x not in known]
    if bad:
        raise SystemExit(f"unknown card section(s) {bad}; choose from {list(known)}")
    return picked


def build_conditions(names, card: dict, bundle: dict, sections: tuple,
                     label: str, claim: str = "", cache_dir: str = ".pdfcache",
                     prompt: str = None) -> dict:
    """condition name -> the `current_evidence` dict that defines it. Under the fieldnorms
    prompt the card (its titled text sections) or the rag passages ride in FIELD_NORMS_KEY, which the prompt
    renders as its own Field Norms block, and Current Evidence is empty in every condition. Under
    fieldnorms2 the rag passages ride in LITERATURE_KEY instead (a Relevant Literature block)."""
    prompt = prompt or scify_proposer.PROPOSER_PROMPT
    out = build_conditions_scify(names, card, bundle, sections, label, claim, cache_dir)
    if prompt == "scify":
        return out
    fn = {}
    for c, ev in out.items():
        if c.startswith("card"):
            fn[c] = {scify_proposer.FIELD_NORMS_KEY: scify_proposer.norm_card_text(card, sections)}
        elif c in ("rag", "retrieval"):
            key = scify_proposer.LITERATURE_KEY if prompt == "fieldnorms2" else scify_proposer.FIELD_NORMS_KEY
            fn[c] = {key: "\n\n".join(ev.values())}
        else:
            fn[c] = {}
    return fn


def build_conditions_scify(names, card: dict, bundle: dict, sections: tuple,
                           label: str, claim: str = "", cache_dir: str = ".pdfcache") -> dict:
    """condition name -> `current_evidence` as the SciFy-verbatim prompt took it."""
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
            # passages. Not trimmed to the card's size: retrieval is itself a way of
            # delivering norms, and the comparison that matters is against nocard.
            ev = (scify_proposer.fulltext_retrieval_evidence(
                      bundle, claim, cache_dir=cache_dir)
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
                workers: int = 3, budget: int = scify_proposer.RUN_BUDGET,
                repeats: int = 1, prompt: str = None) -> dict:
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
            exps = scify_proposer.propose(claim, subclaims, ev, model=model, budget=budget,
                                          prompt=prompt)
            # The budget is a cap, not a suggestion: Qwen3.8-27B returned 12 experiments
            # for "up to 9", and every extra one is another chance to cover a reference.
            rec = {"model": model,
                   "condition": arm.split("+", 1)[1].split("#", 1)[0],
                   "repeat": int(arm.split("#")[1]) if "#" in arm else 0,
                   "evidence_chars": sum(len(v) for v in ev.values()),
                   "experiments": exps[:budget]}
            if len(exps) > budget:
                rec["n_returned"] = len(exps)
            return arm, rec
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
    ap.add_argument("--prompt", default=scify_proposer.PROPOSER_PROMPT, choices=["fieldnorms", "fieldnorms2", "scify"],
                    help="fieldnorms2 (default): card as Field Norms, passages as Relevant Literature, "
                         "empty fields say None, each arm's system prompt explains only its own key; "
                         "fieldnorms: card or passages under one shared Field Norms key; "
                         "scify: SciFy's prompt verbatim, everything in Current Evidence")
    ap.add_argument("--decomposer", default="gpt-6-luna",
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
    ap.add_argument("--card_label", default="",
                    help="name the card arm 'card:<label>' — for a second set of cards, "
                         "beside the default card arm")
    ap.add_argument("--budget", type=int, default=scify_proposer.RUN_BUDGET,
                    help="'propose up to N experiments' cap (default 5; SciFy itself runs 3)")
    ap.add_argument("--cache_dir", default=".pdfcache",
                    help="PDF cache the rag arm reads full text from")
    ap.add_argument("--out", default=os.path.join("results", "proposals"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="add arms to an existing proposals file instead of skipping "
                         "the claim. The skip is per CLAIM, so adding a model to a "
                         "finished run otherwise no-ops on every claim and exits "
                         "looking like success; --force would 'fix' that by "
                         "discarding the other models' arms. This runs only the arms "
                         "(model, condition) arms that are missing and merges them in.")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    n_skipped = 0
    sections = resolve_sections(args.sections)
    label = args.card_label or (
        "" if args.sections.strip().lower() == "all" else args.sections.strip().lower())
    claims = load_claims()
    ids = (sorted(claims) if args.problems == "all"
           else [p.strip() for p in args.problems.split(",") if p.strip()])

    for pid in ids:
        if pid not in claims:
            print(f"[{pid}] not in the claims file — skipping")
            continue
        out_dir = os.path.join(args.out, f"problem_{pid}")
        out_path = os.path.join(out_dir, "scify_proposer.json")
        existing = {}
        if os.path.exists(out_path) and args.merge and not args.force:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            have = set(existing.get("arms") or {})
            # arm keys are "<model>+<condition>"; a model is done only when every
            # condition asked for is there, so new conditions can be added to a run
            # whose models already have other arms.
            want = [f"card:{label}" if n == "card" and label else n for n in names]
            done = {m for m in models if all(f"{m}+{c}" in have for c in want)}
            if set(models) <= done:
                print(f"[{pid}] all {len(models)} model(s) already present "
                      f"({len(have)} arms); nothing to merge")
                n_skipped += 1
                continue
            models_here = [m for m in models if m not in done]
            print(f"[{pid}] merging {models_here} into {len(have)} existing arm(s)")
        else:
            models_here = models
        if os.path.exists(out_path) and not args.force and not args.merge:
            # The skip is per CLAIM, not per arm, so adding a model to an existing
            # run skips everything and exits looking like success. Say which arms
            # are already there, and fail loudly if the whole run was a no-op.
            have = sorted(json.load(open(out_path, encoding="utf-8")).get("arms") or {})
            print(f"[{pid}] proposals exist ({len(have)} arms: "
                  f"{', '.join(have[:4])}{'…' if len(have) > 4 else ''}); --force to redo")
            n_skipped += 1
            continue

        problem = claims[pid]
        if existing.get("subclaims"):
            # Merged arms must see the SAME subclaims as the arms already there.
            # Decomposition is sampled, so re-running it hands the added model a
            # different question list than the models it is compared with — and the
            # rewrite below would then relabel the old arms with subclaims they never saw.
            subclaims = existing["subclaims"]
            print(f"\n=== problem {pid} · reusing {len(subclaims)} stored subclaims ===")
        else:
            print(f"\n=== problem {pid} · decomposing with {args.decomposer} ===")
            subclaims = scify_proposer.decompose(problem["claim"], model=args.decomposer)
            print(f"    {len(subclaims)} subclaims")

        card = load_json(args.cards, pid, "norm_card.json") if args.cards else {}
        bundle = load_json(args.cards, pid, "bundle.json") if args.cards else {}
        conditions = build_conditions(names, card, bundle, sections, label,
                                      claim=problem["claim"], cache_dir=args.cache_dir, prompt=args.prompt)
        if not conditions:
            print(f"[{pid}] no runnable condition — skipping")
            continue
        for c, ev in conditions.items():
            print(f"    condition {c:16s} evidence {sum(len(v) for v in ev.values()):6d} chars")

        print(f"    proposing: {len(models_here)} model(s) x {len(conditions)} condition(s), "
              f"budget {args.budget}")
        arms = propose_all(problem, models_here, subclaims, conditions,
                           workers=args.workers, budget=args.budget,
                           repeats=args.repeats, prompt=args.prompt)
        if existing:
            merged = dict(existing.get("arms") or {})
            merged.update(arms)
            arms = merged

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"problem_id": pid, "claim": problem["claim"],
                       "decomposer_model": args.decomposer, "subclaims": subclaims,
                       "proposer_prompt": args.prompt,
                       "conditions": sorted(conditions), "budget": args.budget,
                       "repeats": args.repeats,
                       "card_sections": list(sections) if sections else "all",
                       "card_format": card.get("format_version", ""),
                       "norm_card_used": bool(card),
                       # evaluate.py reads this stamp and hides the evidence from the
                       # judge; a contrast the judge can see is not a contrast.
                       "ablation": True,
                       "arms": arms}, f, indent=2)
        for m, a in sorted(arms.items()):
            print(f"      {m:26s} " + (a.get("error") or f"{len(a['experiments'])} experiments"))
        print(f"    -> {out_path}")


    if n_skipped == len(ids):
        raise SystemExit(
            f"nothing to do: all {n_skipped} claim(s) already have a proposals file. "
            f"--force overwrites them, which DISCARDS arms from other models in the "
            f"same file; to add a model to an existing run, merge into `arms` instead.")


if __name__ == "__main__":
    _main()
