"""The reference recipe is an INPUT to this harness, not something it produces.

    python -m norm_cards.eval.reference --template 7   # blank skeleton to author
    python -m norm_cards.eval.reference --check        # validate + render every one

A reference is **a list of experiments** — the ones a competent team actually ran to
decide the claim, usually transcribed from the experiment section of the paper the
claim came from. Nothing else.

That shape is deliberate and load-bearing: it is the *same type* as a proposer run's
`experiments`, a list of prose descriptions. Because the two sides are the same kind
of object you can swap them, which is what makes the judge testable — feed the
reference in as though it were a proposal and recall must come back at ceiling; if it
does not, the judge is broken and no number it produces means anything.

An earlier version made the reference a structured decision graph with roles,
dependencies, pass criteria, evidence and self-critique. It rendered to 130 lines
against a proposal's 5, and the judge duly marked every requirement `partial` — you
cannot compare a specification against a paragraph and expect the paragraph to look
complete. It also could not be swapped, so the judge could not be validated at all.

Why it is not generated here: a reference produced by asking a strong LLM to design
experiments for the claim is not an independent standard. It is one more system's
output, and if that system were trustworthy enough to define correctness you would
ship it as the proposer instead of scoring against it.

Every reference carries a `provenance` block saying who authored it and how. That line
is shown to the judge, so a reference of unknown or weak origin is visibly weak at the
point of use.
"""

import argparse
import json
import os

from . import EVAL_ROOT, gt_dir, gt_path, load_claims, load_ground_truth
from . import schemas


def provenance_line(ref: dict) -> str:
    """One line describing where this reference came from. Shown to the judge."""
    p = {k: v for k, v in (ref.get("provenance") or {}).items() if v}
    if p:
        bits = [p.get("kind") or "origin unstated"]
        if p.get("author"):
            bits.append(f"authored by {p['author']}")
        if p.get("date"):
            bits.append(p["date"])
        if p.get("sources"):
            bits.append("derived from " + "; ".join(p["sources"][:3]))
        return " · ".join(b for b in bits if b)
    return "provenance not stated — treat with corresponding suspicion"


def experiments(ref: dict) -> list:
    """The reference experiment list. Same type as a proposer arm's experiments."""
    return list(ref.get("experiments") or [])


def experiments_text(exps: list, label: str = "EXPERIMENT") -> str:
    """Render a list of experiments. Used for BOTH sides, so neither is presented
    more richly than the other — the formatting asymmetry is what broke the last
    version of this harness."""
    return "\n\n".join(f"--- {label} {i} ---\n{e}" for i, e in enumerate(exps))


def to_markdown(ref: dict) -> str:
    """Human-readable rendering — what you read before trusting a reference."""
    L = [f"# Reference experiments — problem {ref['problem_id']}", "",
         f"**Claim.** {ref['claim']}", "",
         f"**Provenance.** {provenance_line(ref)}", ""]
    notes = (ref.get("provenance") or {}).get("notes")
    if notes:
        L += [f"*{notes}*", ""]
    for i, e in enumerate(experiments(ref)):
        L += [f"### Experiment {i}", "", e, ""]
    return "\n".join(L)


TEMPLATE_HELP = (
    "Fill in `experiments` by hand, or transcribe them from the source paper's "
    "experiment section, and save as results/eval_v2/ground_truth/problem_<id>/"
    "ground_truth.json. Write each one as a single prose paragraph at the same "
    "granularity a proposer would: what is run, on what data and model, what is "
    "measured, and what result would decide the question. Include the concrete "
    "threshold or reported number so the decision rule is unambiguous. Do NOT write "
    "a structured specification — the reference has to stay the same kind of object "
    "as the pipeline's output, or the two cannot be compared or swapped. Then run "
    "`python -m norm_cards.eval.reference --check`. See docs/reference_format.md."
)


def template(problem: dict) -> dict:
    return {
        "_instructions": TEMPLATE_HELP,
        "type": "ground_truth", "format_version": "3.0",
        "problem_id": str(problem["problem_id"]),
        "domain": problem.get("domain", ""),
        "claim": problem["claim"],
        "provenance": {
            "kind": "",            # human | paper_derived | expert_dictated | other
            "author": "",
            "date": "",
            "sources": [],         # papers/docs this was built from
            "notes": "",
        },
        "experiments": ["", ""],
    }


def _check_one(pid: str) -> bool:
    ref = load_ground_truth(pid)
    errs = schemas.reference_errors(ref)
    warns = schemas.reference_warnings(ref)
    exps = experiments(ref)
    lens = [len(e) for e in exps] or [0]
    print(f"\n[{pid}] {len(exps)} experiments, {min(lens)}-{max(lens)} chars each")
    print(f"      provenance: {provenance_line(ref)}")
    for w in warns:
        print(f"      warn: {w}")
    if errs:
        for e in errs:
            print(f"      ERROR: {e}")
        return False
    md_path = os.path.join(gt_dir(pid), "ground_truth.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(ref))
    print(f"      ok -> {md_path}")
    return True


def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", metavar="ID",
                    help="write a blank reference skeleton for this problem id")
    ap.add_argument("--check", action="store_true",
                    help="validate existing references and re-render their markdown")
    ap.add_argument("--problems", default="all",
                    help="comma-separated ids to check (default: every one present)")
    args = ap.parse_args()

    claims = load_claims()

    if args.template:
        pid = args.template.strip()
        if pid not in claims:
            raise SystemExit(f"problem {pid} is not in the claims file")
        os.makedirs(gt_dir(pid), exist_ok=True)
        path = gt_path(pid)
        if os.path.exists(path):
            raise SystemExit(f"{path} already exists — move it aside first")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template(claims[pid]), f, indent=2)
        print(f"wrote {path}\n\n{TEMPLATE_HELP}")
        return

    root = os.path.join(EVAL_ROOT, "ground_truth")
    if args.problems == "all":
        ids = sorted(d.replace("problem_", "") for d in os.listdir(root)
                     if d.startswith("problem_")) if os.path.isdir(root) else []
    else:
        ids = [p.strip() for p in args.problems.split(",") if p.strip()]
    if not ids:
        print(f"no references under {root} — start one with "
              f"`python -m norm_cards.eval.reference --template <id>`")
        return

    bad = [pid for pid in ids if not _check_one(pid)]
    print(f"\n{len(ids) - len(bad)}/{len(ids)} references valid"
          + (f"; fix {', '.join(bad)} before evaluating against them" if bad else ""))


if __name__ == "__main__":
    _main()
