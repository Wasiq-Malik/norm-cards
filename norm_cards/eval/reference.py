"""The reference recipe is an INPUT to this harness, not something it produces.

    python -m norm_cards.eval.reference --template 7   # blank skeleton to author
    python -m norm_cards.eval.reference --check        # validate + render every one

A reference recipe is the experiment chain a competent team would run to decide a
claim. Where it comes from is deliberately outside this codebase: a human writes
it, or it is transcribed from the experiment section of the paper the claim was
taken from, or a domain expert dictates it. What this module does is check that
whatever arrives is structurally sound and renders it for review.

Why it is not generated here: a reference produced by asking a strong LLM to design
experiments for the claim is not an independent standard. It is one more system's
output, and if that system were trustworthy enough to define correctness you would
ship it as the proposer instead of scoring against it. So the harness takes the
reference as given, and Stage B is built to treat it as fallible rough notes rather
than an oracle (see prompts.EVALUATION_SYSTEM).

Every reference carries a `provenance` block saying who authored it and how. That
line is shown to the judge, so a reference of unknown or weak origin is visibly
weak at the point of use.
"""

import argparse
import json
import os

from . import EVAL_ROOT, gt_dir, gt_path, load_claims, load_ground_truth
from . import schemas

NORM_KEYS = ("datasets", "models", "metrics", "protocols")


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
    if ref.get("generator"):                       # legacy LLM-generated references
        g = ref["generator"]
        return (f"llm_generated · {g.get('model')} @ {g.get('reasoning_effort')} "
                f"effort — machine-written, unverified by a human")
    return "provenance not stated — treat with corresponding suspicion"


def to_markdown(ref: dict) -> str:
    """Human-readable rendering — this is what you actually read before trusting a
    reference recipe, and what Stage B is shown."""
    L = [f"# Reference recipe — problem {ref['problem_id']}", "",
         f"**Claim.** {ref['claim']}", "",
         f"**Provenance.** {provenance_line(ref)}", ""]
    ca = ref.get("claim_analysis") or {}
    L += ["## What is asserted", ""]
    L += [f"- {a}" for a in ca.get("assertions") or []]
    if ca.get("ambiguities"):
        L += ["", "**Underspecified / reading adopted:**"]
        L += [f"- {a}" for a in ca["ambiguities"]]

    L += ["", "## Field norms (as established from sources)", ""]
    for key in NORM_KEYS:
        items = (ref.get("field_norms") or {}).get(key) or []
        if not items:
            continue
        L.append(f"**{key}**")
        for it in items:
            cites = "; ".join(f"{c.get('source', '')}" for c in it.get("citations") or [])
            L.append(f"- `{it.get('name', '')}` — {it.get('why_standard', '')}"
                     + (f"  \n  <sub>{cites}</sub>" if cites else ""))
        L.append("")

    L += ["## Recipe", ""]
    for s in ref.get("recipe") or []:
        dep = f" (after {', '.join(s['depends_on'])})" if s.get("depends_on") else ""
        L += [f"### {s.get('id')} — {s.get('role')}{dep}  ·  confidence: "
              f"{s.get('confidence', '?')}", "",
              s.get("description", ""), "",
              f"- **Pass:** {s.get('pass_criteria', '')}"]
        if s.get("fail_meaning"):
            L.append(f"- **Failure means:** {s['fail_meaning']}")
        if s.get("resources"):
            L.append(f"- **Resources:** {', '.join(s['resources'])}")
        L.append(f"- **Why:** {s.get('why_necessary', '')}")
        for c in s.get("evidence") or []:
            L.append(f"  - *{c.get('source', '')}*: \"{(c.get('quote') or '')[:300]}\"")
        for c in s.get("caveats") or []:
            L.append(f"- **Caveat:** {c}")
        L.append("")

    L += ["## Decision logic", "", ref.get("decision_logic", ""), ""]
    if ref.get("self_critique"):
        L += ["## Known weaknesses of this reference", ""]
        L += [f"- {c}" for c in ref["self_critique"]]
    return "\n".join(L)


TEMPLATE_HELP = (
    "Fill this in by hand (or transcribe it from the source paper's experiment "
    "section) and save it as results/eval_v2/ground_truth/problem_<id>/"
    "ground_truth.json. Roles: gate = a cheap prerequisite whose failure refutes the "
    "claim outright; apparatus = something the claim presupposes that must be built "
    "AND validated first; headline = the direct test enforcing every constraint the "
    "claim states; control = checks that the result is not an artifact. Aim for 4-6 "
    "steps, refute-first: cheapest potential refutation goes first, later steps build "
    "on earlier ones. pass_criteria must be a decisive threshold tied to the claim's "
    "own numbers — never 'record the value'. Then run "
    "`python -m norm_cards.eval.reference --check` and fix what it reports. "
    "See docs/reference_format.md for the field-by-field spec."
)


def template(problem: dict) -> dict:
    pid = str(problem["problem_id"])
    return {
        "_instructions": TEMPLATE_HELP,
        "type": "ground_truth", "format_version": "2.0",
        "problem_id": pid, "domain": problem.get("domain", ""),
        "claim": problem["claim"],
        "provenance": {
            "kind": "",            # human | paper_derived | expert_dictated | other
            "author": "",
            "date": "",
            "sources": [],         # papers/docs this recipe was built from
            "notes": "",
        },
        "claim_analysis": {
            "assertions": [""],
            "named_entities": {"models": [], "datasets": [], "metrics": [],
                               "thresholds": []},
            "claim_type": "empirical",
            "ambiguities": [],
        },
        "field_norms": {k: [{"name": "", "why_standard": "",
                             "citations": [{"source": "", "quote": ""}]}]
                        for k in NORM_KEYS},
        "recipe": [
            {"id": "G0", "role": "gate", "description": "", "pass_criteria": "",
             "fail_meaning": "", "resources": [], "depends_on": [],
             "why_necessary": "", "evidence": [{"source": "", "quote": ""}],
             "confidence": "high", "caveats": []},
            {"id": "H1", "role": "headline", "description": "", "pass_criteria": "",
             "fail_meaning": "", "resources": [], "depends_on": ["G0"],
             "why_necessary": "", "evidence": [{"source": "", "quote": ""}],
             "confidence": "high", "caveats": []},
        ],
        "decision_logic": "",
        "self_critique": [],
    }


def _check_one(pid: str) -> bool:
    ref = load_ground_truth(pid)
    errs = schemas.reference_errors(ref)
    warns = schemas.reference_warnings(ref)
    n = len(ref.get("recipe") or [])
    roles = ", ".join(s.get("role", "?") for s in ref.get("recipe") or [])
    print(f"\n[{pid}] {n} steps ({roles})")
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
