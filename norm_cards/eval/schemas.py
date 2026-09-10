"""The judge's `submit_evaluation` schema, plus the validators either side of it:
the gate a reference must pass on the way in, and the evidence discipline a judgment
must pass on the way out.

Three ideas run through the design:

- **The two sides are the same kind of object.** A reference is a list of experiments
  and so is a proposal, which is what lets you swap the sides to test the judge itself
  (`evaluate.py --self-test`).

- **The model emits small judgments; code composes the numbers.** Nowhere does an
  agent hand back "recall: 0.75". It returns one status per reference experiment and
  scoring.py arithmetics them. A gpt-5.5 judge asked for holistic scores previously
  under-scored correct work.

- **Validation is a real gate, not prompt-level pleading.** A submission that fails it
  is handed back to the agent with the error, to fix and resubmit.
"""

from typing import Dict, List, Optional

COVER_STATUS = ("covered", "partial", "missing")
# Graded, because the binary version did not discriminate: every proposed experiment
# touches the reference somewhere, so a "does this correspond to reference work" flag
# came back true 27/27 and the metric built on it was a constant. What varies is not
# WHETHER an experiment relates to the decision but HOW MUCH it advances it.
SUFFICIENCY = ("sufficient", "sufficient_with_gaps", "insufficient")

_EVIDENCE = {"type": "object", "properties": {
    "source": {"type": "string"},
    "quote": {"type": "string", "description": "verbatim text — never paraphrase"},
    "bearing": {"type": "string", "description": "what this establishes here"}},
    "required": ["source", "quote", "bearing"]}


# --------------------------------------------------------------------------- #
# The reference — an INPUT, validated on the way in
# --------------------------------------------------------------------------- #
# There is no submit_* schema here. A reference is authored outside this harness
# (see reference.py) and arrives as a file; what follows only checks it is usable.
MIN_EXPERIMENT_CHARS = 120


# What an experiment is FOR, as opposed to what it does. Optional per reference,
# but the only way to see WHERE a proposer fails rather than how much: aggregated
# across the first three claims, apparatus and headline came out at 0.83 while
# confound-elimination sat at 0.43, which no per-claim recall number showed.
REFERENCE_ROLES = {
    "apparatus",   # build/validate the thing the headline test presupposes
    "headline",    # the direct test of the claim's assertion
    "mechanism",   # test that the asserted cause is the operative one
    "control",     # rule out that the result is an artifact of the intervention
    "confound",    # kill a rival explanation under which the result looks the same
    "external",    # show the finding generalises beyond the setup that produced it
}


def validate_reference(ref: Dict) -> Optional[str]:
    """Hard gate: is this reference usable to score against at all?"""
    return "; ".join(reference_errors(ref)) or None


def reference_errors(ref: Dict) -> List[str]:
    """The same gate as `validate_reference`, one message per entry."""
    errs: List[str] = []
    exps = ref.get("experiments")
    if not isinstance(exps, list) or not exps:
        errs.append("`experiments` must be a non-empty list of prose descriptions")
        return errs
    if len(exps) < 2:
        errs.append(f"only {len(exps)} experiment(s); a reference that a proposal can "
                    f"be scored against needs at least 2")
    for i, e in enumerate(exps):
        if not isinstance(e, str):
            errs.append(f"experiment {i}: must be a string, not {type(e).__name__} — "
                        f"the reference has to stay the same shape as a proposer run")
        elif len(e.strip()) < MIN_EXPERIMENT_CHARS:
            errs.append(f"experiment {i}: only {len(e.strip())} chars; say what is run, "
                        f"on what, what is measured, and what result decides it")
    if not (ref.get("claim") or "").strip():
        errs.append("no claim text")
    kind = ((ref.get("provenance") or {}).get("kind") or "").lower()
    if "not reviewed" in kind or "draft" in kind:
        errs.append("this is an unreviewed transcription draft. Read it against the "
                    "cited sections, then promote it with "
                    "`python -m norm_cards.eval.transcribe --problem <id> --accept "
                    "--author '<name>'` — a draft nobody has checked is not a standard")
    roles = ref.get("roles")
    if roles is not None:
        if not isinstance(roles, list) or len(roles) != len(exps):
            errs.append(f"`roles`, when present, must be one label per experiment "
                        f"({len(exps)} needed, got "
                        f"{len(roles) if isinstance(roles, list) else type(roles).__name__})")
        else:
            bad = [r for r in roles if r not in REFERENCE_ROLES]
            if bad:
                errs.append(f"unknown reference role(s) {sorted(set(bad))}; "
                            f"choose from {sorted(REFERENCE_ROLES)}")
    return errs


def reference_warnings(ref: Dict) -> List[str]:
    """Soft signals about how much weight this reference can bear."""
    warns: List[str] = []
    p = ref.get("provenance") or {}
    if not p.get("kind") or not p.get("author"):
        warns.append("provenance.kind/author not filled in — the judge is shown this "
                     "line, and an unattributed reference reads as a weak one")
    if not p.get("sources"):
        warns.append("provenance.sources is empty — nothing records where these "
                     "experiments came from")
    exps = [e for e in (ref.get("experiments") or []) if isinstance(e, str)]
    # A reference that dwarfs a proposal is the failure this format was built to fix:
    # the judge reads the extra specification as detail the proposal is missing.
    long = [i for i, e in enumerate(exps) if len(e) > 2000]
    if long:
        warns.append(f"experiment(s) {long} run past 2000 chars. A reference much "
                     f"richer than a proposer's output biases the comparison — keep "
                     f"each one to a paragraph")
    return warns


# --------------------------------------------------------------------------- #
# The judge's submission
# --------------------------------------------------------------------------- #
# One judgment per REFERENCE experiment, and nothing else. Earlier versions also
# collected a per-proposed-experiment audit — soundness, norm alignment, redundancy,
# resource grounding — feeding metrics that were not being read. Which proposed
# experiments did nothing is still recoverable in code: they are the ones that never
# appear in any `covered_by`.
_REF_COVERAGE = {"type": "object", "properties": {
    "ref_index": {"type": "integer", "description": "0-based index in the reference list"},
    # Written before the status, and instrument-free by construction. Naming the
    # property rather than the tool is what stops the judge scoring "you did not use
    # the R_k statistic" when any equivalent measure settles the same question.
    "adequacy": {"type": "string",
                 "description": "what property this reference experiment's instrument "
                                "or scope provides that makes it adequate, stated "
                                "WITHOUT naming the instrument, dataset, model or "
                                "library. Write this before choosing a status."},
    "status": {"type": "string", "enum": list(COVER_STATUS),
               "description": "covered = the proposed set would establish this by "
                              "whatever route; partial = it gets at the question but "
                              "the outcome stays ambiguous; missing = nothing in the "
                              "set bears on it"},
    "covered_by": {"type": "array", "items": {"type": "integer"},
                   "description": "indices of every PROPOSED experiment contributing — "
                                  "many-to-many, order irrelevant"},
    "rationale": {"type": "string",
                  "description": "what specifically does or does not carry it"},
    "reasoning_chain": {"type": "array", "items": {"type": "string"},
                        "description": "one inference per entry, ending in what the "
                                       "team would fail to learn. Required for partial "
                                       "and missing."},
    "evidence": {"type": "array", "items": _EVIDENCE,
                 "description": "quotes from any tool check that informed this"}},
    "required": ["ref_index", "adequacy", "status", "covered_by", "rationale"]}

SUBMIT_EVALUATION = {"type": "function", "function": {
    "name": "submit_evaluation",
    "description": "Submit your finished evaluation. Call this only after working "
                   "through every reference experiment.",
    "parameters": {"type": "object", "properties": {
        "reference_coverage": {"type": "array", "items": _REF_COVERAGE,
                               "description": "one entry per REFERENCE experiment, in "
                                              "order"},
        "decision_sufficiency": {"type": "string", "enum": list(SUFFICIENCY),
                                 "description": "if the proposed set were executed as "
                                                "written, would its outcomes settle the "
                                                "claim?"},
        "sufficiency_reasoning": {"type": "string"},
        "missing": {"type": "array", "items": {"type": "string"},
                    "description": "what the proposed set would still need"}},
        "required": ["reference_coverage", "decision_sufficiency",
                     "sufficiency_reasoning"]}}}


def validate_evaluation(ev: Dict, n_proposed: int, n_reference: int,
                        tools_available: bool = True) -> Optional[str]:
    """Structural gate, plus the one discipline that matters.

    Saying a reference experiment is not covered asserts something about the set in
    front of the judge, which no search can confirm — so it needs a reasoning chain,
    not a citation. Demanding quotes for absence would push the judge toward calling
    things covered to avoid the burden.
    """
    errs: List[str] = []

    # A language model can return a bare string where an object belongs; that must
    # surface as a validation error the agent can fix, never as an AttributeError.
    items = ev.get("reference_coverage")
    if not isinstance(items, list):
        return f"reference_coverage must be a list, got {type(items).__name__}"
    cov = [c for c in items if isinstance(c, dict)]
    if len(cov) != len(items):
        errs.append("reference_coverage must contain objects, not bare strings")

    seen = {c.get("ref_index") for c in cov}
    if len(cov) != n_reference or seen != set(range(n_reference)):
        errs.append(f"needs exactly one entry per reference experiment, indices "
                    f"0..{n_reference - 1}; got "
                    f"{sorted(x for x in seen if x is not None)}")

    for c in cov:
        i = c.get("ref_index")
        if c.get("status") not in COVER_STATUS:
            errs.append(f"ref {i}: status must be one of {COVER_STATUS}")
        chain = [x for x in (c.get("reasoning_chain") or []) if str(x).strip()]
        if c.get("status") in ("partial", "missing") and len(chain) < 2:
            errs.append(f"ref {i}: a '{c.get('status')}' judgment needs a "
                        f"reasoning_chain showing why the set does not settle it")
        if c.get("status") == "covered" and not (c.get("covered_by") or []):
            errs.append(f"ref {i}: marked covered but no proposed experiment listed "
                        f"in covered_by")
        if c.get("status") == "missing" and (c.get("covered_by") or []):
            errs.append(f"ref {i}: marked missing but lists contributing experiments — "
                        f"use 'partial' if something bears on it")
        for j in c.get("covered_by") or []:
            if not isinstance(j, int) or not 0 <= j < n_proposed:
                errs.append(f"ref {i}: covered_by {j!r} is not a proposed index")
        if len((c.get("rationale") or "").strip()) < 25:
            errs.append(f"ref {i}: rationale must say what does or does not carry it")
        if len((c.get("adequacy") or "").strip()) < 20:
            errs.append(f"ref {i}: state the adequacy property — what the reference's "
                        f"instrument provides — without naming the instrument")

    if ev.get("decision_sufficiency") not in SUFFICIENCY:
        errs.append(f"decision_sufficiency must be one of {SUFFICIENCY}")
    if len((ev.get("sufficiency_reasoning") or "").strip()) < 40:
        errs.append("sufficiency_reasoning must justify the sufficiency call")
    return "; ".join(errs) if errs else None
