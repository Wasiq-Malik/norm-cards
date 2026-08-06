"""The judge's `submit_evaluation` schema, plus the validators either side of it:
the gate a reference recipe must pass on the way in, and the evidence discipline a
judgment must pass on the way out.

Two ideas run through the design:

- **The model emits small judgments; code composes the numbers.** Nowhere does an
  agent hand back "coverage: 0.75". It returns per-experiment verdicts and
  per-role coverage statuses, and scoring.py arithmetics them. A gpt-5.5 judge
  asked for holistic scores previously under-scored correct recipes; small
  evidenced decisions are much harder to get diffusely wrong.

- **Validation is where the evidence discipline is actually enforced.** A verdict
  that faults an experiment (IRRELEVANT/MIXED) is rejected unless it carries
  quoted evidence *and* a reasoning chain connecting that quote to the
  conclusion. The agent is handed the error and resubmits, so this is a real
  gate rather than prompt-level pleading.
"""

from typing import Dict, List, Optional

ROLES = ("gate", "apparatus", "headline", "control")
VERDICTS = ("ACCURATE", "MIXED", "IRRELEVANT")
TRIAGE = ("MATCH", "PARTIAL", "NOVEL", "CONTRADICTS")
COVERAGE_STATUS = ("covered", "partial", "missing", "not_required")
SUFFICIENCY = ("sufficient", "sufficient_with_gaps", "insufficient")
TIERS = ("T1", "T2", "T3", "T0")

_EVIDENCE = {"type": "object", "properties": {
    "source": {"type": "string"},
    "quote": {"type": "string", "description": "verbatim text — never paraphrase"},
    "bearing": {"type": "string", "description": "what this establishes here"}},
    "required": ["source", "quote", "bearing"]}


# --------------------------------------------------------------------------- #
# The reference recipe — an INPUT, validated on the way in
# --------------------------------------------------------------------------- #
# There is no submit_* tool schema here. The reference recipe is authored outside
# this harness (see reference.py) and arrives as a file; what follows is the gate
# it has to pass before anything is scored against it.
def validate_reference(ref: Dict) -> Optional[str]:
    """Hard gate: is this reference recipe usable to score against at all?

    Only structural defects live here — things that would make a comparison
    meaningless (no step that tests the claim, a pass criterion you cannot pass or
    fail, a dependency on a step that does not exist). Judgments about how well
    researched the reference is belong in `reference_warnings`, because a
    hand-written reference and a paper-transcribed one carry their authority
    differently and neither should be blocked for the shape of its bibliography.
    """
    return "; ".join(reference_errors(ref)) or None


def reference_errors(ref: Dict) -> List[str]:
    """The same gate as `validate_reference`, one message per entry."""
    errs: List[str] = []
    recipe = ref.get("recipe") or []
    if not 3 <= len(recipe) <= 8:
        errs.append(f"recipe has {len(recipe)} steps, aim for 4-6 (3-8 allowed)")

    roles = [s.get("role") for s in recipe]
    if "headline" not in roles:
        errs.append("no 'headline' step — the recipe never directly tests the claim")
    if "gate" not in roles:
        errs.append("no 'gate' step — the chain should start with the cheapest check "
                    "that could refute the claim outright (if none exists, say so in "
                    "self_critique and make the first step a gate anyway)")

    ids = [s.get("id") for s in recipe]
    for i, s in enumerate(recipe):
        sid = s.get("id") or f"#{i}"
        if not s.get("id"):
            errs.append(f"step #{i}: needs an id (e.g. G0, A1, H2, C3)")
        if s.get("role") not in ROLES:
            errs.append(f"step {sid}: role must be one of {ROLES}")
        if len((s.get("pass_criteria") or "").strip()) < 15:
            errs.append(f"step {sid}: pass_criteria must be a concrete, decisive "
                        f"threshold, not 'record the value'")
        if len((s.get("why_necessary") or "").strip()) < 40:
            errs.append(f"step {sid}: why_necessary must actually justify the step")
        for d in s.get("depends_on") or []:
            if d not in ids:
                errs.append(f"step {sid}: depends_on {d!r} is not a step id")

    if len((ref.get("decision_logic") or "").strip()) < 40:
        errs.append("decision_logic must explain how step outcomes decide the claim")
    if not [a for a in (ref.get("claim_analysis") or {}).get("assertions") or []
            if str(a).strip()]:
        errs.append("claim_analysis.assertions is empty — state what the claim asserts")
    return errs


def reference_warnings(ref: Dict) -> List[str]:
    """Soft signals about how much weight this reference can bear.

    None of these block an evaluation. They tell you how far to trust the numbers
    that come out of one, which is the honest place for the question now that the
    reference is a human input rather than a machine artifact this harness made.
    """
    warns: List[str] = []
    p = ref.get("provenance") or {}
    if not p and ref.get("generator"):
        warns.append("no provenance block; this reference was machine-generated by an "
                     "earlier version of the harness and no human has signed off on it")
    elif not p.get("kind") or not p.get("author"):
        warns.append("provenance.kind/author not filled in — the judge is shown this "
                     "line, and an unattributed reference reads as a weak one")

    norms = ref.get("field_norms") or {}
    n_cited = sum(1 for k in ("datasets", "models", "metrics", "protocols")
                  for it in (norms.get(k) or [])
                  if any((c.get("quote") or "").strip()
                         for c in (it.get("citations") or [])))
    if n_cited < 3:
        warns.append(f"only {n_cited} field_norms entries carry a quoted citation; "
                     f"steps asserted without a source are the ones the judge will "
                     f"most easily overturn")

    uncited = [s.get("id") for s in ref.get("recipe") or []
               if not (s.get("evidence") or [])
               and len(s.get("why_necessary") or "") < 120]
    if uncited:
        warns.append(f"steps {', '.join(str(u) for u in uncited)} have neither a "
                     f"citation nor a spelled-out rationale")
    if not (ref.get("self_critique") or []):
        warns.append("self_critique is empty — record where this reference is weak, "
                     "so a divergent proposal is not wrongly penalized there")
    return warns


# --------------------------------------------------------------------------- #
# The judge's submission
# --------------------------------------------------------------------------- #
_ROLE_COVERAGE = {"type": "object", "properties": {
    "status": {"type": "string", "enum": list(COVERAGE_STATUS)},
    "by": {"type": "array", "items": {"type": "integer"},
           "description": "indices of the experiments that fill this role"},
    "note": {"type": "string"}},
    "required": ["status", "note"]}

SUBMIT_EVALUATION = {"type": "function", "function": {
    "name": "submit_evaluation",
    "description": "Submit your finished evaluation of this experiment set. Call this "
                   "only after running the full protocol on every experiment.",
    "parameters": {"type": "object", "properties": {
        "experiments": {"type": "array", "items": {"type": "object", "properties": {
            "index": {"type": "integer", "description": "0-based index in the set given"},
            "normalized": {"type": "object", "properties": {
                "target_subclaim": {"type": "string"},
                "method": {"type": "string"},
                "resources": {"type": "array", "items": {"type": "string"}},
                "measurement": {"type": "string"},
                "decision_rule": {"type": "string",
                                  "description": "what result would verify or refute, "
                                                 "or 'none stated'"}}},
            "gt_triage": {"type": "string", "enum": list(TRIAGE),
                          "description": "relation to the reference recipe"},
            "gt_steps_covered": {"type": "array", "items": {"type": "string"}},
            "verdict": {"type": "string", "enum": list(VERDICTS),
                        "description": "ACCURATE = conveys the correct thing; MIXED = "
                                       "right idea, under-specified or missing a minor "
                                       "element; IRRELEVANT = would not yield useful "
                                       "information about the claim"},
            "novel": {"type": "boolean",
                      "description": "sound but absent from the reference recipe"},
            "redundant_with": {"type": "array", "items": {"type": "integer"},
                               "description": "indices this duplicates, if any"},
            "reasoning_chain": {"type": "array", "items": {"type": "string"},
                                "description": "the argument, step by step, from your "
                                               "quoted facts to this verdict. Each entry "
                                               "is one inference. Required whenever you "
                                               "fault an experiment."},
            "evidence": {"type": "array", "items": _EVIDENCE},
            "evidence_tier": {"type": "string", "enum": list(TIERS),
                              "description": "T1 peer-reviewed paper quote; T2 official "
                                             "docs/leaderboard/dataset card; T3 claim "
                                             "text or reference recipe only; T0 none"},
            "resource_audit": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"},
                "exists": {"type": "boolean"},
                "publicly_available": {"type": "boolean"},
                "appropriate": {"type": "boolean",
                                "description": "fit for the use this experiment makes"},
                "note": {"type": "string"}},
                "required": ["name", "exists", "note"]}},
            "gt_defects": {"type": "array", "items": {"type": "object", "properties": {
                "gt_step": {"type": "string", "description": "step id, or 'missing'"},
                "defect": {"type": "string"},
                "evidence": {"type": "array", "items": _EVIDENCE}},
                "required": ["gt_step", "defect"]}}},
            "required": ["index", "verdict", "gt_triage", "reasoning_chain",
                         "evidence_tier"]}},

        "role_coverage": {"type": "object",
                          "description": "does the SET, as a whole, fill each role of "
                                         "the decision structure? Judge functionally, "
                                         "not by matching reference step ids.",
                          "properties": {"gate": _ROLE_COVERAGE,
                                         "apparatus": _ROLE_COVERAGE,
                                         "headline": _ROLE_COVERAGE,
                                         "control": _ROLE_COVERAGE},
                          "required": ["gate", "apparatus", "headline", "control"]},
        "decision_sufficiency": {"type": "string", "enum": list(SUFFICIENCY),
                                 "description": "if this set were executed as written, "
                                                "would its outcomes verify or refute "
                                                "the claim?"},
        "sufficiency_reasoning": {"type": "string"},
        "missing": {"type": "array", "items": {"type": "string"},
                    "description": "what the set would still need"},
        "set_gt_defects": {"type": "array", "items": {"type": "object", "properties": {
            "gt_step": {"type": "string"},
            "defect": {"type": "string"},
            "evidence": {"type": "array", "items": _EVIDENCE}},
            "required": ["gt_step", "defect"]}}},
        "required": ["experiments", "role_coverage", "decision_sufficiency",
                     "sufficiency_reasoning"]}}}


def validate_evaluation(ev: Dict, n_experiments: int) -> Optional[str]:
    """Structural gate + the evidence discipline.

    The load-bearing rule: any verdict that faults an experiment must carry quoted
    evidence AND a reasoning chain. Without this the judge can quietly fall back on
    "the reference recipe disagrees", which is exactly the failure mode that made
    the earlier gpt-5.5 judge unusable.
    """
    errs: List[str] = []
    exps = ev.get("experiments") or []
    seen = {e.get("index") for e in exps}
    if len(exps) != n_experiments or seen != set(range(n_experiments)):
        errs.append(f"must return exactly one entry per experiment, indices "
                    f"0..{n_experiments - 1}; got {sorted(x for x in seen if x is not None)}")

    for e in exps:
        i = e.get("index")
        if e.get("verdict") not in VERDICTS:
            errs.append(f"exp {i}: verdict must be one of {VERDICTS}")
        if e.get("gt_triage") not in TRIAGE:
            errs.append(f"exp {i}: gt_triage must be one of {TRIAGE}")

        chain = [c for c in (e.get("reasoning_chain") or []) if str(c).strip()]
        evid = [x for x in (e.get("evidence") or []) if (x.get("quote") or "").strip()]
        tier = e.get("evidence_tier")

        if e.get("verdict") in ("IRRELEVANT", "MIXED"):
            if not evid:
                errs.append(f"exp {i}: a {e.get('verdict')} verdict needs at least one "
                            f"quoted piece of evidence — the reference recipe alone is "
                            f"not grounds to fault an experiment")
            if len(chain) < 2:
                errs.append(f"exp {i}: a {e.get('verdict')} verdict needs a "
                            f"reasoning_chain showing how you get from your quoted "
                            f"facts to the fault you are alleging")
            if tier == "T0":
                errs.append(f"exp {i}: evidence_tier T0 cannot support a "
                            f"{e.get('verdict')} verdict")
        if e.get("novel") and not evid:
            errs.append(f"exp {i}: marked novel — quote the evidence that this design "
                        f"is sound and advances the decision")
        if tier not in TIERS:
            errs.append(f"exp {i}: evidence_tier must be one of {TIERS}")
        if evid and not any((x.get("source") or "").strip() for x in evid):
            errs.append(f"exp {i}: every quote needs an attributable source")

    rc = ev.get("role_coverage") or {}
    for role in ROLES:
        st = (rc.get(role) or {}).get("status")
        if st not in COVERAGE_STATUS:
            errs.append(f"role_coverage.{role}.status must be one of {COVERAGE_STATUS}")
    if ev.get("decision_sufficiency") not in SUFFICIENCY:
        errs.append(f"decision_sufficiency must be one of {SUFFICIENCY}")
    if len((ev.get("sufficiency_reasoning") or "").strip()) < 40:
        errs.append("sufficiency_reasoning must justify the sufficiency call")
    return "; ".join(errs) if errs else None
