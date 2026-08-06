"""Faithful, self-contained copy of SciFy's experiment Proposer for the
norm-card ablation.

We do NOT import anything from the dryrun codebase — the SciFy `SYSTEM_PROPOSER`
text is copied here verbatim (with the GPU/runtime resource-constraint block
REMOVED, per the experiment design: we want to test whether the norm card helps
design the *ideal* experiment set, unconstrained by an 8GB/30-min budget).

The ONLY difference between the two arms is `current_evidence`:
  - baseline: {}                      (claim + subclaims alone)
  - method:   {norm-card menu}        (claim + subclaims + the field's norm card)

Everything else — prompt, model (gpt-5.5), state formatting — is identical to
SciFy's real proposer (proposer.py / module_prompts/proposer.py in dryrun).
"""

import json
from . import llm

# --- SciFy's SYSTEM_PROPOSER, verbatim, minus the resource-constraint block --- #
SYSTEM_PROPOSER = (
    "Propose up to 3 experiments to assess the feasibility of the given claim and subclaims.\n"
    "If provided, each experiment must specifically address a particular subclaim or piece of evidence.\n"
    "When doing so, begin with 'To address subclaim X...' or 'With respect to evidence piece Y...'.\n"
    "After describing the experiment, briefly discuss conclusions one could draw from possible results.\n"
    "If the experiment could be made unnecessary by more evidence, briefly end by discussing how so.\n"
    "List experiments in order of priority; earlier experiments may inform later ones.\n"
    "Any datasets required should be publically available via HuggingFace, Kaggle, etc."
)

MENU_KEYS = ("datasets", "models", "metrics", "protocols")

# --- SciFy's DECOMPOSITION_PROMPT, verbatim (decom_prompts.py) --------------- #
# Turns a claim into independent factual sub-questions; SciFy runs it on
# gpt-5-mini @ temp 0. We feed the SAME subclaims to both arms, so the decomposer
# model can't bias the baseline-vs-method comparison.
DECOMPOSITION_PROMPT = """\
You are given a claim, your task is to decompose it into multiple independent and individual questions. DON'T generate any other text than the questions. You are given some examples below and the input claim at the end.

Claim: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Questions:
- Did Lord Steven Regal win the World Television Championship?
- Did The Nasty Boys win the World Tag Team Championship?

Claim: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Questions:
- When was the parkway opened?
- How long was the construction period for the parkway?
- How many years of community requests preceded the opening of the parkway?

Claim: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Questions:
- When did the company partner with AWS?
- What was the purpose of the partnership?

Claim: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2-0.
Questions:
- How many times had Sebastian Vettel won the Nations' Cup before?
- What title did Sebastian Vettel achieve for the first time?
- Whom did Sebastian Vettel defeat in the final?
- How many finals had Tom Kristensen reached?
- What was the final score between Sebastian Vettel and Tom Kristensen?

Claim: {claim}
Questions:
"""


def decompose(claim: str, model: str = "gpt-5.5") -> list:
    """Replicate SciFy's QuestionDecomposer: claim -> independent sub-questions.
    (SciFy's default decomposer model is gpt-5-mini; we standardize on gpt-5.5.
    Result is shared by both arms, so the choice does not bias the comparison.)"""
    text = llm.complete(DECOMPOSITION_PROMPT.format(claim=claim), model=model)
    lines = [ln.strip().lstrip("-").strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return list(dict.fromkeys(lines))  # dedup, order-preserving


def _state_handler(claim: str, subclaims: list, current_evidence: dict) -> str:
    """Byte-for-byte the same user-prompt shape as SciFy's proposer._state_handler."""
    user_prompt = "Claim:\n"
    user_prompt += claim
    user_prompt += "\n\n"
    user_prompt += "Subclaims:\n"
    user_prompt += str(subclaims)
    user_prompt += "\n\n"
    user_prompt += "Current Evidence:"
    user_prompt += str(current_evidence)
    return user_prompt


def norm_card_evidence(card_json: dict, n_per_key: int = 18) -> dict:
    """Format a norm-card set into a `current_evidence`-style dict, mirroring what
    the real recipe step feeds (top-n names per key per subfield, _menu_text-style)."""
    ev = {}
    for c in card_json.get("norm_cards", []):
        menu = c["menu"]
        ev[c["subfield"]] = {
            k: [it["name"] for it in menu.get(k, [])[:n_per_key]] for k in MENU_KEYS
        }
    return {"scientific_norm_card": ev}


def propose(claim: str, subclaims: list, current_evidence: dict,
            model: str = "gpt-5.5") -> list:
    """Run the proposer. Returns the list of proposed experiment strings."""
    system = SYSTEM_PROPOSER
    user = _state_handler(claim, subclaims, current_evidence)
    prompt = (
        system
        + "\n\n"
        + user
        + '\n\nReturn JSON of exactly the form {"experiments": ["...", "..."]} '
          "where each item is one full experiment description as prose."
    )
    out = _complete_json_strict(prompt, model=model)
    return out.get("experiments", [])


def _complete_json_strict(prompt: str, model: str) -> dict:
    """Force valid JSON via the API's json_object mode (mirrors SciFy's strict
    json_schema response_format) — the proposer's long prose otherwise breaks a
    naive json.loads on embedded quotes."""
    litellm = llm._litellm()
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1 if model.startswith("gpt-5") else 0.1,
        "response_format": {"type": "json_object"},
    }
    resp = litellm.completion(**kwargs)
    return json.loads(resp.choices[0].message.content)


def run_ablation(card_path: str, model: str = "gpt-5.5",
                 subclaims: list = None, claim: str = None,
                 problem_id: str = None) -> dict:
    """Baseline (no norm card) vs method (norm card in current_evidence).

    Mirrors SciFy's real flow: the claim is first decomposed into subclaims
    (identical for both arms), then the proposer runs with/without the norm card.
    Pass `subclaims` explicitly to skip decomposition (e.g. to reuse a fixed set).

    `claim` overrides the card's own claim field. That is what makes a FIELD-LEVEL
    card (one not anchored to a single claim) usable here: point it at any claim in
    its field. Reuse the subclaims from an existing run for that claim too, so the
    card stays the only variable across every arm being compared.
    """
    card = json.load(open(card_path))
    claim = claim or card["claim"]
    if subclaims is None:
        subclaims = decompose(claim, model=model)
    baseline = propose(claim, subclaims, {}, model=model)
    method = propose(claim, subclaims, norm_card_evidence(card), model=model)
    return {
        "problem_id": problem_id or card.get("problem_id"),
        "claim": claim,
        "model": model,
        "card_path": card_path,
        "card_source": card.get("provenance", {}).get("note", ""),
        "subclaims": subclaims,
        "baseline_experiments": baseline,
        "method_experiments": method,
    }


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, help="path to a norm_card.json")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--problem", default=None,
                    help="problem id from the sprint claims file; use its claim "
                         "instead of the card's own (for field-level cards)")
    ap.add_argument("--reuse_subclaims", default=None,
                    help="path to an existing scify_proposer.json whose subclaims to "
                         "reuse, so arms differ only by the card")
    a = ap.parse_args()

    claim, subclaims = None, None
    if a.problem:
        from .eval import load_claims
        claim = load_claims()[a.problem]["claim"]
    if a.reuse_subclaims:
        subclaims = json.load(open(a.reuse_subclaims))["subclaims"]
    res = run_ablation(a.card, model=a.model, claim=claim, subclaims=subclaims,
                       problem_id=a.problem)
    js = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(js)
        print(f"wrote {a.out}")
    else:
        print(js)
