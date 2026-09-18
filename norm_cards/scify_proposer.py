"""Faithful, self-contained copy of SciFy's experiment Proposer for the
norm-card ablation.

Nothing is imported from the dryrun codebase — the prompts are copied here so a
run of this harness is reproducible from this repo alone. Resynced against
dryrun @400e677 (`module_prompts/proposer.yaml`, `proposer.py`, `decom.py`),
which resolved three drifts that had accumulated in the earlier hand-copy:

  * system and instance prompts are two messages rendered from Jinja templates,
    not one concatenated string;
  * `current_evidence` is `dict[str, str]` — a section name mapping to PROSE.
    The old copy built a nested dict and `str()`-ed it, so the proposer received
    a Python repr, `{'scientific_norm_card': {'Knowledge Editing': {'datasets':
    [...]`. That is the shape in which a norm card reads as a bag of nouns;
  * the decomposition prompt was missing two of its six exemplars.

The ONE deliberate divergence from upstream is the resource-constraint block
(CPU/GPU/runtime), removed per the experiment design: we are asking whether the
norm card helps design the *ideal* experiment set, not one fitted to an 8GB,
30-minute budget. `experiment_budget` is parameterized so the upstream cap of 3
can be varied as a diagnostic — see propose.py --budget — but 3 is the default
because 3 is what SciFy actually runs.
"""

import json

from jinja2 import Template

from . import llm

# --- dryrun/module_prompts/proposer.yaml : system_proposer -------------------
# Verbatim, less the trailing "Fit your experiments within these constraints"
# block and its CPU/GPU/runtime fields.
SYSTEM_PROPOSER_TEMPLATE = """\
Propose up to {{ budget }} experiments to assess the feasibility of the given claim and subclaims.
If provided, each experiment must specifically address a particular subclaim or piece of evidence.
When doing so, begin with 'To address subclaim X...' or 'With respect to evidence piece Y...'.
After describing the experiment, briefly discuss conclusions one could draw from possible results.
If the experiment could be made unnecessary by more evidence, briefly end by discussing how so.
List experiments in order of priority; earlier experiments may inform later ones.
Any datasets required should be publically available via HuggingFace, Kaggle, etc."""

# --- dryrun/module_prompts/proposer.yaml : instance_proposer -----------------
INSTANCE_PROPOSER_TEMPLATE = """\
Claim:
  {{ claim }}

Subclaims:
{% for subclaim in subclaims %}
  {{ subclaim }}
{% endfor %}

Current Evidence:
{% for evidence_key in current_evidence %}
  {{ evidence_key }}:
    {{ current_evidence[evidence_key] | indent(4) }}
{% endfor %}"""

DEFAULT_BUDGET = 3       # SciFy's own cap; vary only as a labelled diagnostic

MENU_KEYS = ("datasets", "models", "metrics", "protocols")

# --- dryrun/module_prompts/decom.py : DECOMPOSITION_PROMPT, verbatim --------
# Claim -> independent factual sub-questions. Upstream runs it on gpt-5-mini at
# temp 0; we standardize on gpt-5.5. The SAME subclaims go to every arm, so the
# decomposer cannot bias a card-vs-nocard comparison.
DECOMPOSITION_PROMPT = """
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

Claim: Touring began in Europe in April–June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July–August, and concluding in North America in October–November.
Questions:
- When did touring begin in Europe?
- Who was the opening act during the touring in Europe?
- Which months covered the Australia tour?
- Which months covered the New Zealand tour?
- Which months covered the Mexico tour?
- Which months covered the South America tour?
- Which months covered the North America tour?
- Where did the touring conclude?

Claim: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Questions:
- When did the company partner with AWS?
- What was the purpose of the partnership?

Claim: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Questions:
- Which country hosts the largest Yazidi community?
- How large is the Yazidi community in Germany?
- In which cities are the Yazidi community in Germany primarily located?

Claim: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2–0.
Questions:
- How many times had Sebastian Vettel won the Nations' Cup before?
- What title did Sebastian Vettel achieve for the first time?
- Whom did Sebastian Vettel defeat in the final?
- How many finals had Tom Kristensen reached?
- What was the final score between Sebastian Vettel and Tom Kristensen?

Claim: {claim}
Questions:
""".strip()


def decompose(claim: str, model: str = "gpt-5.5") -> list:
    """Replicate SciFy's QuestionDecomposer: claim -> independent sub-questions."""
    text = llm.complete(DECOMPOSITION_PROMPT.format(claim=claim), model=model)
    lines = [ln.strip().lstrip("-").strip() for ln in text.split("\n")]
    return list(dict.fromkeys([ln for ln in lines if ln]))  # dedup, order-preserving


# --------------------------------------------------------------------------- #
# evidence rendering
#
# `current_evidence` is `dict[str, str]` upstream: a section name mapping to
# prose the proposer reads. Both builders below honour that, so the norm-card arm
# and the retrieval arm differ in WHAT they say, never in how they are formatted
# — otherwise the ablation would be partly measuring presentation.
# --------------------------------------------------------------------------- #
RESOURCE_KEYS = ("datasets", "models", "metrics")
DESIGN_KEYS = ("protocols", "controls", "confounds")
# Design first, deliberately. The measured failure of the v0.2 card was that its
# resource lists (75-86% of it) crowded control conditions out of a 3-experiment
# budget. Naming a dataset tells a proposer where to run; naming a control tells
# it what to run, and only the latter is a thing it cannot already guess. So the
# proposer reads how the field argues before it reads what the field owns.
MENU_KEYS = DESIGN_KEYS + RESOURCE_KEYS

_SECTION_TITLE = {
    "datasets": "Datasets this subfield evaluates on",
    "models": "Models and methods this subfield runs",
    "metrics": "Metrics this subfield reports",
    "protocols": "How this subfield takes a measurement",
    "controls": "Control conditions this subfield expects alongside an intervention",
    "confounds": "Rival explanations a referee in this subfield will raise, and what "
                 "discriminates against each",
}


def _render_section(key: str, items: list) -> str:
    if key in RESOURCE_KEYS:
        return ", ".join(
            it["name"] + (f" ({it['role']})" if it.get("role") else "")
            + (f" ({it['direction']})" if it.get("direction") else "")
            for it in items)
    lines = []
    for it in items:
        tail = " ".join(x for x in (it.get("detail"), it.get("rules_out"),
                                    it.get("ruled_out_by")) if x)
        lines.append(f"- {it['name']}" + (f": {tail}" if tail else ""))
    return "\n".join(lines)


def norm_card_evidence(card_json: dict, n_per_key: int = 18,
                       sections: tuple = MENU_KEYS) -> dict:
    """Render a norm card as `current_evidence`.

    Reads the curated claim-level `card` (format 0.3). Older cards only have the
    per-subfield `norm_cards`, so those are curated on the fly rather than fed
    raw — a v0.2 card is 228-351 entries, three-quarters of them cited by one
    paper, and handing that over is the condition we already measured as
    displacing controls out of a 3-experiment budget.

    `sections` selects which parts of the card to show, which is how the
    keep-or-discard question gets answered by experiment instead of by opinion:
    run one arm per subset and read off which sections carry the effect."""
    card = card_json.get("card")
    if not card:
        from .curate import curate as _curate
        card, _ = _curate(card_json.get("norm_cards") or [])
    ev = {}
    for k in sections:
        items = (card.get(k) or [])[:n_per_key]
        if items:
            ev[_SECTION_TITLE[k]] = _render_section(k, items)
    return ev


def retrieval_evidence(bundle: dict, n_papers: int = 10, chars: int = 700) -> dict:
    """The simple-retrieval control arm: the top-ranked papers themselves.

    A norm card is a synthesis over retrieved papers, so "card beats no card" is
    the wrong bar — it does not separate the synthesis from the mere presence of
    relevant literature. This arm hands over what retrieval alone returns, titles
    and truncated abstracts, at a comparable context cost. If the card cannot beat
    it, the extraction pipeline is not earning its keep."""
    out = []
    for i, p in enumerate((bundle.get("papers") or [])[:n_papers]):
        abstract = (p.get("abstract") or "").strip().replace("\n", " ")
        out.append(f"[{i + 1}] {p.get('title', '').strip()}\n    {abstract[:chars]}")
    return {"Retrieved papers on this subfield": "\n".join(out)} if out else {}


# --------------------------------------------------------------------------- #
# full-text retrieval arm (the honest control)
#
# The abstract arm below is a weak baseline, and knowingly so: the norm-card
# pipeline's own design notes record that abstracts named ZERO datasets, which is
# why extraction reads full text. Beating ten truncated abstracts therefore does
# not show that synthesis is worth anything — it shows that abstracts are thin.
#
# This arm removes that excuse. It reads the SAME papers the card was built from,
# at full text, retrieves the passages most relevant to the claim, and is trimmed
# to the SAME character budget the card occupies. Information access and context
# cost are held constant; the only thing that varies is whether that content
# arrives as raw passages or as a structured card. That is the comparison that
# actually tests the extraction pipeline.
# --------------------------------------------------------------------------- #
_WORD = None


def _tok(text: str) -> list:
    global _WORD
    if _WORD is None:
        import re as _re
        _WORD = _re.compile(r"[a-z0-9][a-z0-9\-]{2,}")
    return _WORD.findall((text or "").lower())


def _bm25_rank(query: str, docs: list, k1: float = 1.5, b: float = 0.75) -> list:
    """Plain BM25, no dependency. Returns doc indices, best first.

    Written out rather than pulled in because the environment has neither bm25s
    nor numpy, and a retrieval BASELINE that needs a new dependency to run is a
    baseline nobody re-runs."""
    import math
    from collections import Counter
    toks = [_tok(d) for d in docs]
    n = len(toks) or 1
    avg = sum(len(t) for t in toks) / n
    df = Counter()
    for t in toks:
        df.update(set(t))
    q = [w for w in set(_tok(query)) if df.get(w)]
    idf = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in q}
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        ln = len(t) or 1
        s = sum(idf[w] * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b * ln / avg))
                for w in q if tf[w])
        scores.append((s, i))
    scores.sort(reverse=True)
    return [i for s, i in scores if s > 0]


def _chunks(text: str, size: int = 1100, overlap: int = 150) -> list:
    """Passages on paragraph boundaries where possible, else a sliding window."""
    text = (text or "").strip()
    out, buf = [], ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                out.append(buf)
            while len(para) > size:
                out.append(para[:size])
                para = para[size - overlap:]
            buf = para
    if buf:
        out.append(buf)
    return [c for c in out if len(c) > 200]


def fulltext_retrieval_evidence(bundle: dict, claim: str, cache_dir: str = ".pdfcache",
                                budget: int = 9500, n_papers: int = 30,
                                per_paper: int = 3) -> dict:
    """Top BM25 passages from the bundle's papers, trimmed to `budget` chars.

    `per_paper` caps how many passages any one paper contributes, so a single
    long, on-topic paper cannot crowd out the rest of the field — the card is a
    view over many papers and the control should be too."""
    from . import fulltext
    passages, seen = [], 0
    for p in (bundle.get("papers") or [])[:n_papers]:
        text, _ = fulltext.fetch_fulltext(p, cache_dir)
        if not text:
            continue
        seen += 1
        for c in _chunks(text):
            passages.append((p.get("title", ""), c))
    if not passages:
        return {}
    order = _bm25_rank(claim, [c for _, c in passages])
    used, per, out = 0, {}, []
    for i in order:
        title, c = passages[i]
        if per.get(title, 0) >= per_paper:
            continue
        if used + len(c) > budget:
            continue
        per[title] = per.get(title, 0) + 1
        out.append(f"[from: {title}]\n{c}")
        used += len(c)
        if used >= budget * 0.97:
            break
    return {"Passages retrieved from the field's papers": "\n\n".join(out)} if out else {}


# --------------------------------------------------------------------------- #
# the proposer itself
# --------------------------------------------------------------------------- #
def render_prompts(claim: str, subclaims: list, current_evidence: dict,
                   budget: int = DEFAULT_BUDGET) -> tuple:
    """(system, instance), rendered exactly as dryrun/proposer.py renders them."""
    system = Template(SYSTEM_PROPOSER_TEMPLATE, trim_blocks=True,
                      lstrip_blocks=True).render(budget=budget).strip()
    instance = Template(INSTANCE_PROPOSER_TEMPLATE, trim_blocks=True,
                        lstrip_blocks=True).render(
        claim=claim, subclaims=subclaims,
        current_evidence=current_evidence or {}).strip()
    return system, instance


def propose(claim: str, subclaims: list, current_evidence: dict,
            model: str = "gpt-5.5", budget: int = DEFAULT_BUDGET) -> list:
    """Run the proposer. Returns the list of proposed experiment strings."""
    system, instance = render_prompts(claim, subclaims, current_evidence, budget)
    out = _complete_json_strict(system, instance, model=model)
    return out.get("experiments", [])


def _complete_json_strict(system: str, instance: str, model: str) -> dict:
    """Two messages, mirroring Query(system_prompt=..., instance_prompt=...), with
    JSON mode standing in for upstream's ProposerResponse structured output — the
    proposer's long prose otherwise breaks a naive json.loads on embedded quotes."""
    litellm = llm._litellm()
    resp = litellm.completion(
        timeout=llm.TIMEOUT,
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": instance + "\n\nReturn JSON of exactly the "
             'form {"experiments": ["...", "..."]} where each item is one full '
             "experiment description as prose."},
        ],
        temperature=1 if llm.wants_default_temperature(model) else 0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def run_ablation(card_path: str, model: str = "gpt-5.5", subclaims: list = None,
                 budget: int = DEFAULT_BUDGET) -> dict:
    """Baseline (no norm card) vs method (norm card in current_evidence)."""
    card = json.load(open(card_path, encoding="utf-8"))
    claim = card["claim"]
    if subclaims is None:
        subclaims = decompose(claim, model=model)
    return {
        "problem_id": card.get("problem_id"), "claim": claim, "model": model,
        "subclaims": subclaims, "budget": budget,
        "baseline_experiments": propose(claim, subclaims, {}, model, budget),
        "method_experiments": propose(claim, subclaims, norm_card_evidence(card),
                                      model, budget),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, help="path to a norm_card.json")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    js = json.dumps(run_ablation(a.card, model=a.model, budget=a.budget), indent=2)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(js)
        print(f"wrote {a.out}")
    else:
        print(js)
