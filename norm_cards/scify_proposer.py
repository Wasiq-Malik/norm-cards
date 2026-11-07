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
30-minute budget. `experiment_budget` is parameterized: the upstream cap of 3 is
DEFAULT_BUDGET here, for rendering SciFy as it runs; our runs use RUN_BUDGET (5), see
propose.py --budget.
"""

import json
import os
import time

from jinja2 import Template

from . import config, llm

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

# "fieldnorms" (from dataset50's second run, 2026-10-30). SciFy's "Current Evidence" is the
# evidence its agent has gathered ON THE CLAIM (literature, experiment reports), and its prompt
# asks each experiment to address "a subclaim or piece of evidence". Passing a norm card or field
# passages there labelled them as evidence: gpt-6.1-sol then spent ~1 experiment per proposal
# "with respect to evidence piece" on card items and its subclaim coverage fell 79% -> 62%.
# So Current Evidence stays empty (we gather no claim evidence), and the card (its titled
# sections, as norm_card_text renders them) or the rag passages arrive under their own FIELD NORMS key, with one system line
# saying what they are. The system prompt is the same in every condition. "scify" is the
# upstream prompt verbatim, as every run through dataset50's first run used it. The default is
# "fieldnorms2", below.
PROPOSER_PROMPT = "fieldnorms2"
FIELD_NORMS_KEY = "__field_norms__"          # current_evidence key carrying the field-norms text

SYSTEM_PROPOSER_TEMPLATE_FN = SYSTEM_PROPOSER_TEMPLATE.replace(
    "begin with 'To address subclaim X...' or 'With respect to evidence piece Y...'.\n",
    "begin with 'To address subclaim X...' or 'With respect to evidence piece Y...'.\n"
    "If provided, use the Field Norms, which describe how this claim's scientific field designs and "
    "evaluates experiments, to help ground your experiments in the existing literature.\n")
assert SYSTEM_PROPOSER_TEMPLATE_FN != SYSTEM_PROPOSER_TEMPLATE

INSTANCE_PROPOSER_TEMPLATE_FN = INSTANCE_PROPOSER_TEMPLATE + """
{% if field_norms %}

Field Norms:
  {{ field_norms | indent(2) }}
{% endif %}"""

# "fieldnorms2" (2026-11-05; the default since 2026-11-06). Some proposers still wrote experiments
# about the norms rather than the claim. Here an empty field says "None" instead of leaving a bare header,
# the rag passages get their own Relevant Literature key instead of sharing Field Norms with the
# card, and each arm's system prompt explains only the key that arm carries (no card: none).
LITERATURE_KEY = "__relevant_literature__"   # current_evidence key carrying the rag passages

SYSTEM_LINE_FIELD_NORMS = (
    "If provided, the Field Norms are a norm card for the scientific field(s) this claim belongs to. "
    "A norm card describes how that field designs and evaluates experiments: the control conditions "
    "and rival explanations referees expect, what counts as evidence that a difference is real, what "
    "a fair baseline comparison holds equal, and the metrics and datasets the field uses. Use them to "
    "design better experiments for the claim; they are not evidence about the claim itself.")
SYSTEM_LINE_LITERATURE = (
    "If provided, the Relevant Literature is a set of passages from published papers in the claim's "
    "field. Use it to design better experiments grounded in existing work (its methods, baselines, "
    "datasets and metrics); it is background, not evidence about the claim itself.")
_SYSTEM_ANCHOR = "begin with 'To address subclaim X...' or 'With respect to evidence piece Y...'.\n"
assert _SYSTEM_ANCHOR in SYSTEM_PROPOSER_TEMPLATE

INSTANCE_PROPOSER_TEMPLATE_FN2 = """\
Claim:
  {{ claim }}

{% if subclaims %}
Subclaims:
{% for subclaim in subclaims %}
  {{ subclaim }}
{% endfor %}
{% else %}
Subclaims: None
{% endif %}

{% if current_evidence %}
Current Evidence:
{% for evidence_key in current_evidence %}
  {{ evidence_key }}:
    {{ current_evidence[evidence_key] | indent(4) }}
{% endfor %}
{% else %}
Current Evidence: None
{% endif %}
{% if field_norms %}

Field Norms:
  {{ field_norms | indent(2) }}
{% endif %}
{% if literature %}

Relevant Literature:
  {{ literature | indent(2) }}
{% endif %}"""


def system_template_fn2(field_norms: bool, literature: bool) -> str:
    """The base system prompt plus a line for each key this arm actually carries."""
    lines = ([SYSTEM_LINE_FIELD_NORMS] if field_norms else []) + ([SYSTEM_LINE_LITERATURE] if literature else [])
    return SYSTEM_PROPOSER_TEMPLATE.replace(_SYSTEM_ANCHOR, _SYSTEM_ANCHOR + "".join(l + "\n" for l in lines))


DEFAULT_BUDGET = 3       # SciFy's own cap, for rendering its prompt as upstream runs it
RUN_BUDGET = 5           # our runs: on dataset50, 5 lost no recall against 9 and the card's gain held

MENU_KEYS = ("datasets", "models", "metrics", "protocols")

# --- dryrun/module_prompts/decom.py : DECOMPOSITION_PROMPT, verbatim --------
# Claim -> independent factual sub-questions. Upstream runs it on gpt-5-mini at
# temp 0; we used gpt-5.5 through dataset34v8 and gpt-6-luna from dataset50. The SAME subclaims go to every arm, so the
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


def decompose(claim: str, model: str = "gpt-6-luna") -> list:
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
# Format 0.4, read in the same design-first order.
RESOURCE_KEYS_V04 = ("baselines", "metrics", "datasets")
DESIGN_KEYS_V04 = ("threats", "validity_checks", "evidence_standards", "protocols")
MENU_KEYS_V04 = DESIGN_KEYS_V04 + RESOURCE_KEYS_V04
# Format 2, same order.
MENU_KEYS_V2 = ("threats", "evidence_standards", "baselines", "metrics", "datasets")

_SECTION_TITLE = {
    "datasets": "Datasets this subfield evaluates on",
    "models": "Models and methods this subfield runs",
    "metrics": "Metrics this subfield reports",
    "protocols": "How this subfield takes a measurement",
    "controls": "Control conditions this subfield expects alongside an intervention",
    "confounds": "Rival explanations a referee in this subfield will raise, and what "
                 "discriminates against each",
    "threats": "What a referee in this subfield will expect an experiment to rule out, "
               "what rules each out, and the kinds of claim it applies to",
    "validity_checks": "How this subfield's measurements can mislead, and the check that "
                       "catches each",
    "evidence_standards": "What this subfield accepts as evidence that a difference is real",
    "baselines": "Baselines this subfield compares against, and what a fair comparison "
                 "holds equal",
    # format 2's threats are 0.3's controls and confounds in one kind-tagged list
    "threats_v2": "Control conditions this subfield expects alongside an intervention, and "
                  "rival explanations a referee will raise, with what each rules out",
}


def _render_section(key: str, items: list) -> str:
    # format 2 carries a benchmark's eval recipe and a metric's caveat on the item
    extra = {"datasets": "eval_protocol", "metrics": "caveat"}.get(key)
    if extra and any(it.get(extra) for it in items):
        return "\n".join(
            f"- {it['name']}" + (f" ({it['role']})" if it.get("role") else "")
            + (f" ({it['direction']})" if it.get("direction") else "")
            + (f": {it[extra]}" if it.get(extra) else "") for it in items)
    if key == "threats" and any(it.get("kind") for it in items):
        return "\n".join(
            f"- [{it.get('kind', '')}] {it['name']}" + (f": {it['detail']}" if it.get("detail") else "")
            + (f" Rules out: {it['rules_out']}" if it.get("rules_out") else "")
            + (f" Ruled out by: {it['ruled_out_by']}" if it.get("ruled_out_by") else "")
            for it in items)
    if key in RESOURCE_KEYS or key in ("metrics", "datasets"):
        return ", ".join(
            it["name"] + (f" ({it['role']})" if it.get("role") else "")
            + (f" ({it['direction']})" if it.get("direction") else "")
            for it in items)
    if key == "baselines":
        return "\n".join(f"- {it['name']}" + (f" ({it['role']})" if it.get("role") else "")
                         + (f": hold equal {it['fair_comparison']}"
                            if it.get("fair_comparison") else "") for it in items)
    if key == "threats":
        return "\n".join(
            f"- {it['name']}" + (f": {it['detail']}" if it.get("detail") else "")
            + (f" Rule out: {it['rule_out']}" if it.get("rule_out") else "")
            + (f" [applies when the claim is: {'; '.join(it['applies_when'])}]"
               if it.get("applies_when") else "") for it in items)
    if key == "validity_checks":
        return "\n".join(
            f"- {it.get('instrument') or it['name']}"
            + (f" misleads when {it['misleads_when']}" if it.get("misleads_when") else "")
            + (f" Check: {it['check']}" if it.get("check") else "") for it in items)
    lines = []
    for it in items:
        tail = " ".join(x for x in (it.get("detail"), it.get("rules_out"),
                                    it.get("ruled_out_by")) if x)
        lines.append(f"- {it['name']}" + (f": {tail}" if tail else ""))
    return "\n".join(lines)


def norm_card_evidence(card_json: dict, n_per_key: int = 20,
                       sections: tuple = None) -> dict:
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
    from .curate import format_of
    fmt = format_of(card)
    if sections is None:
        sections = {"2": MENU_KEYS_V2, "0.4": MENU_KEYS_V04}.get(fmt, MENU_KEYS)
    ev = {}
    for k in sections:
        items = (card.get(k) or [])[:n_per_key]
        if items:
            title = _SECTION_TITLE["threats_v2" if fmt == "2" and k == "threats" else k]
            ev[title] = _render_section(k, items)
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


EMBED_MODEL = "text-embedding-3-large"
EMBED_DIMS = 1024        # the model's own Matryoshka truncation; keeps the cache ~100 MB


def _embed(texts: list) -> list:
    """Unit-normalised embeddings, batched."""
    import math
    litellm = llm._litellm()
    out = []
    for i in range(0, len(texts), 128):
        resp = litellm.embedding(model=EMBED_MODEL, input=texts[i:i + 128],
                                 dimensions=EMBED_DIMS, timeout=llm.TIMEOUT)
        for d in resp.data:
            v = d["embedding"] if isinstance(d, dict) else d.embedding
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
    return out


def _paper_embeddings(text: str, chunks: list, cache_dir: str) -> list:
    """Chunk embeddings for one paper, cached on disk by the text's hash: the same
    paper serves every claim whose bundle includes its subfield."""
    import array, hashlib
    key = hashlib.sha1(f"{EMBED_MODEL}:{EMBED_DIMS}:{text}".encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{key}.emb.f32")
    if os.path.exists(path):
        a = array.array("f")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        if len(a) == len(chunks) * EMBED_DIMS:
            return [a[i * EMBED_DIMS:(i + 1) * EMBED_DIMS] for i in range(len(chunks))]
    vecs = _embed(chunks)
    with open(path, "wb") as f:
        array.array("f", [x for v in vecs for x in v]).tofile(f)
    return vecs


def _dense_rank(query: str, vecs: list) -> list:
    q = _embed([query])[0]
    scores = sorted(((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(vecs)),
                    reverse=True)
    return [i for _, i in scores]


def _rrf(*rankings, k: int = 60) -> list:
    """Reciprocal rank fusion: robust to the two scorers' incomparable scales."""
    score = {}
    for r in rankings:
        for pos, i in enumerate(r):
            score[i] = score.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


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


# Since 2026-10-19 the rag arm is a norm source in its own right, not a length-matched
# control: the claim is that norms help, whether they arrive as passages or as a card.
# It used to be trimmed to the card's own character count and to read only the first
# 30 bundle papers (in effect the claim's first subfield). Now it reads every paper in
# the claim's bundle and keeps the top RAG_PASSAGES passages, whatever the card's size.
RAG_PASSAGES = 30


def fulltext_retrieval_evidence(bundle: dict, claim: str, cache_dir: str = ".pdfcache",
                                n_passages: int = RAG_PASSAGES, n_papers: int = None,
                                per_paper: int = 3, budget: int = None,
                                retriever: str = "bm25") -> dict:
    """Top BM25 passages from the bundle's papers.

    `per_paper` caps how many passages any one paper contributes, so a single
    long, on-topic paper cannot crowd out the rest of the field. `budget` (chars) and
    `n_papers` reproduce the older length-matched arm; both are off by default.

    `retriever`: "bm25" (lexical, the default), "dense" (EMBED_MODEL cosine) or
    "hybrid" (both, reciprocal-rank fused). Hybrid was tried on dataset34v7 and did
    worse (rag − nocard +0.014 against BM25's +0.038): dense pulls topical intro and
    discussion passages, where BM25 on the claim's own terms lands in methods sections."""
    from . import fulltext
    passages, vecs = [], []
    for p in (bundle.get("papers") or [])[:n_papers]:   # None: every paper
        # Main body only, as the card's map step reads it; no abstract fallback.
        text, _ = fulltext.fetch_body(p, cache_dir)
        if not text:
            continue
        chunks = _chunks(text)
        passages += [(p.get("title", ""), c) for c in chunks]
        if retriever != "bm25" and chunks:
            vecs += _paper_embeddings(text, chunks, cache_dir)
    if not passages:
        return {}
    lexical = (_bm25_rank(claim, [c for _, c in passages])
               if retriever != "dense" else [])
    dense = _dense_rank(claim, vecs) if retriever != "bm25" else []
    order = _rrf(lexical, dense) if lexical and dense else (lexical or dense)
    used, per, out = 0, {}, []
    for i in order:
        title, c = passages[i]
        if per.get(title, 0) >= per_paper:
            continue
        if budget and used + len(c) > budget:
            continue
        per[title] = per.get(title, 0) + 1
        out.append(f"[from: {title}]\n{c}")
        used += len(c)
        if len(out) >= n_passages or (budget and used >= budget * 0.97):
            break
    return {"Passages retrieved from the field's papers": "\n\n".join(out)} if out else {}


# --------------------------------------------------------------------------- #
# the proposer itself
# --------------------------------------------------------------------------- #
def render_prompts(claim: str, subclaims: list, current_evidence: dict,
                   budget: int = DEFAULT_BUDGET, prompt: str = None) -> tuple:
    """(system, instance). prompt="scify" renders exactly as dryrun/proposer.py does;
    "fieldnorms" moves FIELD_NORMS_KEY out of Current Evidence into its own block;
    "fieldnorms2" also gives LITERATURE_KEY its own block and explains only the keys present."""
    prompt = prompt or PROPOSER_PROMPT
    ev = dict(current_evidence or {})
    norms = ev.pop(FIELD_NORMS_KEY, "")
    lit = ev.pop(LITERATURE_KEY, "")
    if lit and prompt != "fieldnorms2":
        raise ValueError("relevant literature needs prompt='fieldnorms2'")
    if prompt == "scify":
        if norms:
            raise ValueError("field norms need prompt='fieldnorms'")
        sys_t, inst_t = SYSTEM_PROPOSER_TEMPLATE, INSTANCE_PROPOSER_TEMPLATE
    elif prompt == "fieldnorms2":
        sys_t, inst_t = system_template_fn2(bool(norms), bool(lit)), INSTANCE_PROPOSER_TEMPLATE_FN2
    else:
        sys_t, inst_t = SYSTEM_PROPOSER_TEMPLATE_FN, INSTANCE_PROPOSER_TEMPLATE_FN
    system = Template(sys_t, trim_blocks=True, lstrip_blocks=True).render(budget=budget).strip()
    instance = Template(inst_t, trim_blocks=True, lstrip_blocks=True).render(
        claim=claim, subclaims=subclaims, current_evidence=ev, field_norms=norms, literature=lit).strip()
    return system, instance


def norm_card_text(card_json: dict, sections: tuple = None) -> str:
    """The card as the fieldnorms prompt shows it: the same titled sections and one-line items
    norm_card_evidence renders for Current Evidence, as one block of text."""
    ev = norm_card_evidence(card_json, sections=sections)
    return "\n".join(f"{title}:\n" + "\n".join("  " + ln for ln in body.split("\n"))
                     for title, body in ev.items())


def propose(claim: str, subclaims: list, current_evidence: dict,
            model: str = "gpt-5.5", budget: int = DEFAULT_BUDGET, prompt: str = None) -> list:
    """Run the proposer. Returns the list of proposed experiment strings."""
    system, instance = render_prompts(claim, subclaims, current_evidence, budget, prompt)
    out = _complete_json_strict(system, instance, model=model)
    return out.get("experiments", [])


def _complete_json_strict(system: str, instance: str, model: str) -> dict:
    """Two messages, mirroring Query(system_prompt=..., instance_prompt=...), with
    JSON mode standing in for upstream's ProposerResponse structured output — the
    proposer's long prose otherwise breaks a naive json.loads on embedded quotes."""
    litellm = llm._litellm()
    om = config.OPEN_MODELS.get(model)
    if om and not os.environ.get(om["base_env"]):
        raise RuntimeError(f"{model} needs {om['base_env']} (its endpoint URL) and "
                           f"{om['key_env']} set in .env")
    route = ({"model": om["model"], "api_base": os.environ[om["base_env"]],
              "api_key": os.environ.get(om["key_env"], "")} if om else {"model": model})
    kwargs = dict(
        **route,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": instance + "\n\nReturn JSON of exactly the "
             'form {"experiments": ["...", "..."]} where each item is one full '
             "experiment description as prose."},
        ],
        temperature=(om["temperature"] if om else
                     1 if llm.wants_default_temperature(model) else 0.1),
        response_format={"type": "json_object"},
    )
    if not om:
        resp = litellm.completion(timeout=llm.TIMEOUT, **kwargs)
        return json.loads(resp.choices[0].message.content)
    # An open model reasons for minutes before it answers (Qwen3.8-27B: ~9k tokens at budget
    # 5). Unstreamed, the connection sits silent that long and the endpoint's gateway drops
    # it ("Connection error"); streamed, tokens keep it open. Dropped connections are retried.
    for attempt in range(3):
        try:
            # streamed, the timeout bounds the wait for the next chunk, so a stalled stream
            # fails in minutes instead of holding a worker for half an hour
            chunks = litellm.completion(timeout=300, stream=True, **kwargs)
            text = "".join((c.choices[0].delta.content or "") for c in chunks if c.choices)
            return json.loads(text)
        except Exception as e:
            if attempt == 2 or not any(s in type(e).__name__ for s in
                                       ("Connection", "InternalServer", "Timeout", "ServiceUnavailable")):
                raise
            time.sleep(30 * (attempt + 1))


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
