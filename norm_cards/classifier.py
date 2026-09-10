"""Stage 0: claim -> subfield(s) + claim type + temporal mode + entities (LLM).

Two axes from here steer the rest of the pipeline:
  - temporal_mode: static | forecast       (forecast adds trend queries)

The taxonomy's heuristic matcher only *grounds* the prompt (gives the LLM the
controlled vocabulary's closest entries); the decision is the LLM's.
"""

from typing import List

from .models import ClaimAnalysis
from .taxonomy import Taxonomy
from . import topics as topics_runtime
from . import llm

_PROMPT = """You are an expert at organizing AI research by subfield.

Given a scientific capability CLAIM, identify which AI subfield(s) it belongs to,
using the PwC TASK controlled vocabulary (benchmark-task granularity, e.g.
"Object Detection") — the primary subfield label for the claim.

(OpenAlex literature topics are resolved separately by OpenAlex's own classifier,
so you do NOT need to pick them here.)

CLAIM:
{claim}

PwC CONTROLLED VOCABULARY (subfield names by area):
{vocab}

CLOSEST PwC HEURISTIC MATCHES (hints, may be wrong): {matches}

Return JSON with EXACTLY these keys:
{{
  "subfields": ["canonical PwC subfield names from the vocabulary, ranked, 1-3"],
  "new_subfields": ["proposed names ONLY if nothing in the PwC vocabulary fits"],
  "temporal_mode": "static" or "forecast",
  "entities": {{
     "models": [...], "datasets": [...], "metrics": [...], "thresholds": [...]
  }},
  "confidence": 0.0-1.0
}}

Guidance:
- temporal_mode = "forecast" when the claim asserts a future capability by a date.
- entities: extract concrete models, datasets/benchmarks, metrics, and numeric
  thresholds mentioned in the claim.
"""


def _vocab_by_area(taxonomy: Taxonomy) -> str:
    by_area = {}
    for t in taxonomy.tasks:
        by_area.setdefault(t.get("area") or "Other", []).append(t["task"])
    return "\n".join(f"[{area}] " + ", ".join(sorted(names))
                     for area, names in sorted(by_area.items()))


def classify(claim: str, taxonomy: Taxonomy, model: str = None) -> ClaimAnalysis:
    matches = taxonomy.heuristic_match(claim, top_k=8)
    match_str = ", ".join(m["task"] for m in matches) or "(none)"
    data = llm.complete_json(
        _PROMPT.format(claim=claim, vocab=_vocab_by_area(taxonomy), matches=match_str),
        model=model)

    subfields: List[str] = []
    for s in data.get("subfields", []) or []:
        hit = taxonomy.get(s)
        if hit:
            subfields.append(hit["task"])  # canonical casing
        else:
            near = taxonomy.find_near_duplicate(s)
            subfields.append(near or s)

    new_subfields: List[str] = []
    for s in data.get("new_subfields", []) or []:
        canonical = taxonomy.add_custom(s)
        new_subfields.append(canonical)
        if canonical not in subfields:
            subfields.append(canonical)
    if new_subfields:
        taxonomy.save()  # persist newly added subfields

    ent = data.get("entities", {}) or {}
    # OpenAlex topics: resolved by OpenAlex's own text classifier over the full
    # taxonomy (not the LLM), so the claim lands in the same topic space that
    # tags the papers we later filter/rank against. The raw claim alone
    # mis-classifies on jargon, so we hand it the LLM's subfields + entities as
    # disambiguating context.
    ctx_terms = subfields + sum((ent.get(k) or [] for k in
                                 ("models", "datasets", "metrics")), [])
    context = "; ".join(str(t) for t in ctx_terms if t)
    resolved = topics_runtime.resolve_topics(claim, context=context)
    topic_ids = [tid for tid, _ in resolved]
    topic_names = [name for _, name in resolved]

    return ClaimAnalysis(
        subfields=subfields,
        new_subfields=new_subfields,
        openalex_topic_ids=topic_ids,
        openalex_topic_names=topic_names,
        temporal_mode=data.get("temporal_mode", "static"),
        entities={
            "models": ent.get("models", []),
            "datasets": ent.get("datasets", []),
            "metrics": ent.get("metrics", []),
            "thresholds": ent.get("thresholds", []),
        },
        confidence=float(data.get("confidence", 0.5) or 0.5),
        method="llm",
    )
