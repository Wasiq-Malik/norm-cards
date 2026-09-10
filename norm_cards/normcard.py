"""Scientific norm-card generation from a paper bundle (empirical claims).

    bundle -> dedup papers -> fetch full text -> per-paper extract (map, cheap
    model, parallel) -> consolidate to a subfield MENU (reduce, strong model,
    canonicalize + subfield-filter + evidence) -> grounding guard -> compose
    EXPERIMENT RECIPES from claim + menu (separate call) -> norm_card

Design rationale (validated by experiment on the sample claims):
- Full text, not abstracts: abstracts don't name datasets (they yielded ZERO);
  the experimental-setup sections do. Map-reduce beats a single mega-dump
  (context blow-up) and bm25s-RAG (lower recall) on menu coverage.
- Map on a cheap model per paper, reduce on a strong one: the reduce is where
  canonicalization ("Cifar10"->"CIFAR-10") and off-subfield filtering happen.
- Menu then recipes in SEPARATE calls: recipe generation cannot degrade the
  menu because the menu is frozen first. Recipes see the claim + finished menu
  so they pinpoint the experiments that would verify THAT claim.
- Evidence on every item; the guard drops any item whose evidence doesn't
  survive (nothing ungrounded reaches the card).
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .models import normalize_title
from . import llm, fulltext, curate

from .curate import ALL_KEYS as MENU_KEYS, RESOURCE_KEYS, DESIGN_KEYS  # noqa: E402


# --------------------------------------------------------------------------- #
# dedup (bundle papers are dicts; collapse cross-source + preprint/published)
# --------------------------------------------------------------------------- #
def _dedup_papers(papers: List[Dict]) -> List[Dict]:
    def dkey(p):
        doi = (p.get("doi") or "").lower().strip()
        if doi:
            return "doi:" + doi
        ax = (p.get("arxiv_id") or "").lower().strip()
        return "arxiv:" + ax if ax else "title:" + normalize_title(p.get("title", ""))

    def vkey(p):
        au = (p.get("authors") or [""])
        return normalize_title(p.get("title", "")) + "|" + (au[0] if au else "").lower().strip() \
            if p.get("title") and au and au[0] else ""

    seen, out = {}, []
    for p in papers:
        seen.setdefault(dkey(p), []).append(p)
    merged = [max(grp, key=lambda x: len(x.get("abstract", ""))) for grp in seen.values()]
    # second pass on title+first-author for preprint/published DOI variants
    vseen, final = {}, []
    for p in merged:
        k = vkey(p)
        if not k:
            final.append(p)
        elif k in vseen:
            if len(p.get("abstract", "")) > len(vseen[k].get("abstract", "")):
                final[final.index(vseen[k])] = p
                vseen[k] = p
        else:
            vseen[k] = p
            final.append(p)
    return final


# --------------------------------------------------------------------------- #
# map: per-paper extraction
# --------------------------------------------------------------------------- #
_MAP_PROMPT = """This paper is in the subfield(s) {subfields}. Read its experimental sections and
extract ONLY the norms THIS paper actually uses. Do not invent, do not
generalize, do not import practice from papers you happen to know.

Six lists. The first three are WHAT it runs on; the last three are HOW it argues.

  datasets   named datasets/benchmarks it evaluates on
  models     named checkpoints or named methods it runs. NOT architecture
             classes — "Transformer", "MLP", "LLM", "neural network" are
             categories, not things a paper runs, so omit them.
  metrics    named quantities it reports
  protocols  how a measurement is actually taken: the sweep, the split, the
             decoding setting, the number of seeds, the aggregation rule.
             One line each, with the specifics: "poison rate swept 1-10%",
             not "poison rate sweep".

  controls   the comparison conditions it runs ALONGSIDE its main condition so
             that a positive result cannot be explained by the intervention
             merely having happened. A control is a condition, not a metric and
             not a baseline method: a random or size-matched version of the
             intervention, a placebo/sham, an unmodified reference model, an
             ablation that removes the paper's own mechanism, a shuffled or
             permuted input. For each, say what it is and what a difference
             against it licenses:
             {{"name":"random size-matched head set",
               "detail":"same number of heads chosen at random",
               "rules_out":"that any equally-sized intervention would do it"}}

  confounds  the RIVAL EXPLANATIONS the paper explicitly takes seriously — an
             account under which its headline number would come out the same
             without its claim being true — and the specific measurement it
             makes to rule each one out. Look for "one might worry that", "to
             ensure this is not simply", "an alternative explanation", "we
             therefore also measure", and for limitations the authors concede.
             {{"name":"effect is general capability damage, not belief change",
               "detail":"why the headline result would look identical if so",
               "ruled_out_by":"score the steered model on an unrelated held-out
                               task and show it is unchanged"}}

If the paper runs no controls or acknowledges no rival explanation, return empty
lists — an absent norm is a finding, and inventing one corrupts the card.

Return JSON {{"datasets":[...],"models":[...],"metrics":[...],
"protocols":[...],"controls":[...],"confounds":[...]}} — datasets/models/metrics
are plain strings; protocols/controls/confounds are objects as shown above.

PAPER: {title}
TEXT (may be truncated; may be only the abstract if full text was unavailable):
{text}"""


def _extract_one(args):
    title, subfields, text, model = args
    try:
        d = llm.complete_json(_MAP_PROMPT.format(
            subfields=subfields, title=title, text=text[:40000]), model=model)
        return title, {k: (d.get(k) or []) for k in MENU_KEYS}
    except Exception as e:
        return title, {"error": str(e)}


def _map(docs: List[Dict], subfields, model: str, workers: int = 6) -> Dict[str, Dict]:
    args = [(d["title"], subfields, d["text"], model) for d in docs]
    per_paper = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for title, out in ex.map(_extract_one, args):
            per_paper[title] = out
    return per_paper


# --------------------------------------------------------------------------- #
# reduce: consolidate to a clean, subfield-filtered, grounded menu
# --------------------------------------------------------------------------- #
_REDUCE_PROMPT = """You are consolidating a scientific NORM CARD for the subfield: {subfields}.

A norm card answers one question for someone competent but new to this subfield:
*what would a referee here expect an experiment on this claim to contain?* It is
not a literature summary and not a reading list.

Below are per-paper extractions (paper Pn lists what IT does). Consolidate them
into ONE card for the subfield above.

RULES
- CANONICALIZE. One entry per thing. Collapse checkpoint variants to the family
  and release the field actually names — "Llama-2-7B-Chat", "LLaMA2-7B-Chat-HF"
  and "Llama 2" are ONE entry, "Llama-2"; so are "Llama 3.1" and "Llama 3.2",
  as "Llama-3". Sizes, -Chat/-Instruct/-hf suffixes and point releases are not
  norms. Same for datasets: "AdvBench", "AdvBench (subset)" and "AdvBench:
  Harmful Behaviors" are one entry. Do NOT merge things that are genuinely
  distinct (MQuAKE-CF and MQuAKE-T are two datasets).
- SUBFIELD ONLY. Drop anything that leaked in from an off-subfield paper.
- NOTHING NEW. Every item must come from the extractions below.
- RANK by how many papers support it, most standard first, and cite the P-ids.
- PREFER THE SPECIFIC. "Report mean over 5 seeds with 95% CI" is a norm;
  "report results carefully" is not. Drop items that would be true of any
  empirical field.

The two design sections carry most of the card's value, so consolidate them with
more care than the resource lists:

  controls   Merge into the distinct CONTROL CONDITIONS this subfield expects,
             stated so they can be applied to a new intervention rather than
             copied. "A size-matched random version of whatever component was
             intervened on" transfers; "random heads" does not. For each, state
             what a difference against it licenses.
  confounds  Merge into the distinct RIVAL EXPLANATIONS a referee in this
             subfield raises — the accounts under which a headline result comes
             out the same without the claim being true — each with the
             measurement that discriminates against it. These are the norms most
             often missing from a proposed experiment, so keep every distinct
             one you find, even when only one paper raises it.

Claim context (for judging relevance only — do NOT tailor the card to it, and do
NOT copy the claim's own entities into the card): {claim}

PER-PAPER EXTRACTIONS:
{blob}

Return JSON:
{{"datasets":[{{"name":"...","role":"train|eval|robustness_eval","evidence":["P0"]}}],
  "models":[{{"name":"...","role":"backbone|method_under_test|baseline","evidence":[...]}}],
  "metrics":[{{"name":"...","direction":"higher_better|lower_better","evidence":[...]}}],
  "protocols":[{{"name":"...","detail":"the specifics, one line","evidence":[...]}}],
  "controls":[{{"name":"...","detail":"how the condition is constructed",
                "rules_out":"what a difference against it licenses","evidence":[...]}}],
  "confounds":[{{"name":"the rival explanation","detail":"why the headline result
                 would look the same under it",
                 "ruled_out_by":"the measurement that discriminates","evidence":[...]}}]}}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _text_of(x) -> str:
    """Per-paper extractions hold plain strings for resources and objects for the
    design keys; flatten either to searchable text."""
    if isinstance(x, dict):
        return " ".join(str(v) for v in x.values() if isinstance(v, str))
    return str(x)


_STOPWORDS = {"the", "a", "an", "of", "for", "and", "or", "to", "on", "in", "with",
              "that", "is", "are", "be", "by", "as", "it", "this", "not", "same"}


def _faithful(item, paper_extract: Dict, key: str) -> bool:
    """Did this item actually come from the cited paper's own extraction?

    Resource names are near-verbatim after canonicalization, so they get a
    substring test. Design items are paraphrases by construction — the reduce
    step is supposed to restate a control so it transfers — so demanding a
    substring there would delete exactly the content the card exists to carry.
    They instead have to share vocabulary with something that paper actually
    reported under the same key."""
    pool = " ".join(_norm(_text_of(x)) for k in MENU_KEYS
                    for x in (paper_extract.get(k) or []))
    name = item.get("name", "") if isinstance(item, dict) else str(item)
    if key in DESIGN_KEYS:
        own = " ".join(_text_of(x) for x in (paper_extract.get(key) or []))
        if not own:
            return False            # paper reported no such norm; cite is spurious
        own_w = {w for w in re.findall(r"[a-z]{4,}", own.lower()) if w not in _STOPWORDS}
        mine = {w for w in re.findall(r"[a-z]{4,}", _text_of(item).lower())
                if w not in _STOPWORDS}
        return len(own_w & mine) >= 2
    n = _norm(name)
    alt = _norm(re.sub(r"\(.*?\)", "", name))
    if n in pool or (alt and alt in pool):
        return True
    return len(n) >= 6 and any(n[i:i + 6] in pool for i in range(len(n) - 5))


def _reduce(per_paper: Dict[str, Dict], subfield: str, claim: str, model: str) -> Dict:
    """Consolidate the per-paper extractions into ONE menu scoped to a SINGLE
    subfield. Called once per subfield so each card stays focused and papers
    from adjacent subfields contribute nothing to it."""
    titles = [t for t, d in per_paper.items() if isinstance(d, dict) and "error" not in d]
    tid = {t: i for i, t in enumerate(titles)}
    blob = "\n".join(f"[P{tid[t]}] {t}\n   " +
                     __import__("json").dumps({k: per_paper[t].get(k, []) for k in MENU_KEYS})
                     for t in titles)
    menu = llm.complete_json(_REDUCE_PROMPT.format(
        subfields=subfield, claim=claim[:300], blob=blob[:70000]), model=model)

    # grounding + faithfulness guard: resolve P-ids to titles, keep only links
    # whose item actually appears in that paper's extraction; drop empty items.
    kept, n_drop_item, n_drop_link = {}, 0, 0
    for k in MENU_KEYS:
        kept[k] = []
        for it in (menu.get(k) or []):
            ev = []
            for e in (it.get("evidence") or []):
                m = re.match(r"P(\d+)", str(e))
                if not m or int(m.group(1)) >= len(titles):
                    continue
                title = titles[int(m.group(1))]
                if _faithful(it, per_paper[title], k):
                    ev.append(title)
                else:
                    n_drop_link += 1
            if not ev:
                n_drop_item += 1
                continue
            it["evidence"] = sorted(set(ev))  # dedup evidence
            kept[k].append(it)
    kept["_guard"] = {"items_dropped": n_drop_item, "evidence_links_dropped": n_drop_link,
                      "kept_per_key": {k: len(kept[k]) for k in MENU_KEYS}}
    return kept


# --------------------------------------------------------------------------- #
# recipes: compose runnable experiments from claim + finished menu
# --------------------------------------------------------------------------- #
_RECIPE_DISCIPLINE = """How to design the recipe SET (applies to every recipe):
1. DECOMPOSE the claim into its dependency chain and cover every stage that
   applies, in order:
   - "gate": cheap prerequisite checks that can refute the claim outright
     (e.g. if the claim needs adversarial metric >= X, clean performance >= X is
     a prerequisite — attacks only degrade). Run these FIRST.
   - "apparatus": anything the claim PRESUPPOSES but that must be BUILT and
     VALIDATED before the headline test (a trained generator, a fitted
     surrogate, a constructed poison set). Validation needs its own measurable
     pass_condition: the apparatus must demonstrably be what the claim says it
     is (e.g. an "illumination-consistent" generator must produce
     illumination-like, structure-preserving perturbations — not arbitrary
     noise).
   - "headline": the direct test of the claim's assertion, enforcing EVERY
     constraint the claim states (norm bounds, budgets, rates, splits,
     resolutions). If a stated constraint is not enforced by the design, list
     it in "constraints_not_enforced" rather than silently dropping it.
   - "control": checks that the result is not an artifact AND attribute it to the
     claim's specific mechanism: strength/robustness sweeps, random/weak-baseline
     comparisons (the method must beat random), matched-budget comparisons, and —
     crucially — an ABLATION that removes the claim's OWN mechanism (not merely a
     comparison to a trivial alternative), so the effect is attributable to it.
2. Every pass_condition must be a REAL, decisive gate tied to the claim's own
   numeric thresholds — never "record/log the value". State the comparison,
   the threshold, and what refutes.
3. Order recipes so the cheapest potential refutation comes first; note
   dependencies between recipes in "depends_on" (list of recipe goals).
"""

_RECIPE_PROMPT = """You are composing the EXECUTABLE part of a scientific norm card: design 3-6
concrete experiment recipes a verification agent could RUN to verify or refute
the CLAIM, for subfield(s) {subfields}.

CLAIM: {claim}

CLAIM ENTITIES (the specific models/datasets/metrics/thresholds the CLAIM names).
Use these for the EXACT artifact under test — the model/dataset/metric the claim
is actually about — even if it is newer than the literature and not in the menu:
{entities}

MENU (the field's standard datasets/models/metrics/protocols, grounded in the
gathered papers). Use these for the standard BASELINES, benchmarks, metrics, and
experimental protocol the field expects:
{menu}

""" + _RECIPE_DISCIPLINE + """
Grounding rules:
- When the CLAIM names a specific model/dataset/metric, use it (from CLAIM
  ENTITIES) rather than a menu substitute.
- Introduce nothing that is in NEITHER the claim entities NOR the menu.

Return JSON {{"recipes":[
  {{"stage":"gate|apparatus|headline|control",
    "goal":"what this establishes",
    "dataset":"<claim entity or menu>","model":"<claim entity or menu>",
    "method_under_test":"<claim entity or menu>","attack_or_condition":"<from menu protocols>",
    "metrics":["<claim entity or menu>"],"pass_condition":"concrete, decisive",
    "constraints_not_enforced":["claim-stated constraints this recipe does not enforce"],
    "depends_on":["goals of prerequisite recipes"],
    "rationale":"why this is the standard test"}}]}}"""


def _menu_text(menu: Dict, n: int = 18) -> str:
    """Render a menu as text for a prompt.

    Earlier versions emitted only `[name, name, ...]` per key, which threw away
    `role`, `direction` and the protocol/control/confound `detail` — everything
    except the noun. A reader got a word list and no way to tell a baseline from
    a backbone, or what a listed protocol actually specifies."""
    out = []
    for k in RESOURCE_KEYS:
        items = (menu.get(k) or [])[:n]
        if not items:
            continue
        out.append(f"{k}: " + ", ".join(
            it["name"] + (f" ({it['role']})" if it.get("role") else "")
            + (f" ({it['direction']})" if it.get("direction") else "")
            for it in items))
    for k in DESIGN_KEYS:
        items = (menu.get(k) or [])[:n]
        if not items:
            continue
        out.append(f"{k}:")
        for it in items:
            tail = " | ".join(x for x in (it.get("detail"), it.get("rules_out"),
                                          it.get("ruled_out_by")) if x)
            out.append(f"  - {it['name']}" + (f" — {tail}" if tail else ""))
    return "\n".join(out)


def _cards_text(cards: List[Dict]) -> str:
    """One labeled menu block per subfield card, for the recipe prompt."""
    return "\n\n".join(f"=== subfield: {c['subfield']} ===\n{_menu_text(c['menu'])}"
                       for c in cards)


def card_text(card: Dict, n: int = 18) -> str:
    """Render a curated claim-level card (the merged menu) as prompt text."""
    return _menu_text(card, n)


def _entities_text(entities: Dict) -> str:
    entities = entities or {}
    return "\n".join(f"{k}: {entities.get(k) or []}"
                     for k in ("models", "datasets", "metrics", "thresholds"))


def _recipes(cards: List[Dict], subfields, claim: str, entities: Dict,
             model: str) -> List[Dict]:
    """Claim-level recipes composed across the per-subfield menus AND the claim's
    own extracted entities (so the exact model/dataset/metric the claim names is
    used for the artifact under test, while the menu supplies field-standard
    baselines/protocol/metrics)."""
    out = llm.complete_json(_RECIPE_PROMPT.format(
        subfields=subfields, claim=claim[:400], entities=_entities_text(entities),
        menu=_cards_text(cards)), model=model)
    # grounding set = every menu item across every card, PLUS the claim entities
    names = {_norm(it["name"]) for c in cards for k in MENU_KEYS for it in c["menu"][k]}
    for k in ("models", "datasets", "metrics", "thresholds"):
        names |= {_norm(str(e)) for e in ((entities or {}).get(k) or []) if _norm(str(e))}

    def grounded(v):
        x = _norm(v)
        return bool(x) and any(x in m or m in x for m in names if len(m) > 3)

    for r in out.get("recipes", []):
        checks = {f: grounded(str(r.get(f, ""))) for f in
                  ("dataset", "model", "method_under_test")}
        # metrics are a list — every named metric must trace to a menu item too
        bad_metrics = [mtr for mtr in (r.get("metrics") or []) if not grounded(str(mtr))]
        checks["metrics"] = not bad_metrics
        if bad_metrics:
            r["_ungrounded_metrics"] = bad_metrics
        r["_grounded"] = checks
    return out.get("recipes", [])


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def generate_card(bundle: Dict, cache_dir: str, map_model: str = "gpt-5-mini",
                  reduce_model: str = "gpt-5", recipe_model: str = "gpt-5",
                  min_evidence: int = 2, resource_cap: int = 8, design_cap: int = 14,
                  progress: bool = True) -> Dict:
    analysis = bundle["analysis"]
    claim, subfields = bundle["claim"], analysis["subfields"]
    papers = _dedup_papers(bundle["papers"])

    docs, sources = [], {}
    for p in papers:
        text, src = fulltext.fetch_fulltext(p, cache_dir)
        sources[src] = sources.get(src, 0) + 1
        docs.append({"title": p["title"], "text": text or (p.get("abstract") or ""),
                     "has_fulltext": bool(text)})
    n_ft = sum(1 for d in docs if d["has_fulltext"])
    if progress:
        print(f"  papers: {len(bundle['papers'])} -> {len(papers)} deduped; "
              f"full text {n_ft}/{len(docs)}  {sources}")

    # MAP once over all papers; REDUCE once PER SUBFIELD -> one focused card each.
    per_paper = _map(docs, subfields, map_model)
    norm_cards = []
    for sf in subfields:
        menu = _reduce(per_paper, sf, claim, reduce_model)
        card = {"subfield": sf, "menu": {k: menu[k] for k in MENU_KEYS},
                "guard": menu.get("_guard", {})}
        norm_cards.append(card)
        if progress:
            print(f"  [{sf}] " + ", ".join(f"{k}={len(menu[k])}" for k in MENU_KEYS)
                  + f"  guard={menu.get('_guard', {})}")

    # Per-subfield menus are the raw material, not the deliverable. A claim routed
    # to three subfields used to ship three full menus whose union was ~30%
    # literally duplicate strings and 59-75% single-paper items — 228-351 entries,
    # of which the reader could not tell which were norms. Merge and prune to one
    # claim-level card; keep the raw menus for provenance.
    card, curation = curate.curate(norm_cards, min_evidence=min_evidence,
                                   resource_cap=resource_cap, design_cap=design_cap)
    if progress:
        t = curation["totals"]
        print(f"  curated: {t['in']} -> {t['out']} items  "
              + ", ".join(f"{k}={len(card[k])}" for k in MENU_KEYS))

    # Claim-level recipes composed across ALL the per-subfield menus + entities.
    recipes = _recipes(norm_cards, subfields, claim, analysis.get("entities") or {},
                       recipe_model)
    if progress:
        print(f"  recipes={len(recipes)}")

    return {
        "type": "scientific_norm_card_set", "format_version": "0.3",
        "problem_id": bundle.get("problem_id"),
        "claim": claim, "subfields": subfields,
        "provenance": {"n_papers_bundle": len(bundle["papers"]),
                       "n_papers_deduped": len(papers), "n_fulltext": n_ft,
                       "pdf_sources": sources, "map_model": map_model,
                       "reduce_model": reduce_model, "recipe_model": recipe_model},
        "norm_cards": norm_cards,          # per subfield, raw — provenance
        "card": card,                      # merged + curated — what consumers read
        "curation": curation,
        "experiment_recipes": recipes,
    }


_BASELINE_PROMPT = """Design 3-6 concrete experiment recipes a verification agent could RUN to verify
or refute the CLAIM below. Use your own knowledge of standard practice for this
kind of claim; NO papers or norm card are provided.

CLAIM: {claim}

CLAIM ENTITIES (models/datasets/metrics/thresholds named in the claim):
{entities}

""" + _RECIPE_DISCIPLINE + """
Return JSON {{"recipes":[
  {{"stage":"gate|apparatus|headline|control",
    "goal":"what this establishes",
    "dataset":"...","model":"...","method_under_test":"...",
    "attack_or_condition":"...","metrics":["..."],
    "pass_condition":"concrete, decisive",
    "constraints_not_enforced":["claim-stated constraints this recipe does not enforce"],
    "depends_on":["goals of prerequisite recipes"],
    "rationale":"why this is the standard test"}}]}}"""


def baseline_recipes(claim: str, entities: Dict, model: str = "gpt-5") -> List[Dict]:
    """ABLATION: recipes from the claim + its entities ALONE, with no gathered
    papers and no norm-card menu — the naive 'just ask the LLM' approach, to
    measure what the full search + norm-card pipeline actually adds."""
    out = llm.complete_json(_BASELINE_PROMPT.format(
        claim=claim[:600], entities=_entities_text(entities)), model=model)
    return out.get("recipes", [])


def _main():
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(
        description="Generate a scientific norm card from a paper bundle "
                    "(the output of `python -m norm_cards.run`).")
    ap.add_argument("--bundle", required=True, help="Path to a bundle.json")
    ap.add_argument("--out", default="", help="Output path (default: norm_card.json "
                                              "alongside the bundle)")
    ap.add_argument("--cache_dir", default=".pdfcache", help="Where to cache fetched PDFs")
    ap.add_argument("--map_model", default="gpt-5-mini")
    ap.add_argument("--reduce_model", default="gpt-5")
    ap.add_argument("--recipe_model", default="gpt-5")
    args = ap.parse_args()

    bundle = json.load(open(args.bundle, encoding="utf-8"))
    card = generate_card(bundle, args.cache_dir, map_model=args.map_model,
                         reduce_model=args.reduce_model, recipe_model=args.recipe_model)
    out = args.out or os.path.join(os.path.dirname(args.bundle) or ".", "norm_card.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    _main()
