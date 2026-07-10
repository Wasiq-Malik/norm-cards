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
from . import llm, fulltext

MENU_KEYS = ("datasets", "models", "metrics", "protocols")


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
_MAP_PROMPT = """This paper is in the subfield(s) {subfields}. From its text, extract ONLY the
experimental norms THIS paper actually uses or establishes.

Return JSON {{"datasets":[...],"models":[...],"metrics":[...],"protocols":[...]}} —
each a short string (dataset/model/metric name; protocol = one line, e.g.
"poison rate swept 1-10%"). Empty lists if absent. Do not invent.

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
_REDUCE_PROMPT = """You are building a GENERIC scientific norm-card MENU for the subfield(s):
{subfields}.

Below are per-paper extractions (each paper Pn lists what IT uses). Consolidate
into ONE clean menu for the subfield.

RULES:
- CANONICALIZE duplicates ("Cifar10"/"CIFAR-10" -> "CIFAR-10").
- KEEP ONLY items belonging to the subfield(s) above; DROP items that leaked in
  from off-subfield papers.
- Do NOT invent anything absent from the extractions.
- Rank each list by how many papers support it (most standard first).
- For every item, list the supporting paper ids (the P-numbers).

Claim context (relevance only): {claim}

PER-PAPER EXTRACTIONS:
{blob}

Return JSON:
{{"datasets":[{{"name":"...","role":"train|eval|robustness_eval","evidence":["P0"]}}],
  "models":[{{"name":"...","role":"backbone|method_under_test|baseline","evidence":[...]}}],
  "metrics":[{{"name":"...","direction":"higher_better|lower_better","evidence":[...]}}],
  "protocols":[{{"name":"...","detail":"one line","evidence":[...]}}]}}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _faithful(item_name: str, paper_extract: Dict) -> bool:
    """Did this item actually appear in the cited paper's own extraction?"""
    n = _norm(item_name)
    alt = _norm(re.sub(r"\(.*?\)", "", item_name))
    pool = " ".join(_norm(x) for k in MENU_KEYS for x in (paper_extract.get(k) or []))
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
                if _faithful(it.get("name", ""), per_paper[title]):
                    ev.append(title)
                else:
                    n_drop_link += 1
            if not ev:
                n_drop_item += 1
                continue
            it["evidence"] = sorted(set(ev))  # dedup evidence
            kept[k].append(it)
    kept["_guard"] = {"items_dropped": n_drop_item, "evidence_links_dropped": n_drop_link}
    return kept


# --------------------------------------------------------------------------- #
# recipes: compose runnable experiments from claim + finished menu
# --------------------------------------------------------------------------- #
_RECIPE_PROMPT = """You are composing the EXECUTABLE part of a scientific norm card. Using ONLY the
menu below (introduce no datasets/models/metrics not in it), design 2-4 concrete
experiment recipes a verification agent could RUN to test claims in subfield(s)
{subfields}. Keep them representative of the subfield's standard tests, but the
CLAIM shows the kind of assertion they must be able to check.

CLAIM: {claim}

MENU (the only allowed building blocks):
{menu}

Return JSON {{"recipes":[
  {{"goal":"what this establishes",
    "dataset":"<from menu>","model":"<backbone from menu>",
    "method_under_test":"<from menu>","attack_or_condition":"<from menu protocols>",
    "metrics":["<from menu>"],"pass_condition":"concrete, measurable",
    "rationale":"why this is the standard test"}}]}}"""


def _menu_text(menu: Dict, n: int = 18) -> str:
    return "\n".join(f"{k}: {[it['name'] for it in menu[k][:n]]}" for k in MENU_KEYS)


def _cards_text(cards: List[Dict]) -> str:
    """One labeled menu block per subfield card, for the recipe prompt."""
    return "\n\n".join(f"=== subfield: {c['subfield']} ===\n{_menu_text(c['menu'])}"
                       for c in cards)


def _recipes(cards: List[Dict], subfields, claim: str, model: str) -> List[Dict]:
    """Claim-level recipes composed across ALL the claim's per-subfield menus."""
    out = llm.complete_json(_RECIPE_PROMPT.format(
        subfields=subfields, claim=claim[:400], menu=_cards_text(cards)), model=model)
    # grounding name set = union of every item across every subfield card
    names = {_norm(it["name"]) for c in cards for k in MENU_KEYS for it in c["menu"][k]}

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
                  progress: bool = True) -> Dict:
    analysis = bundle["analysis"]
    if analysis.get("claim_type") != "empirical":
        return {"type": "scientific_norm_card_set", "skipped": True,
                "reason": f"claim_type={analysis.get('claim_type')} (empirical only)",
                "problem_id": bundle.get("problem_id"), "claim": bundle.get("claim")}

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

    # Claim-level recipes composed across ALL the per-subfield menus.
    recipes = _recipes(norm_cards, subfields, claim, recipe_model)
    if progress:
        print(f"  recipes={len(recipes)}")

    return {
        "type": "scientific_norm_card_set", "format_version": "0.2",
        "claim_type": "empirical", "problem_id": bundle.get("problem_id"),
        "claim": claim, "subfields": subfields,
        "provenance": {"n_papers_bundle": len(bundle["papers"]),
                       "n_papers_deduped": len(papers), "n_fulltext": n_ft,
                       "pdf_sources": sources, "map_model": map_model,
                       "reduce_model": reduce_model, "recipe_model": recipe_model},
        "norm_cards": norm_cards,
        "experiment_recipes": recipes,
    }


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
