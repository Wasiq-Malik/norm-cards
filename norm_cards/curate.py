"""Deterministic curation of a norm-card menu: canonicalize, merge, prune.

The extraction stages (normcard.map/reduce) are generous on purpose — recall
first, a model can only consolidate what it was shown. This module is the
opposite half: a set of rules, no model, that turns the raw union into
something small enough to read.

Three problems it solves, all measured on the v0.2 cards:

  version bloat    "Llama-2-7B-Chat", "LLaMA2-7B-Chat-HF", "Llama 2",
                   "Llama-3 (8B/70B)", "Llama-3.1-8B" are five entries for two
                   families. Checkpoint sizes and chat/instruct suffixes are not
                   field norms; the family is.
  category errors  "Transformer", "MLP", "RNN", "GAN", "Large Language Models"
                   appeared under `models`. An architecture class is not a model
                   the field runs on.
  one-paper items  59-75% of every card was cited by exactly ONE paper. That is
                   not a norm, it is one group's choice. A norm is what the
                   subfield does repeatedly, so resources need corroboration.

Kept deliberately conservative: anything that does not match a known family
passes through untouched, so methods (ROME, MEMIT, SERAC) and one-off datasets
are never mangled by a rule meant for checkpoints.
"""

import os
import re
from typing import Dict, List, Tuple

# Item keys that describe WHAT the field runs on. These bloat, and they are the
# ones a norm needs corroboration for.
RESOURCE_KEYS = ("datasets", "models", "metrics")
# Item keys that describe HOW the field runs things. These are scarce and are
# what actually transfers to an experiment design, so they are pruned gently.
DESIGN_KEYS = ("protocols", "controls", "confounds")
ALL_KEYS = RESOURCE_KEYS + DESIGN_KEYS


# --------------------------------------------------------------------------- #
# model-family canonicalization
# --------------------------------------------------------------------------- #
# (family regex, display name). Order matters: the first match wins, so more
# specific families ("mixtral", "llama guard") must precede their prefixes.
_FAMILIES: List[Tuple[str, str]] = [
    (r"llama[\s\-_]*guard", "Llama Guard"),
    (r"code[\s\-_]*llama", "Code Llama"),
    (r"tiny[\s\-_]*llama", "TinyLlama"),
    (r"\bl+ama\b|\bllama", "Llama"),
    (r"mixtral", "Mixtral"),
    (r"mistral", "Mistral"),
    (r"\bgpt[\s\-_]*neox", "GPT-NeoX"),
    (r"\bgpt[\s\-_]*neo\b", "GPT-Neo"),      # a distinct family, not a GPT release
    (r"\bgpt[\s\-_]*j\b", "GPT-J"),
    (r"\bgpt\b|\bchatgpt\b", "GPT"),
    (r"claude", "Claude"),
    (r"gemini", "Gemini"),
    (r"\bgemma", "Gemma"),
    (r"\bpalm\b", "PaLM"),
    (r"qwen", "Qwen"),
    (r"vicuna", "Vicuna"),
    (r"deepseek", "DeepSeek"),
    (r"\bopt[\s\-_]*\d", "OPT"),
    (r"pythia", "Pythia"),
    (r"falcon", "Falcon"),
    (r"chatglm", "ChatGLM"),
    (r"mini[\s\-_]*gpt", "MiniGPT"),
    (r"llava", "LLaVA"),
    (r"instructblip", "InstructBLIP"),
    (r"cogvlm", "CogVLM"),
    (r"guanaco", "Guanaco"),
    (r"\bmpt[\s\-_]*\d", "MPT"),
    (r"\bphi[\s\-_]*\d", "Phi"),
    (r"zephyr", "Zephyr"),
    (r"\bbloom", "BLOOM"),
    (r"distil[\s\-_]*bert", "DistilBERT"),
    (r"roberta", "RoBERTa"),
    (r"\bbert\b", "BERT"),
    (r"\bt5\b|flan[\s\-_]*t5", "T5"),
]
_FAMILIES = [(re.compile(p, re.I), name) for p, name in _FAMILIES]

# Architecture classes and umbrella terms. They name a category, not something a
# paper can be said to run on, so they carry no experimental information.
_GENERIC_MODELS = {
    "transformer", "transformers", "mlp", "rnn", "lstm", "gru", "cnn", "gan",
    "vae", "resnet", "alexnet", "llm", "llms", "large language model",
    "large language models", "large language models (llms)", "language model",
    "language models", "neural network", "neural networks", "deep neural network",
    "pretrained language model", "pretrained language models",
    "pre-trained word embeddings", "word embeddings", "encoder", "decoder",
    "autoregressive language model", "vision language model",
    "vision-language models", "vlm", "vlms", "diffusion model", "diffusion models",
    "foundation model", "foundation models", "chat model", "chat models",
    "instruction-tuned model", "instruction-tuned models", "base model",
    "sparse autoencoder", "sparse autoencoders", "sparse autoencoders (saes)",
}


def _version(text: str) -> str:
    """Family version, rounded to the release tier the field actually talks about.

    Major only, except a `.5` minor which is conventionally its own tier
    (GPT-3.5 and Claude 3.5 are distinct releases; Llama 3.1 and 3.2 are not
    distinct from Llama 3 for the purpose of naming a field norm)."""
    m = re.search(r"(?<![\d.])(\d{1,2})(?:\.(\d))?(?![\d])", text)
    if not m:
        return ""
    major, minor = m.group(1), m.group(2)
    if int(major) > 20 or major == "0":
        return ""                  # a parameter count (7B/70B), or a v0.x point release
    return f"{major}.5" if minor == "5" else major


def canon_model(name: str) -> str:
    """Collapse a checkpoint string to family + release tier.

    Returns "" for an architecture class that should be dropped, and the input
    unchanged when nothing matches — methods and tools must survive intact."""
    raw = (name or "").strip()
    if not raw:
        return ""
    bare = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", raw).strip(" .,;:-")
    if bare.lower() in _GENERIC_MODELS or raw.lower() in _GENERIC_MODELS:
        return ""
    for rx, disp in _FAMILIES:
        if rx.search(raw):
            # Strip everything that looks like scale rather than release: the
            # parenthetical variant list, parameter counts (7B/175M), and MoE
            # shapes (8x7B). What survives is the version, if there is one.
            # A capitalized letter+digit token (DeepSeek-R1, Qwen-V3) names a
            # model LINE, not a release number; a lowercase "v1.3" does.
            scale = re.sub(r"(?<![A-Za-z0-9])[RV]\d+(?![0-9])", " ", bare)
            scale = re.sub(r"\d+\s*x\s*\d+\s*[bm]\b", " ", scale, flags=re.I)
            scale = re.sub(r"\d+\s*[bm]\b", " ", scale, flags=re.I)
            ver = _version(scale)
            return f"{disp}-{ver}" if ver else disp
    # Not a known checkpoint family — a method, a tool, an API. Leave the name
    # alone beyond the generic qualifier strip, so "IKE (In-Context Editing)"
    # and "IKE (In-Context Knowledge Editing)" stop being two methods.
    return canon_generic(raw)


def canon_generic(name: str) -> str:
    """Light canonicalization for datasets/metrics/protocols: drop a trailing
    parenthetical or colon-qualifier so "AdvBench", "AdvBench (subset)" and
    "AdvBench: Harmful Behaviors" collapse. Hyphenated variants are left alone —
    MQuAKE-CF and MQuAKE-T really are different datasets."""
    raw = (name or "").strip()
    base = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*$", "", raw).strip()
    base = re.split(r"\s*:\s+", base)[0].strip() if ":" in base else base
    return (base or raw).strip(" .,;-")


def _key(k: str, name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def aliases(name: str) -> set:
    """Every string this item might also be listed under.

    Papers write a method both ways — "KN (Knowledge Neurons)" in one and
    "Knowledge Neurons (KN)" in the next. Stripping the parenthetical takes the
    disambiguating half off each, leaving "KN" and "Knowledge Neurons" as two
    entries for one method. Keeping both halves as keys lets them find each
    other."""
    raw = (name or "").strip()
    out = {_key("", raw)}
    for m in re.finditer(r"[\(\[]([^\)\]]{1,60})[\)\]]", raw):
        inner = m.group(1).strip()
        # "(6B)", "(higher_better)" and slash-lists are qualifiers, not alternate
        # names for the thing.
        if inner and not re.fullmatch(r"[\d.,\s]*[bmkBMK]?", inner) and "/" not in inner:
            out.add(_key("", inner))
    out.add(_key("", re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*", " ", raw)))
    return {a for a in out if len(a) >= 2}


# --------------------------------------------------------------------------- #
# merge + prune
# --------------------------------------------------------------------------- #
_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ors", "or", "ies",
             "es", "ed", "s")


def _stem(w: str) -> str:
    """Crude suffix strip, enough to see that "editing", "editor" and "edits" are
    the same word. Not linguistics — a dedup key."""
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[:-len(suf)]
            break
    # "baselines" strips to "baselin" while "baseline" strips to nothing, so a
    # trailing e comes off last and both land on the same key.
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def _content(name: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z]{3,}", (name or "").lower())
            if w not in _STOP}


_STOP = {"the", "and", "for", "with", "that", "this", "not", "any", "its", "via",
         "per", "use", "using", "used", "than", "from", "into", "onto", "same",
         "each", "one", "two", "all", "other", "such", "when", "only", "also"}


def _absorb_design(items: List[Dict], threshold: float = 0.6) -> List[Dict]:
    """Collapse design items that say the same thing in different words.

    The reduce step runs once per subfield, so a claim routed to three subfields
    over one shared paper pool produces the same practice three times under three
    phrasings — "Pre-edit model baseline", "Pre-edit (unedited) model baseline",
    "Multiple editing algorithms side-by-side" and "Multiple editor algorithms"
    all appeared on one card, eating six of fourteen slots to say four things.
    Prefix containment alone missed most of them, so items merge on stemmed
    content-word overlap (Jaccard) as well, with the more corroborated phrasing
    kept as the host."""
    out: List[Dict] = []
    for it in sorted(items, key=lambda x: (-len(x["evidence"]), len(x["name"]))):
        mine = _content(it["name"])
        host = None
        for h in out:
            theirs = _content(h["name"])
            if not (mine and theirs):
                continue
            jac = len(mine & theirs) / len(mine | theirs)
            contained = (mine <= theirs or theirs <= mine) and min(len(mine), len(theirs)) >= 2
            if jac >= threshold or contained:
                host = h
                break
        if host is None:
            out.append(it)
            continue
        host["evidence"] = sorted(set(host["evidence"]) | set(it["evidence"]))
        host["subfields"] = sorted(set(host["subfields"]) | set(it["subfields"]))
        host["variants"] = sorted(set(host["variants"]) | {it["name"]})
        for f in ("detail", "rules_out", "ruled_out_by"):
            if it.get(f) and len(str(it[f])) > len(str(host.get(f) or "")):
                host[f] = it[f]     # keep whichever phrasing carries more specifics
    return sorted(out, key=lambda x: (-len(x["evidence"]), x["name"]))


def merge(cards: List[Dict]) -> Dict:
    """Union the per-subfield menus into ONE claim-level menu.

    A claim routed to three subfields used to get three full cards, ~30% of whose
    items were literally the same string — the proposer paid for the duplication
    in context and got nothing for it. Merging keeps each item once and tags it
    with every subfield that vouched for it, which is strictly more information
    than the repetition carried."""
    out = {k: {} for k in ALL_KEYS}
    for card in cards:
        sf = card.get("subfield", "")
        for k in ALL_KEYS:
            for it in (card.get("menu", {}).get(k) or []):
                name = (it.get("name") or "").strip()
                canon = canon_model(name) if k == "models" else canon_generic(name)
                if not canon:
                    continue                      # architecture class: dropped
                # Find an existing slot this is another name for, before making one.
                alias = aliases(name) | {_key(k, canon)}
                slot = next((v for v in out[k].values() if v["_alias"] & alias), None)
                if slot is None:
                    slot = out[k][_key(k, canon)] = {
                        "name": canon, "evidence": set(), "subfields": set(),
                        "variants": set(), "_alias": set(),
                    }
                slot["_alias"] |= alias
                # An expansion names the thing for a reader who does not already know
                # the field, which is exactly who a norm card is for; an acronym does
                # not. So the longer form wins the display name.
                if len(canon) > len(slot["name"]):
                    slot["variants"].add(slot["name"])
                    slot["name"] = canon
                slot["evidence"] |= set(it.get("evidence") or [])
                slot["subfields"].add(sf)
                if name != slot["name"]:
                    slot["variants"].add(name)
                # `rules_out`/`ruled_out_by` are the whole point of a control or a
                # confound — the name alone says what was run, not what it licenses.
                for f in ("role", "direction", "detail", "rules_out", "ruled_out_by"):
                    if it.get(f) and len(str(it[f])) > len(str(slot.get(f) or "")):
                        slot[f] = it[f]
    for k in ALL_KEYS:
        items = []
        for it in (_absorb_design(list(out[k].values())) if k in DESIGN_KEYS
                   else out[k].values()):
            it.pop("_alias", None)
            it["evidence"] = sorted(it["evidence"])
            it["subfields"] = sorted(x for x in it["subfields"] if x)
            it["variants"] = sorted(it["variants"])
            items.append(it)
        out[k] = sorted(items, key=lambda x: (-len(x["evidence"]), x["name"]))
    return out


def _allocate(items: List[Dict], cap: int, subfield_order: List[str]) -> List[Dict]:
    """Fill `cap` slots round-robin over subfields, in the classifier's order.

    Ranking the merged list by evidence count alone lets the BROADEST subfield win
    every slot, because breadth is what puts papers in the pool. Measured on the
    v0.3 cards: the steering claim's primary subfield, Activation Steering, held 2
    of 14 control slots and 1 of 14 confound slots — the rest were jailbreak and
    alignment norms, for a paper about evaluation awareness. Round-robin means the
    primary subfield's most-corroborated norm is in the card before the umbrella
    subfield's second one is, and evidence count still orders within a subfield."""
    if not subfield_order:
        return items[:cap]
    pools = {sf: [it for it in items if sf in (it.get("subfields") or [])]
             for sf in subfield_order}
    out, seen = [], set()
    while len(out) < cap:
        added = False
        for sf in subfield_order:
            while pools[sf]:
                it = pools[sf].pop(0)
                if id(it) in seen:
                    continue
                seen.add(id(it))
                out.append(it)
                added = True
                break
            if len(out) >= cap:
                break
        if not added:
            break
    # Items no subfield claimed (a blank tag) still deserve the leftover slots.
    for it in items:
        if len(out) >= cap:
            break
        if id(it) not in seen:
            out.append(it)
    return out[:cap]


def prune(menu: Dict, min_evidence: int = 2, resource_cap: int = 8,
          design_cap: int = 14, subfield_order: List[str] = None) -> Tuple[Dict, Dict]:
    """Keep the corroborated core; report what was cut and why.

    Resources need `min_evidence` independent papers: a dataset one group used
    once is that group's choice, not the subfield's norm, and it was 59-75% of
    every v0.2 card. Design items (protocols/controls/confounds) are scarce and
    are the part that transfers, so they only face the cap. If corroboration
    would empty a resource list entirely the threshold is relaxed for it — a
    genuinely new subfield may have no repeated practice yet, and that is the
    case the whole project is about."""
    subfield_order = list(subfield_order or [])
    kept, stats = {}, {}
    for k in ALL_KEYS:
        items = menu.get(k) or []
        cap = resource_cap if k in RESOURCE_KEYS else design_cap
        if k in RESOURCE_KEYS:
            core = [it for it in items if len(it["evidence"]) >= min_evidence]
            relaxed = not core and items
            if relaxed:
                core = items
            kept[k] = _allocate(core, cap, subfield_order)
            stats[k] = {"in": len(items), "out": len(kept[k]),
                        "dropped_uncorroborated": len(items) - len(core),
                        "relaxed": bool(relaxed)}
        else:
            kept[k] = _allocate(items, cap, subfield_order)
            stats[k] = {"in": len(items), "out": len(kept[k])}
        # Which subfield each slot came from — the check that the primary subfield
        # is actually represented rather than crowded out.
        by_sf: Dict[str, int] = {}
        for it in kept[k]:
            for sf in (it.get("subfields") or ["(untagged)"]):
                by_sf[sf] = by_sf.get(sf, 0) + 1
        stats[k]["by_subfield"] = by_sf
    stats["totals"] = {"in": sum(len(menu.get(k) or []) for k in ALL_KEYS),
                       "out": sum(len(kept[k]) for k in ALL_KEYS)}
    return kept, stats


def curate(cards: List[Dict], min_evidence: int = 2, resource_cap: int = 8,
           design_cap: int = 14, subfield_order: List[str] = None) -> Tuple[Dict, Dict]:
    """merge -> prune, the whole deterministic half in one call.

    `subfield_order` is the classifier's ranking, most relevant first; slots are
    dealt round-robin over it so the claim's primary subfield is not outvoted by a
    broader neighbour that simply had more papers in the pool."""
    order = list(subfield_order or [c.get("subfield", "") for c in cards])
    return prune(merge(cards), min_evidence, resource_cap, design_cap, order)


def recurate_file(path: str, **kw) -> Dict:
    """Rebuild a card's `card`/`curation` from its stored per-subfield menus.

    Curation is deterministic and the raw menus are kept in the file, so tuning a
    threshold or a family rule costs nothing — no papers refetched, no model
    called. Only the extraction stages need an API key."""
    import json
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    kw.setdefault("subfield_order", doc.get("subfields") or [])
    doc["card"], doc["curation"] = curate(doc.get("norm_cards") or [], **kw)
    doc["format_version"] = "0.3"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc


def _main():
    import argparse
    import glob
    import json

    ap = argparse.ArgumentParser(
        description="Re-run the deterministic curation over existing norm cards.")
    ap.add_argument("--cards", required=True,
                    help="a norm_card.json, or a dir searched for problem_*/norm_card.json")
    ap.add_argument("--min_evidence", type=int, default=2)
    ap.add_argument("--resource_cap", type=int, default=8)
    ap.add_argument("--design_cap", type=int, default=14)
    args = ap.parse_args()

    paths = ([args.cards] if args.cards.endswith(".json")
             else sorted(glob.glob(os.path.join(args.cards, "problem_*", "norm_card.json"))))
    if not paths:
        raise SystemExit(f"no norm_card.json under {args.cards}")
    for p in paths:
        doc = recurate_file(p, min_evidence=args.min_evidence,
                            resource_cap=args.resource_cap, design_cap=args.design_cap)
        t = doc["curation"]["totals"]
        print(f"{doc.get('problem_id', p)}: {t['in']} -> {t['out']}  "
              + ", ".join(f"{k}={len(doc['card'][k])}" for k in ALL_KEYS))


if __name__ == "__main__":
    _main()
