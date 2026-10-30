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

# Format 0.4 (2026-10-18). Built from the misses of the 0.3 cards: the card usually
# had the right KIND of control but stated too generally to apply, the measurement's
# own soundness and the field's bar for "this difference is real" had no home, and
# resources took a third of the slots while mattering least. So controls and
# confounds become one `threats` list tagged with the kinds of claim each applies
# to, two design keys are added, `models` becomes `baselines` with what a fair
# comparison must match, and the resource caps shrink.
RESOURCE_KEYS_V04 = ("datasets", "baselines", "metrics")
DESIGN_KEYS_V04 = ("protocols", "threats", "validity_checks", "evidence_standards")
ALL_KEYS_V04 = RESOURCE_KEYS_V04 + DESIGN_KEYS_V04
CAPS_V04 = {"threats": 14, "validity_checks": 6, "evidence_standards": 6, "protocols": 8,
            "baselines": 6, "metrics": 6, "datasets": 4}

# Format 2 (2026-10-29, dataset50). protocols and validity_checks go: ~80% of protocols
# restated a threat's rule-out or a validity check, and the rest (benchmark recipes such
# as "LIBERO: 50 episodes per task") move onto the dataset as `eval_protocol`; the few
# validity checks worth keeping move onto the metric as `caveat`. Threats are 0.3's
# controls and confounds, extracted and consolidated with the 0.3 wording and tagged
# `kind`, with 20 slots instead of 0.4's 14 (0.3 cards carried ~26 of the two; 0.4 halved
# the room and confounds slipped). No `applies_when` tags.
RESOURCE_KEYS_V2 = ("datasets", "baselines", "metrics")
DESIGN_KEYS_V2 = ("threats", "evidence_standards")
ALL_KEYS_V2 = RESOURCE_KEYS_V2 + DESIGN_KEYS_V2
CAPS_V2 = {"threats": 20, "evidence_standards": 7, "baselines": 8, "metrics": 8, "datasets": 7}
THREAT_KINDS = ("control", "confound")

# The kinds of claim a threat can be tagged with. Fixed before any 0.4 card was built
# and deliberately generic: a tag names a TYPE of claim, never a specific one, so a
# card stays claim-blind and matching a tag to the claim at hand is the proposer's job.
CLAIM_TYPES = (
    "intervention on model internals",   # steering, editing, ablation, patching
    "method comparison",                 # a method beats existing ones
    "robustness or attack",              # behaviour under adversarial or corrupted input
    "generalisation or transfer",        # holds on unseen data, tasks, settings
    "detection or classification",       # a system identifies a property or class
    "benchmark or measurement",          # an evaluation measures what it says it does
    "agent or embodied task success",    # a policy or agent completes tasks
    "training-signal effect",            # a change to data, reward or recipe causes a change
    "any empirical claim",
)


def keys_for(fmt: str):
    """(resource keys, design keys) for a card format."""
    f = str(fmt)
    if f.startswith("2"):
        return RESOURCE_KEYS_V2, DESIGN_KEYS_V2
    return ((RESOURCE_KEYS_V04, DESIGN_KEYS_V04) if f.startswith("0.4")
            else (RESOURCE_KEYS, DESIGN_KEYS))


def format_of(menu: Dict) -> str:
    """Infer the format from the keys a menu or card actually carries."""
    m = menu or {}
    if "threats" in m and not any(k in m for k in ("validity_checks", "protocols")):
        return "2"
    return "0.4" if any(k in m for k in ("threats", "baselines")) else "0.3"


def keys_of(menu: Dict) -> Tuple[tuple, tuple]:
    return keys_for(format_of(menu))


# Free-text fields a merge keeps the most specific phrasing of, across both formats.
_TEXT_FIELDS = ("role", "direction", "detail", "rules_out", "ruled_out_by", "rule_out",
                "fair_comparison", "instrument", "misleads_when", "check",
                "eval_protocol", "caveat")


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
        for f in _TEXT_FIELDS:
            if it.get(f) and len(str(it[f])) > len(str(host.get(f) or "")):
                host[f] = it[f]     # keep whichever phrasing carries more specifics
        if "applies_when" in it or "applies_when" in host:
            host["applies_when"] = _union_types(host.get("applies_when"),
                                                it.get("applies_when"))
    return sorted(out, key=lambda x: (-len(x["evidence"]), x["name"]))


def _union_types(*lists) -> List[str]:
    """Claim-type tags, restricted to CLAIM_TYPES and kept in that list's order."""
    got = set()
    for xs in lists:
        got |= {str(t).strip().lower() for t in ([xs] if isinstance(xs, str) else xs or [])}
    return [t for t in CLAIM_TYPES if t in got]


def merge(cards: List[Dict], keys: tuple = None) -> Dict:
    """Union the per-subfield menus into ONE claim-level menu.

    A claim routed to three subfields used to get three full cards, ~30% of whose
    items were literally the same string — the proposer paid for the duplication
    in context and got nothing for it. Merging keeps each item once and tags it
    with every subfield that vouched for it, which is strictly more information
    than the repetition carried."""
    if keys is None:
        res, des = keys_of(next((c.get("menu") for c in cards if c.get("menu")), {}))
        keys = res + des
    design = set(DESIGN_KEYS) | set(DESIGN_KEYS_V04) | set(DESIGN_KEYS_V2)
    out = {k: {} for k in keys}
    for card in cards:
        sf = card.get("subfield", "")
        for k in keys:
            for it in (card.get("menu", {}).get(k) or []):
                name = (it.get("name") or "").strip()
                canon = (canon_model(name) if k in ("models", "baselines")
                         else canon_generic(name))
                if not canon:
                    continue                      # architecture class: dropped
                # Find an existing slot this is another name for, before making one.
                # A control and a confound never merge: the kind is part of the key.
                kind = it.get("kind") or ""
                alias = {kind + a for a in aliases(name) | {_key(k, canon)}}
                slot = next((v for v in out[k].values() if v["_alias"] & alias), None)
                if slot is None:
                    slot = out[k][kind + _key(k, canon)] = {
                        "name": canon, "evidence": set(), "subfields": set(),
                        "variants": set(), "_alias": set(),
                        **({"kind": kind} if kind else {}),
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
                for f in _TEXT_FIELDS:
                    if it.get(f) and len(str(it[f])) > len(str(slot.get(f) or "")):
                        slot[f] = it[f]
                if it.get("applies_when") is not None or "applies_when" in slot:
                    slot["applies_when"] = _union_types(slot.get("applies_when"),
                                                        it.get("applies_when"))
    for k in keys:
        items = []
        vals = list(out[k].values())
        if k in design:
            kinds = list(dict.fromkeys(v.get("kind", "") for v in vals))
            vals = [x for kd in kinds for x in _absorb_design([v for v in vals
                                                               if v.get("kind", "") == kd])]
        for it in vals:
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


def _allocate_kinds(items: List[Dict], cap: int, subfield_order: List[str]) -> List[Dict]:
    """Half the threat slots to controls and half to confounds, each dealt round-robin
    over subfields; a kind short of its half gives the rest to the other. Controls are
    listed first, as 0.3 cards showed them."""
    pools = {kd: [it for it in items if it.get("kind") == kd] for kd in THREAT_KINDS}
    half = {kd: _allocate(pools[kd], cap // len(THREAT_KINDS), subfield_order)
            for kd in THREAT_KINDS}
    spare = cap - sum(len(v) for v in half.values())
    for kd in THREAT_KINDS:
        if spare <= 0:
            break
        more = [it for it in _allocate(pools[kd], len(pools[kd]), subfield_order)
                if all(it is not x for x in half[kd])][:spare]
        half[kd] += more
        spare -= len(more)
    return [it for kd in THREAT_KINDS for it in half[kd]]


def prune(menu: Dict, min_evidence: int = 2, resource_cap: int = 8,
          design_cap: int = 14, subfield_order: List[str] = None,
          caps: Dict[str, int] = None) -> Tuple[Dict, Dict]:
    """Keep the corroborated core; report what was cut and why.

    Resources need `min_evidence` independent papers: a dataset one group used
    once is that group's choice, not the subfield's norm, and it was 59-75% of
    every v0.2 card. Design items (protocols/controls/confounds) are scarce and
    are the part that transfers, so they only face the cap. If corroboration
    would empty a resource list entirely the threshold is relaxed for it — a
    genuinely new subfield may have no repeated practice yet, and that is the
    case the whole project is about."""
    subfield_order = list(subfield_order or [])
    fmt = format_of(menu)
    res_keys, des_keys = keys_for(fmt)
    if caps is None and fmt != "0.3":
        caps = CAPS_V2 if fmt == "2" else CAPS_V04   # per-key caps, not two numbers
    kept, stats = {}, {}
    for k in res_keys + des_keys:
        items = menu.get(k) or []
        cap = (caps or {}).get(k) or (resource_cap if k in res_keys else design_cap)
        if k in res_keys:
            core = [it for it in items if len(it["evidence"]) >= min_evidence]
            relaxed = not core and items
            if relaxed:
                core = items
            kept[k] = _allocate(core, cap, subfield_order)
            stats[k] = {"in": len(items), "out": len(kept[k]),
                        "dropped_uncorroborated": len(items) - len(core),
                        "relaxed": bool(relaxed)}
        elif fmt == "2" and k == "threats":
            kept[k] = _allocate_kinds(items, cap, subfield_order)
            stats[k] = {"in": len(items), "out": len(kept[k]),
                        **{kd: sum(it.get("kind") == kd for it in kept[k]) for kd in THREAT_KINDS}}
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
    stats["totals"] = {"in": sum(len(menu.get(k) or []) for k in res_keys + des_keys),
                       "out": sum(len(kept[k]) for k in res_keys + des_keys)}
    return kept, stats


def curate(cards: List[Dict], min_evidence: int = 2, resource_cap: int = 8,
           design_cap: int = 14, subfield_order: List[str] = None,
           caps: Dict[str, int] = None) -> Tuple[Dict, Dict]:
    """merge -> prune, the whole deterministic half in one call.

    `subfield_order` is the classifier's ranking, most relevant first; slots are
    dealt round-robin over it so the claim's primary subfield is not outvoted by a
    broader neighbour that simply had more papers in the pool."""
    order = list(subfield_order or [c.get("subfield", "") for c in cards])
    return prune(merge(cards), min_evidence, resource_cap, design_cap, order, caps)


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
    doc["format_version"] = format_of(doc["card"])
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
              + ", ".join(f"{k}={len(doc['card'][k])}" for k in sum(keys_of(doc["card"]), ())))


if __name__ == "__main__":
    _main()
