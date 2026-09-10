"""Tool belt for the evaluation judge.

Every tool is a LOCAL function exposed via function-calling rather than a
provider-native browsing feature. That is deliberate:
  - identical behaviour across model families (sol, luna, anything later),
  - every query and every returned byte is logged and cacheable, so any quoted
    piece of evidence in a verdict can be re-checked by hand later,
  - no dependence on what a provider's built-in search happens to return today.

"Deep search" is emergent: the agent loops search -> fetch -> search as long as it
wants. There is no budget.

Backends, all already used elsewhere in this repo: SerpAPI (web), OpenAlex
(scholarly), the fulltext.py PDF resolver chain + shared .pdfcache (paper bodies),
HuggingFace Hub + Kaggle (resource existence). Papers With Code is deliberately
NOT used for lookup — its free-text endpoint is broken (see sources/pwc.py).
"""

import hashlib
import html
import json
import os
import re
import threading
import time
import warnings
from typing import Dict, List

from .. import config, fulltext
from ..sources.base import request
from ..sources.openalex import OpenAlexSource

PDF_CACHE = ".pdfcache"          # shared with the norm-card pipeline
_TOOL_CACHE_DIR = None           # set by set_cache_dir(); None = no caching

# OpenAlex semantic search is rate-limited to ~1 req/sec (see sources/openalex.py).
# Locked because parallel judge runs share this process.
_last_openalex = [0.0]
_openalex_lock = threading.Lock()


def set_cache_dir(path: str):
    """Enable on-disk caching of tool results (keyed by tool+args)."""
    global _TOOL_CACHE_DIR
    _TOOL_CACHE_DIR = path
    if path:
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _strip_html(raw: str) -> str:
    """HTML -> readable text, stdlib only (repo convention: no extra deps)."""
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\xa0]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    return raw.strip()


def _pdf_text(pdf_bytes: bytes, max_pages: int = 30) -> str:
    """PDF -> text, with a space-recovery retry.

    Some PDFs (notably arXiv papers in certain fonts) extract as
    "77.5%oftheproblems" under pdfplumber's default word tolerance. Verdicts here
    are only as good as the quotes backing them, so when the space ratio comes
    out anomalously low we re-extract with x_tolerance=1.0, which recovers word
    boundaries. Deliberately local to the harness: fulltext.extract_text feeds the
    norm-card pipeline that this harness is measuring, and must not change.
    """
    import io
    import pdfplumber
    warnings.filterwarnings("ignore")

    def _join(pages):
        t = "\n".join(pages)
        t = re.sub(r"[ \t]+", " ", t)
        return re.sub(r"\n{3,}", "\n\n", t).strip()

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages[:max_pages]
            text = _join([pg.extract_text() or "" for pg in pages])
            if text and sum(c == " " for c in text) / len(text) < 0.08:
                text = _join([pg.extract_text(x_tolerance=1.0) or "" for pg in pages])
    except Exception:
        return ""
    return text


def _cached_pdf_text(rec: Dict, max_pages: int = 30):
    """Resolve a paper's OA PDF (same resolver chain and .pdfcache as the
    pipeline) and extract it with _pdf_text. Returns (text, source_tag)."""
    os.makedirs(PDF_CACHE, exist_ok=True)
    h = hashlib.md5((rec.get("title") or "").encode()).hexdigest()[:12]
    path = os.path.join(PDF_CACHE, f"{h}.pdf")
    if os.path.exists(path):
        return _pdf_text(open(path, "rb").read(), max_pages), "cache"
    pdf_bytes, source = fulltext.resolve_pdf(rec)
    if not pdf_bytes:
        return "", source
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return _pdf_text(pdf_bytes, max_pages), source


def _windows(text: str, find: str, width: int = 1200, limit: int = 6) -> List[str]:
    """Context windows around each hit for `find` — how an agent pulls a quote
    out of a 40-page paper without reading (or paying for) the whole thing."""
    out, start = [], 0
    low, needle = text.lower(), find.lower()
    while len(out) < limit:
        i = low.find(needle, start)
        if i == -1:
            break
        a, b = max(0, i - width // 2), min(len(text), i + width // 2)
        out.append(text[a:b].strip())
        start = b
    return out


def _clip(text: str, max_chars: int, find: str = "") -> Dict:
    """Return `text` under a size cap, either as windows around `find` or as a
    head slice. Always reports the true length so the agent knows what it hasn't
    seen and can ask for a different slice."""
    if find:
        hits = _windows(text, find)
        if hits:
            return {"total_chars": len(text), "mode": f"windows around {find!r}",
                    "n_hits": len(hits), "excerpts": hits}
        return {"total_chars": len(text), "mode": f"no match for {find!r}; head slice",
                "text": text[:max_chars]}
    return {"total_chars": len(text), "truncated": len(text) > max_chars,
            "mode": "head slice", "text": text[:max_chars]}


# --------------------------------------------------------------------------- #
# tool implementations
# --------------------------------------------------------------------------- #
def web_search(query: str, n: int = 8) -> Dict:
    """General web search (SerpAPI/Google): docs, leaderboards, dataset cards,
    model release notes — the non-scholarly half of the evidence base."""
    r = request("https://serpapi.com/search",
                params={"engine": "google", "q": query, "num": min(n, 20),
                        "api_key": config.serp_key()})
    j = r.json()
    results = [{"title": x.get("title", ""), "url": x.get("link", ""),
                "snippet": x.get("snippet", "")}
               for x in (j.get("organic_results") or [])[:n]]
    out = {"query": query, "results": results}
    if j.get("answer_box"):
        ab = j["answer_box"]
        out["answer_box"] = {k: ab.get(k) for k in ("title", "answer", "snippet")
                             if ab.get(k)}
    return out


def fetch_page(url: str, find: str = "", max_chars: int = 20000) -> Dict:
    """Read a page found via search. Handles HTML and PDFs. `find` returns
    context windows around a term instead of the head of the document."""
    r = request(url, allow_redirects=True)
    ctype = r.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or r.content[:4] == b"%PDF":
        text = _pdf_text(r.content, max_pages=30)
        kind = "pdf"
    else:
        text = _strip_html(r.text)
        kind = "html"
    return {"url": url, "content_type": kind, **_clip(text, max_chars, find)}


def openalex_search(query: str, n: int = 8, lexical: bool = False) -> Dict:
    """Scholarly search. Default is OpenAlex NATIVE semantic search (embeds the
    query — good recall when wording differs); `lexical=true` is keyword search,
    better when you know an exact title, dataset, or benchmark name."""
    with _openalex_lock:
        wait = 1.1 - (time.time() - _last_openalex[0])
        if wait > 0:
            time.sleep(wait)
        src = OpenAlexSource()
        papers = (src.search_lexical(query, limit=min(n, 25))
                  if lexical else src.search(query, limit=min(n, 25)))
        _last_openalex[0] = time.time()
    return {"query": query, "mode": "lexical" if lexical else "semantic",
            "results": [{"title": p.title, "year": p.year, "venue": p.venue,
                         "citations": p.citations, "doi": p.doi, "url": p.url,
                         "has_pdf": bool(p.pdf_url),
                         "abstract": (p.abstract or "")[:1200]}
                        for p in papers]}


def fetch_paper(paper: str, find: str = "", max_chars: int = 25000) -> Dict:
    """Full text of a paper, for pulling exact quotes. `paper` may be a DOI, an
    arXiv id/url, any URL, or a title (resolved via OpenAlex). Uses the same
    OA-resolver chain and PDF cache as the norm-card pipeline; paywalled-only
    papers legitimately come back empty — say so rather than inventing a quote."""
    rec, resolved_via = {}, "direct"
    p = paper.strip()
    if p.lower().startswith("http"):
        rec = {"title": p, "url": p, "pdf_url": p}
    elif p.lower().startswith("10.") or "doi.org" in p.lower():
        rec = {"title": p, "doi": p.replace("https://doi.org/", "")}
    elif re.fullmatch(r"(arxiv:)?\d{4}\.\d{4,5}(v\d+)?", p, re.I):
        aid = p.lower().replace("arxiv:", "")
        rec = {"title": p, "url": f"https://arxiv.org/abs/{aid}",
               "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf"}
    else:
        hits = OpenAlexSource().search_lexical(p, limit=1)
        if not hits:
            return {"paper": p, "error": "could not resolve this paper via OpenAlex",
                    "text": ""}
        h = hits[0]
        rec = {"title": h.title, "doi": h.doi, "url": h.url, "pdf_url": h.pdf_url}
        resolved_via = f"openalex title match: {h.title!r} ({h.year})"

    text, src = _cached_pdf_text(rec, max_pages=30)
    if not text:
        return {"paper": p, "resolved_via": resolved_via, "pdf_source": src,
                "error": "no open-access full text available for this paper",
                "text": ""}
    return {"paper": p, "resolved_via": resolved_via, "pdf_source": src,
            "title": rec.get("title", ""), **_clip(text, max_chars, find)}


def _hf(kind: str, name: str) -> Dict:
    """HuggingFace Hub: exact lookup when name is org/name, plus a search."""
    out = {}
    if "/" in name:
        try:
            d = request(f"https://huggingface.co/api/{kind}/{name}").json()
            out["exact"] = {"id": d.get("id"), "downloads": d.get("downloads"),
                            "likes": d.get("likes"), "gated": d.get("gated"),
                            "private": d.get("private")}
        except Exception as e:
            out["exact"] = {"error": f"not found as {kind}/{name} ({e.__class__.__name__})"}
    try:
        rs = request(f"https://huggingface.co/api/{kind}",
                     params={"search": name, "limit": 8, "full": "false"}).json()
        out["search"] = [{"id": d.get("id"), "downloads": d.get("downloads"),
                          "likes": d.get("likes")} for d in rs]
    except Exception as e:
        out["search"] = {"error": config.redact(e)[:200]}
    return out


def check_resource(name: str, kind: str = "any") -> Dict:
    """Does a named dataset / model actually exist and is it publicly available?

    The groundedness workhorse: an experiment that runs on a dataset nobody can
    obtain is not runnable. Checks HuggingFace (and Kaggle if credentials are
    configured).

    NOTE the caveat this returns: plenty of real, standard resources (COCO
    val2017, GuacaMol, in-house benchmarks) are not hosted on HF under the name a
    paper uses. A miss here is WEAK evidence of non-existence and must be
    corroborated with web_search / openalex_search before concluding anything.
    """
    res = {"name": name, "kind": kind}
    if kind in ("any", "dataset", "benchmark"):
        res["huggingface_datasets"] = _hf("datasets", name)
    if kind in ("any", "model"):
        res["huggingface_models"] = _hf("models", name)

    user, key = config.get_key("KAGGLE_USERNAME"), config.get_key("KAGGLE_KEY")
    if user and key and kind in ("any", "dataset", "benchmark"):
        try:
            r = request("https://www.kaggle.com/api/v1/datasets/list",
                        params={"search": name}, auth=(user, key))
            res["kaggle"] = [{"ref": d.get("ref"), "title": d.get("title")}
                             for d in r.json()[:8]]
        except Exception as e:
            res["kaggle"] = {"error": config.redact(e)[:200]}
    else:
        res["kaggle"] = "not checked (no Kaggle credentials configured)"

    res["caveat"] = ("A miss on these hubs is WEAK evidence of non-existence — many "
                     "standard resources are distributed elsewhere or under a different "
                     "name. Corroborate with web_search/openalex_search before judging "
                     "a resource unavailable.")
    return res


# --------------------------------------------------------------------------- #
# registry / dispatch
# --------------------------------------------------------------------------- #
_IMPL = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "openalex_search": openalex_search,
    "fetch_paper": fetch_paper,
    "check_resource": check_resource,
}

SPECS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web (Google). Use for documentation, leaderboards, "
                       "dataset/model cards, release notes, blog posts — anything "
                       "non-scholarly. Returns titles, urls, snippets; follow up with "
                       "fetch_page to read and quote a result.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "n": {"type": "integer", "description": "results to return (default 8)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "Read a web page or PDF at a URL and return its text. Pass `find` "
                       "to get context windows around a term instead of the head of a "
                       "long document — this is how you locate an exact quote.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "find": {"type": "string", "description": "term to locate within the page"},
            "max_chars": {"type": "integer"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "openalex_search",
        "description": "Search the scholarly literature (OpenAlex). Semantic by default "
                       "(finds conceptually similar work even with different wording); "
                       "set lexical=true for exact titles/benchmark names or to find "
                       "surveys. Returns title, year, venue, citations, DOI, abstract.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "n": {"type": "integer"},
            "lexical": {"type": "boolean"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_paper",
        "description": "Get the full text of a paper by DOI, arXiv id/url, URL, or title, "
                       "so you can quote it exactly. Pass `find` to jump to the relevant "
                       "passage (e.g. 'val2017', 'attack success rate'). Returns empty "
                       "text for papers with no open-access copy — in that case say so, "
                       "never paraphrase a paper you could not read.",
        "parameters": {"type": "object", "properties": {
            "paper": {"type": "string", "description": "DOI, arXiv id, URL, or title"},
            "find": {"type": "string"},
            "max_chars": {"type": "integer"}},
            "required": ["paper"]}}},
    {"type": "function", "function": {
        "name": "check_resource",
        "description": "Check whether a named dataset or model actually exists and is "
                       "publicly obtainable (HuggingFace, Kaggle). Use on every concrete "
                       "resource an experiment names. A miss is weak evidence — "
                       "corroborate with web_search before calling something nonexistent.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["any", "dataset", "model", "benchmark"]}},
            "required": ["name"]}}},
]


def _cache_path(name: str, args: Dict) -> str:
    key = hashlib.sha1(
        (name + json.dumps(args, sort_keys=True)).encode()).hexdigest()[:16]
    return os.path.join(_TOOL_CACHE_DIR, f"{name}.{key}.json")


def dispatch(name: str, args: Dict) -> str:
    """Run a tool and return its result as a JSON string for the model.

    Errors are returned as data, not raised: a 404 or a rate-limited backend is
    something the agent should see and route around, not a crashed evaluation.
    """
    fn = _IMPL.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name!r}"})

    path = _cache_path(name, args) if _TOOL_CACHE_DIR else None
    if path and os.path.exists(path):
        return open(path, encoding="utf-8").read()

    try:
        out = fn(**args)
    except TypeError as e:                       # bad/missing arguments
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except Exception as e:                       # network/backend failure
        out = {"error": f"{type(e).__name__}: {config.redact(e)[:300]}",
               "note": "tool call failed; try a different query, source, or tool"}

    js = json.dumps(out, ensure_ascii=False)
    if path:
        # Atomic: judge runs execute in parallel threads and can race on the same
        # cache key, and a half-written file would poison later runs.
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(js)
        os.replace(tmp, path)
    return js
