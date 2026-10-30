"""Full-text acquisition for norm-card extraction.

Norm-card generation reads the experimental-setup sections of a bundle's papers
(datasets, metrics, protocols live there, not in the abstract). This module
resolves an open-access PDF for a paper and extracts its text.

Resolver chain (first hit wins): arXiv PDF -> the record's own pdf_url ->
Unpaywall (DOI -> best OA location) -> Semantic Scholar (DOI -> openAccessPdf).
All free; a Semantic Scholar key raises its rate limit. Paywalled-only papers
(no OA anywhere) return "" — the caller proceeds with whatever full text it got
(~19/25 per bundle in practice) plus title/abstract for the rest.
"""

import os
import re
import time
import bisect
import hashlib
import warnings
from typing import Dict, Optional, Tuple

from . import config

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "scify-norm-cards/0.2 (research)")


def _requests():
    import requests
    return requests


def _get(url, **kw):
    try:
        return _requests().get(url, headers={"User-Agent": _UA}, timeout=config.HTTP_TIMEOUT,
                               allow_redirects=True, **kw)
    except Exception:
        return None


def _is_pdf(r) -> bool:
    return r is not None and r.status_code == 200 and (
        "pdf" in r.headers.get("Content-Type", "").lower() or r.content[:4] == b"%PDF")


def _try_pdf(url: Optional[str]) -> Optional[bytes]:
    if not url:
        return None
    r = _get(url)
    return r.content if _is_pdf(r) else None


def _unpaywall(doi: str) -> Optional[bytes]:
    if not doi:
        return None
    r = _get(f"https://api.unpaywall.org/v2/{doi}", params={"email": config.OPENALEX_MAILTO})
    if r is None or r.status_code != 200:
        return None
    j = r.json()
    locs = ([j.get("best_oa_location")] if j.get("best_oa_location") else []) + \
           (j.get("oa_locations") or [])
    for loc in locs:
        b = _try_pdf((loc or {}).get("url_for_pdf"))
        if b:
            return b
    return None


def _semantic_scholar(doi: str) -> Optional[bytes]:
    if not doi:
        return None
    headers = {}
    key = config.semantic_scholar_key()
    if key:
        headers["X-API-KEY"] = key
    try:
        r = _requests().get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                            params={"fields": "openAccessPdf"}, headers=headers,
                            timeout=config.HTTP_TIMEOUT)
    except Exception:
        return None
    if r.status_code == 429:
        time.sleep(3)
        return None
    if r.status_code != 200:
        return None
    return _try_pdf(((r.json() or {}).get("openAccessPdf") or {}).get("url"))


def resolve_pdf(paper: Dict) -> Tuple[Optional[bytes], str]:
    """Return (pdf_bytes, source_tag) for a paper dict, or (None, 'none')."""
    doi = (paper.get("doi") or "").replace("https://doi.org/", "").strip()
    pu, url = paper.get("pdf_url") or "", paper.get("url") or ""
    # An explicit arxiv_id is checked FIRST and on its own. Callers that know the
    # id and nothing else — transcribe.py, make_claims.py — used to fall straight
    # through to Unpaywall with an empty DOI and give up, so every claim in a
    # 16-paper batch was silently drafted from its abstract instead of full text.
    ax = re.sub(r"v\d+$", "", str(paper.get("arxiv_id") or "").strip())
    if not ax:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", f"{pu} {url}")
        ax = m.group(1) if m else ""
    if ax:
        b = _try_pdf(f"https://arxiv.org/pdf/{ax}.pdf") or _try_pdf(f"https://arxiv.org/pdf/{ax}")
        if b:
            return b, "arxiv"
    b = _try_pdf(pu)
    if b:
        return b, "pdf_url"
    b = _unpaywall(doi)
    if b:
        return b, "unpaywall"
    b = _semantic_scholar(doi)
    if b:
        return b, "s2"
    return None, "none"


# Version 2 (2026-10-28). Version 1 used pdfplumber's default x_tolerance of 3 pt, which
# is wider than the inter-word gap of most LaTeX papers, so words ran together
# ("Detectingsuchfailuresduringexecution") and the text cost two to three times the
# tokens to read. It also read two-column pages straight across, interleaving the
# columns line by line. Bumping the version changes the text-cache key, so text cached
# by version 1 (and anything built from it) is left as it was.
EXTRACT_VERSION = 2
_X_TOL = 1


def _page_text(pg) -> str:
    """One page's text, reading a two-column page left column first. A page counts as
    two-column when almost no words straddle the vertical midline and both halves hold
    a real share of the words; title blocks and wide figures then come out split, which
    costs far less than interleaving every line of the body."""
    words = pg.extract_words(x_tolerance=_X_TOL)
    mid = pg.width / 2
    if words:
        cross = sum(1 for w in words if w["x0"] < mid - 4 and w["x1"] > mid + 4)
        left = sum(1 for w in words if w["x1"] <= mid + 4)
        right = sum(1 for w in words if w["x0"] >= mid - 4)
        n = len(words)
        if cross <= 0.03 * n and left > 0.2 * n and right > 0.2 * n:
            halves = [pg.crop((0, 0, mid, pg.height)), pg.crop((mid, 0, pg.width, pg.height))]
            return "\n".join(h.extract_text(x_tolerance=_X_TOL) or "" for h in halves)
    return pg.extract_text(x_tolerance=_X_TOL) or ""


def extract_text(pdf_bytes: bytes, max_pages: int = 18) -> str:
    """PDF bytes -> cleaned text (first `max_pages`; setup details are early,
    long appendices add noise/cost)."""
    import io
    import pdfplumber
    warnings.filterwarnings("ignore")
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [_page_text(pg) for pg in pdf.pages[:max_pages]]
    except Exception:
        return ""
    t = "\n".join(pages)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# Whole paper, appendices included. Anything that reads a paper as its SOURCE OF TRUTH -
# transcribing reference experiments, drafting a claim - must use this. The 18-page
# default below exists for card building, where hundreds of papers are skimmed for norms;
# inherited by transcription, it silently cut appendix experiments out of the references
# (dataset20v3 saw as little as 46% of a paper).
FULL_PAPER = 200


def fetch_fulltext(paper: Dict, cache_dir: str, max_pages: int = 18,
                   min_chars: int = 500) -> Tuple[str, str]:
    """Resolve + extract, caching BOTH the PDF and its extracted text. Returns
    (text, source); text is "" if unfetchable or too short to be a real body.

    The text cache matters more than the PDF one. Downloading is a few hundred
    milliseconds; pdfplumber on a 20-page paper is one to three seconds, and the
    same papers are re-extracted by every stage that touches them — the card
    build, each retrieval arm, transcription. Across one evening that was roughly
    2,500 extractions of ~500 distinct PDFs, nearly all of them redundant.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5((paper.get("title") or "").encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"{h}.pdf")
    # Keyed on max_pages: a 30-page extract is not a valid answer for an 18-page
    # request, and silently returning one would change what a stage reads.
    # ...and on the extractor version, so a better extractor never silently mixes with
    # text an older one produced.
    ver = "" if EXTRACT_VERSION == 1 else f".x{EXTRACT_VERSION}"
    tpath = os.path.join(cache_dir, f"{h}{ver}.p{max_pages}.txt")
    if os.path.exists(tpath):
        try:
            cached = open(tpath, encoding="utf-8").read()
            if len(cached) >= min_chars:
                return cached, "cache:text"
        except Exception:
            pass
    source = "cache"
    if os.path.exists(path):
        pdf_bytes = open(path, "rb").read()
    else:
        pdf_bytes, source = resolve_pdf(paper)
        if not pdf_bytes:
            return "", source
        open(path, "wb").write(pdf_bytes)
    text = extract_text(pdf_bytes, max_pages=max_pages)
    if len(text) >= min_chars:
        try:
            with open(tpath, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass                      # a cache that cannot be written is not an error
        return text, source
    return "", source


# What the norm card and the rag arm read: the paper's MAIN BODY, not the whole PDF. A
# 100-page paper is mostly appendix, and reading all of it for 500 bundle papers costs more
# than it returns; the setup, baselines and controls a card is made of sit in the body.
# BODY_PAGES bounds the extraction, and with it the cost, for PDFs whose reference list
# cannot be found.
BODY_PAGES = 40
_YEAR = re.compile(r"(?<!\d)(?:19[5-9]\d|20[0-3]\d)[a-z]?(?!\d)")
# Capitalised only: "references" in lowercase is body prose ("resolve schema references").
_REFS_WORD = re.compile(r"Refer\s?ences|REFER\s?ENCES|Bibliograph|BIBLIOGRAPH")


def main_body(text: str, min_frac: float = 0.3, window: int = 3000,
              min_years: int = 10, min_years_unlabelled: int = 22) -> str:
    """Text up to where the reference list starts, or the whole text if none is found.

    The list starts at a "References"/"Bibliography" word (from `min_frac` of the paper
    on) followed by a run of publication years — 10+ in the next 3,000 chars. A heading
    alone is unreliable: two-column PDFs merge it into a line of the other column
    ("References ECCV 2022 ...") and some lose their word spaces, so it is matched as a
    word, not as a line. Years alone are unreliable too: an author-year body ("(Katz,
    1995)") reaches 10+, so a run with no such word must be denser still (22+). Whatever
    follows the references — the appendix — goes with them."""
    floor = int(len(text) * min_frac)
    years = [m.start() for m in _YEAR.finditer(text)]

    def n_years(a, b):
        return bisect.bisect_left(years, b) - bisect.bisect_left(years, a)
    for m in _REFS_WORD.finditer(text, floor):
        if n_years(m.start(), m.start() + window) >= min_years:
            return text[:m.start()].rstrip()
    for pos in years:
        if pos >= floor and n_years(pos, pos + window) >= min_years_unlabelled:
            return text[:pos].rstrip()
    return text


def fetch_body(paper: Dict, cache_dir: str, html: bool = True) -> Tuple[str, str]:
    """The paper's main body: arXiv HTML with the appendices dropped when arXiv has it,
    else the PDF cut by main_body. ("", source) when there is no full text.

    HTML first from dataset50 (2026-10-29); dataset34v8 and earlier cards and rag arms
    read the PDF only (html=False reproduces them). A bundle mixes the two sources: the
    foundational papers in it often predate arXiv's HTML rendering."""
    if html:
        ax = arxiv_id_of(paper)
        if ax:
            text = fetch_arxiv_html_body(ax, cache_dir)
            if text:
                return text, "arxiv_html"
    text, src = fetch_fulltext(paper, cache_dir, max_pages=BODY_PAGES)
    return (main_body(text) if text else ""), src


_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.I)


def arxiv_id_of(paper: Dict) -> str:
    """A bundle paper's arXiv id, from its own field, its PDF or landing URL, or an arXiv
    DOI; "" when it has none. OpenAlex leaves `arxiv_id` empty for most arXiv papers and
    only the PDF URL says where it lives (396 of 510 dataset34v8 bundle papers)."""
    ax = paper.get("arxiv_id") or (paper.get("externalIds") or {}).get("ArXiv") or ""
    if not ax:
        for f in ("pdf_url", "url", "oa_url", "landing_page_url"):
            m = _ARXIV_URL.search(str(paper.get(f) or ""))
            if m:
                ax = m.group(1)
                break
    if not ax:
        m = _ARXIV_DOI.search(str(paper.get("doi") or ""))
        ax = m.group(1) if m else ""
    return re.sub(r"v\d+$", "", str(ax).strip())



# arXiv's HTML rendering (LaTeXML, built from the LaTeX source) is the cleanest text a
# paper has: no lost word spaces, no interleaved columns, real section headings, tables
# as tables and equations as their LaTeX. Use it as the source of truth for reading a
# paper (claims, references) whenever arXiv has one; PDF extraction is the fallback.
_SKIP_CLASSES = ("ltx_bibliography", "ltx_page_navbar", "ltx_page_footer", "ltx_dates",
                 "ltx_authors", "ltx_role_affiliationtext", "ltx_TOC", "ltx_page_header")


_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "param", "source", "track", "wbr"}


def html_to_text(html: str, skip_classes: tuple = _SKIP_CLASSES) -> str:
    """LaTeXML HTML -> plain text: headings as '## ', table rows as '|'-joined cells,
    math as its LaTeX alttext, bibliography and page chrome dropped (and anything else
    in `skip_classes`)."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out, self.stack, self.skip, self.math, self.cell = [], [], 0, 0, 0

        def handle_starttag(self, tag, attrs):
            if tag in _VOID:              # never closed, so never on the stack
                if tag == "br" and not self.skip and not self.cell:
                    self.out.append("\n")
                return
            a = dict(attrs)
            cls = set((a.get("class") or "").split())   # whole class names, not substrings
            skip = (tag in ("script", "style", "nav", "header", "footer", "button")
                    or bool(cls & set(skip_classes)))
            self.stack.append((tag, skip))
            if skip:
                self.skip += 1
            if self.skip:
                return
            if tag == "math":
                self.math += 1
                alt = a.get("alttext")
                if alt and self.math == 1:
                    self.out.append(f" {alt} ")
            elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self.out.append("\n\n## ")
            elif tag in ("td", "th"):
                self.cell += 1
                self.out.append(" | ")
            elif tag in ("p", "div", "figcaption", "li", "tr") and not self.cell:
                self.out.append("\n")              # a cell's paragraphs stay on its row
            elif tag == "tr":
                self.out.append("\n")

        def handle_endtag(self, tag):
            while self.stack:
                t, skip = self.stack.pop()
                if skip:
                    self.skip -= 1
                if t == "math" and not skip and self.math:
                    self.math -= 1
                if t in ("td", "th") and not skip and self.cell:
                    self.cell -= 1
                if t == tag:
                    break
            if not self.skip and not self.cell and tag in ("h1", "h2", "h3", "h4", "h5", "h6",
                                                           "p", "table"):
                self.out.append("\n")

        def handle_data(self, data):
            if not self.skip and not self.math:
                # line breaks come from the tags; newlines in the source are just layout
                self.out.append(re.sub(r"\s+", " ", data))

    # Only the paper itself: the page around it (toolbar, report dialog) has its own
    # headers, buttons and unclosed tags.
    m = re.search(r'<article class="ltx_document.*</article>', html, re.S)
    p = P()
    p.feed(m.group(0) if m else html)
    t = "".join(p.out)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def fetch_arxiv_html(arxiv_id: str, cache_dir: str, min_chars: int = 2000) -> str:
    """Text of arxiv.org/html/<id>, cached; "" when arXiv has no HTML for the paper."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"html_{arxiv_id.replace('/', '_')}.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    try:
        r = _requests().get(f"https://arxiv.org/html/{arxiv_id}", timeout=config.HTTP_TIMEOUT * 2,
                         headers={"User-Agent": _UA})
    except Exception:
        return ""
    if r.status_code != 200 or "ltx_document" not in r.text:
        return ""
    text = html_to_text(r.text)
    if len(text) < min_chars:
        return ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# LaTeXML puts each appendix in its own section of this class, after the bibliography.
_APPENDIX_CLASSES = ("ltx_appendix",)


def fetch_arxiv_html_body(arxiv_id: str, cache_dir: str, min_chars: int = 2000) -> str:
    """fetch_arxiv_html without the appendices: the main body, as fetch_body reads it."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"html_{arxiv_id.replace('/', '_')}.body.txt")
    miss = path + ".none"            # arXiv has no HTML for it: do not ask again
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    if os.path.exists(miss):
        return ""
    try:
        r = _requests().get(f"https://arxiv.org/html/{arxiv_id}", timeout=config.HTTP_TIMEOUT * 2,
                         headers={"User-Agent": _UA})
    except Exception:
        return ""                    # a network error is not a verdict; retry next time
    time.sleep(1)                    # hundreds of these per card build; stay polite
    text = (html_to_text(r.text, _SKIP_CLASSES + _APPENDIX_CLASSES)
            if r.status_code == 200 and "ltx_document" in r.text else "")
    if len(text) < min_chars:
        if r.status_code in (200, 404):
            open(miss, "w").close()
        return ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def fetch_paper(paper: Dict, cache_dir: str) -> Tuple[str, str]:
    """The whole paper as a source of truth: arXiv HTML when it exists, else the full PDF
    (appendices included). Returns (text, source)."""
    ax = paper.get("arxiv_id") or (paper.get("externalIds") or {}).get("ArXiv") or ""
    if ax:
        text = fetch_arxiv_html(ax, cache_dir)
        if text:
            return text, "arxiv_html"
    return fetch_fulltext(paper, cache_dir, max_pages=FULL_PAPER)
