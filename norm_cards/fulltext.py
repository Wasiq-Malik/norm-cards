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
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", f"{pu} {url}")
    if m:
        b = _try_pdf(f"https://arxiv.org/pdf/{m.group(1)}.pdf")
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


def extract_text(pdf_bytes: bytes, max_pages: int = 18) -> str:
    """PDF bytes -> cleaned text (first `max_pages`; setup details are early,
    long appendices add noise/cost)."""
    import io
    import pdfplumber
    warnings.filterwarnings("ignore")
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [pg.extract_text() or "" for pg in pdf.pages[:max_pages]]
    except Exception:
        return ""
    t = "\n".join(pages)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def fetch_fulltext(paper: Dict, cache_dir: str, max_pages: int = 18,
                   min_chars: int = 500) -> Tuple[str, str]:
    """Resolve + extract, caching the PDF by title hash. Returns (text, source);
    text is "" if unfetchable or too short to be a real body."""
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5((paper.get("title") or "").encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"{h}.pdf")
    source = "cache"
    if os.path.exists(path):
        pdf_bytes = open(path, "rb").read()
    else:
        pdf_bytes, source = resolve_pdf(paper)
        if not pdf_bytes:
            return "", source
        open(path, "wb").write(pdf_bytes)
    text = extract_text(pdf_bytes, max_pages=max_pages)
    return (text, source) if len(text) >= min_chars else ("", source)
