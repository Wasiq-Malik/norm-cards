"""Semantic Scholar source.

Mirrors the existing scimcp/Tools/General_tools.py call (same endpoint/fields)
but returns normalized Paper objects. Works keyless (rate-limited); a key in
SEMANTIC_SCHOLAR_API_KEY raises limits.
"""

from typing import List

from .base import BaseSource, request
from ..models import Paper
from .. import config

_FIELDS = ("title,abstract,citationCount,tldr,fieldsOfStudy,year,authors,"
           "isOpenAccess,openAccessPdf,url,venue,externalIds")


class SemanticScholarSource(BaseSource):
    name = "semantic_scholar"

    def search(self, query: str, limit: int = 10, context: str = "") -> List[Paper]:
        headers = {}
        key = config.semantic_scholar_key()
        if key:
            headers["X-API-KEY"] = key
        params = {"query": query, "limit": limit, "fields": _FIELDS}
        r = request("https://api.semanticscholar.org/graph/v1/paper/search",
                    headers=headers, params=params)
        return [self._to_paper(d) for d in r.json().get("data", [])]

    def _to_paper(self, d: dict) -> Paper:
        ext = d.get("externalIds") or {}
        pdf = (d.get("openAccessPdf") or {}).get("url") or ""
        abstract = d.get("abstract") or ""
        if not abstract and d.get("tldr"):
            abstract = (d["tldr"] or {}).get("text", "") or ""
        return Paper(
            title=d.get("title") or "",
            abstract=abstract,
            authors=[a.get("name", "") for a in (d.get("authors") or [])][:25],
            year=d.get("year"),
            venue=d.get("venue") or "",
            citations=d.get("citationCount") or 0,
            url=d.get("url") or "",
            pdf_url=pdf,
            doi=(ext.get("DOI") or ""),
            arxiv_id=(ext.get("ArXiv") or ""),
            fields_of_study=list(d.get("fieldsOfStudy") or []),
        )
