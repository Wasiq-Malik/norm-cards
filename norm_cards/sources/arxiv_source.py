"""arXiv source: keyless preprint search via the public Atom API.

Uses the stdlib XML parser to avoid a feedparser dependency.
"""

import re
import xml.etree.ElementTree as ET
from typing import List

from .base import BaseSource, request
from ..models import Paper
from .. import config

_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivSource(BaseSource):
    name = "arxiv"

    def search(self, query: str, limit: int = 10, context: str = "") -> List[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        r = request("http://export.arxiv.org/api/query", params=params)
        root = ET.fromstring(r.text)
        papers = []
        for entry in root.findall(f"{_ATOM}entry"):
            papers.append(self._to_paper(entry))
        return papers

    def _to_paper(self, entry) -> Paper:
        def text(tag):
            el = entry.find(f"{_ATOM}{tag}")
            return (el.text or "").strip() if el is not None else ""

        abs_url = text("id")
        arxiv_id = ""
        m = re.search(r"arxiv\.org/abs/([^v]+)", abs_url)
        if m:
            arxiv_id = m.group(1)
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        year = None
        pub = text("published")
        if len(pub) >= 4 and pub[:4].isdigit():
            year = int(pub[:4])
        authors = [a.find(f"{_ATOM}name").text
                   for a in entry.findall(f"{_ATOM}author")
                   if a.find(f"{_ATOM}name") is not None]
        return Paper(
            title=re.sub(r"\s+", " ", text("title")),
            abstract=re.sub(r"\s+", " ", text("summary")),
            authors=authors[:25],
            year=year,
            venue="arXiv",
            url=abs_url,
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
        )
