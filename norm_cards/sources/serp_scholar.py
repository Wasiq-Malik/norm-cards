"""Google Scholar source via SerpAPI.

Reuses the same approach as scimcp/Tools/General_tools.py. Requires SERP_API_KEY;
disables itself otherwise.
"""

from typing import List

from .base import BaseSource, request
from ..models import Paper
from .. import config


class SerpScholarSource(BaseSource):
    name = "google_scholar"

    def search(self, query: str, limit: int = 10, context: str = "") -> List[Paper]:
        params = {
            "engine": "google_scholar",
            "q": query,
            "num": min(limit, 20),
            "api_key": config.serp_key(),
        }
        r = request("https://serpapi.com/search", params=params)
        papers = []
        for res in r.json().get("organic_results", []):
            info = res.get("publication_info", {})
            authors = ", ".join(a.get("name", "")
                                for a in info.get("authors", []))
            summary = info.get("summary", "")
            year = next((t for t in summary.split()
                         if t.isdigit() and len(t) == 4), None)
            cited = res.get("inline_links", {}).get("cited_by", {}).get("total", 0)
            papers.append(Paper(
                title=res.get("title") or "",
                abstract=res.get("snippet") or "",
                authors=[a for a in authors.split(", ") if a],
                year=int(year) if year else None,
                citations=int(cited) if cited else 0,
                url=res.get("link") or "",
            ))
        return papers
