"""Papers With Code source (paperswithcode.co API, verified live 2026-06-28).

IMPORTANT: the free-text `/papers/?q=` endpoint is BROKEN — it ignores the query
and returns the whole corpus newest-first (verified: identical results + count
for unrelated queries). So PwC is NOT a query-search source. Instead we use it
*after* classification: a subfield mapped to a PwC task id pulls that task's
tagged papers via `/tasks/{id}/papers/` (high precision, on-topic). Papers carry
`tasks`/`introduced_benchmarks`/`hf_datasets`/`methods` — norm-card priors stashed
on Paper.tags.
"""

from typing import List

from .base import request
from ..models import Paper
from .. import config

_BASE = "https://paperswithcode.co/api/v1"


def _names(items):
    out = []
    for it in items or []:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict):
            out.append(it.get("name") or it.get("slug") or "")
    return [x for x in out if x]


def _to_paper(p: dict) -> Paper:
    published = p.get("published") or ""
    year = int(published[:4]) if published[:4].isdigit() else None
    tasks = _names(p.get("tasks"))
    return Paper(
        title=p.get("title") or "",
        abstract=(p.get("abstract") or (p.get("tldr") or "")) or "",
        authors=list(p.get("authors") or [])[:25],
        year=year,
        venue=p.get("conference_name") or p.get("proceeding") or "",
        citations=p.get("citation_count") or 0,
        url=p.get("url_abs") or "",
        pdf_url=p.get("url_pdf") or "",
        arxiv_id=(p.get("arxiv_id") or ""),
        fields_of_study=tasks,
        tags={
            "pwc_tasks": tasks,
            "pwc_methods": _names(p.get("methods")),
            "pwc_introduced_benchmarks": _names(p.get("introduced_benchmarks")),
            "pwc_hf_datasets": _names(p.get("hf_datasets")),
        },
    )


class PapersWithCode:
    """Not a query-search source — used post-classification by task id."""

    name = "papers_with_code"

    def papers_for_task(self, task_id: str, limit: int = 30) -> List[Paper]:
        """Pull papers PwC has tagged to a task (`/tasks/{id}/papers/`)."""
        r = request(f"{_BASE}/tasks/{task_id}/papers/", allow_redirects=True)
        results = r.json().get("results", [])[:limit]
        papers = []
        for p in results:
            paper = _to_paper(p)
            paper.sources.append(self.name)
            paper.buckets.append("pwc_task")
            papers.append(paper)
        return papers
