"""OpenAlex source: NATIVE semantic search (`search.semantic`) + topic-filtered
lexical search (`search=` + `filter=topics.id:...`).

Semantic search embeds the query (GTE-Large-EN, 1024-d) and returns works
ranked by cosine similarity — good recall on conceptually-similar work even
when wording differs. Lexical search hard-filtered to the claim's classified
OpenAlex topic IDs recovers the survey/benchmark papers semantic search
under-ranks (surveys don't embed close to a narrow technical query) without
reintroducing the off-topic noise that got plain lexical search dropped
earlier — the topic filter, not query phrasing, is what keeps it on-topic.
collect.py unions both legs; see pipeline.gather_from_analysis.

Earlier versions appended the raw claim text to the semantic query as
disambiguating context. Validated experiment (union w/ topic-filtered lexical
in place): dropping it is a wash-to-slight-improvement on survey/benchmark
recall, and removes a failure mode (claim LaTeX/unicode math 400ing the API).

Notes: semantic search allows only one search param per request, caps at 50
results, and is rate-limited to ~1 req/sec (paced in collect.run_queries; the
shared http layer also retries 429). A key (Bearer) raises the daily budget.
`topics.id` filtering is NOT supported alongside `search.semantic` (API
rejects it despite docs suggesting broader filter support) — confirmed live,
hence the separate lexical+filter path for topic-constrained retrieval.
Docs: https://developers.openalex.org/guides/semantic-search
"""

from typing import List

from .base import BaseSource, request
from ..models import Paper
from .. import config


class OpenAlexSource(BaseSource):
    name = "openalex"

    def search(self, query: str, limit: int = 10, context: str = "") -> List[Paper]:
        params = {
            "search.semantic": query[:2000],  # API truncates/400s beyond limits
            "per_page": min(limit, 50),
            "mailto": config.OPENALEX_MAILTO,
        }
        headers = {}
        key = config.openalex_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = request("https://api.openalex.org/works", params=params, headers=headers)
        return [self._to_paper(w) for w in r.json().get("results", [])]

    def search_lexical(self, query: str, limit: int = 10,
                       topic_ids: List[str] = None) -> List[Paper]:
        """Keyword search hard-filtered to a set of OpenAlex topic IDs.

        `topics.id` filtering isn't supported alongside `search.semantic` (API
        rejects it), but works fine with plain lexical `search=`. This is what
        makes lexical search viable again: the topic filter removes exactly the
        off-topic keyword-matched noise that got lexical search dropped earlier.
        """
        params = {
            "search": query[:2000],
            "per_page": min(limit, 50),
            "mailto": config.OPENALEX_MAILTO,
        }
        if topic_ids:
            params["filter"] = "topics.id:" + "|".join(topic_ids)
        headers = {}
        key = config.openalex_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = request("https://api.openalex.org/works", params=params, headers=headers)
        return [self._to_paper(w) for w in r.json().get("results", [])]

    def _to_paper(self, w: dict) -> Paper:
        loc = (w.get("primary_location") or {})
        source = (loc.get("source") or {})
        pdf_url = loc.get("pdf_url") or ""
        oa = (w.get("open_access") or {})
        if not pdf_url and oa.get("oa_url"):
            pdf_url = oa["oa_url"]
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        return Paper(
            title=w.get("title") or "",
            abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
            authors=[a.get("author", {}).get("display_name", "")
                     for a in (w.get("authorships") or [])][:25],
            year=w.get("publication_year"),
            venue=source.get("display_name") or "",
            citations=w.get("cited_by_count") or 0,
            url=w.get("id") or "",
            pdf_url=pdf_url,
            doi=doi,
            fields_of_study=[c.get("display_name", "")
                             for c in (w.get("concepts") or [])][:6],
            topic_ids=[t["id"].rsplit("/", 1)[-1] for t in (w.get("topics") or [])],
            semantic_score=float(w.get("relevance_score") or 0.0),
        )


def _reconstruct_abstract(inverted_index) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not inverted_index:
        return ""
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)
