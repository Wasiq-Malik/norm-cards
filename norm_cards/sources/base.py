"""Base class and shared HTTP helper for paper sources."""

import time
from typing import List

from ..models import Paper
from .. import config

_session = None


def http():
    """Shared requests session with a polite User-Agent."""
    global _session
    if _session is None:
        import requests

        _session = requests.Session()
        _session.headers.update({"User-Agent": config.USER_AGENT})
    return _session


def request(url, retries: int = 4, **kwargs):
    """GET with retry/backoff on 429/503 (honors Retry-After). Required to use
    these public APIs at all — not a key/fallback degradation."""
    kwargs.setdefault("timeout", config.HTTP_TIMEOUT)
    last = None
    for attempt in range(retries):
        r = http().get(url, **kwargs)
        if r.status_code in (429, 503) and attempt < retries - 1:
            wait = int(r.headers.get("Retry-After", 0)) or min(2 ** attempt, 30)
            print(f"[http] {r.status_code} on {url.split('?')[0]}; retry in {wait}s "
                  f"({attempt + 1}/{retries})")
            time.sleep(wait)
            last = r
            continue
        r.raise_for_status()
        return r
    last.raise_for_status()
    return last


class BaseSource:
    name = "base"

    def search(self, query: str, limit: int = 10, context: str = "") -> List[Paper]:
        raise NotImplementedError

    def search_tagged(self, query: str, limit: int, context: str = "") -> List[Paper]:
        """Run search and stamp source/query provenance onto each result.

        `context` (the claim) is used by semantic sources to disambiguate the
        query; lexical sources ignore it."""
        papers = self.search(query, limit, context) or []
        for p in papers:
            if self.name not in p.sources:
                p.sources.append(self.name)
            if query not in p.queries:
                p.queries.append(query)
        return papers
