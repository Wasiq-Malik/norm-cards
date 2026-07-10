"""Runtime OpenAlex topic resolution for a claim.

Replaces the earlier static seed (506 topics curated from only 19 of OpenAlex's
252 subfields), which missed the correct topic for any claim outside that slice
— e.g. molecular-ML claims, whose topic "Machine Learning in Materials Science"
(T11948) lived in an un-curated subfield.

OpenAlex's `text/topics` endpoint classifies arbitrary text into the FULL
~4,516-topic taxonomy using the SAME model that tags every work. So the claim
and the corpus we later filter/rank against (papers carry these topic IDs for
free) end up embedded in one consistent topic space, with no vocabulary to
curate or keep in sync. Verified live: molecular-VAE claim -> T11948 @ 0.99.

Docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/
      filter-entity-lists  (the /text endpoints)
"""

from typing import List, Tuple

from .sources.base import http
from . import config

_ENDPOINT = "https://api.openalex.org/text/topics"


class TopicBudgetError(RuntimeError):
    """text/topics is a METERED endpoint (~$0.01/call, small free daily tier
    resetting midnight UTC). Raised on a budget-exhaustion 429 so callers fail
    fast with a clear message instead of honoring the ~5h Retry-After."""


def resolve_topics(text: str, context: str = "", top_k: int = 3,
                   min_score: float = 0.6) -> List[Tuple[str, str]]:
    """Return [(topic_id, display_name), ...] OpenAlex assigns to `text`,
    ranked by classifier score, filtered to `min_score` and capped at `top_k`.

    `text` is the claim (passed as `title`); `context` is disambiguating text
    (passed as `abstract`) — supply the LLM's classified subfields + entities
    here: the raw claim alone mis-classifies when it uses jargon (e.g. a
    "SMILES VAE / GuacaMol" claim reads as image synthesis without the hint
    that it is molecular). Empty on blank input.

    NOTE: text/topics is metered. A budget-exhaustion 429 raises
    TopicBudgetError (don't retry — the daily tier resets at midnight UTC);
    other HTTP errors propagate. No silent fallback."""
    text = (text or "").strip()
    if not text:
        return []
    params = {"title": text[:1900], "mailto": config.OPENALEX_MAILTO}
    if context:
        params["abstract"] = context[:1900]
    headers = {}
    key = config.openalex_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    r = http().get(_ENDPOINT, params=params, headers=headers,
                   timeout=config.HTTP_TIMEOUT)
    if r.status_code == 429 and "budget" in (r.text or "").lower():
        raise TopicBudgetError(r.json().get("message", "text/topics budget exhausted"))
    r.raise_for_status()
    out: List[Tuple[str, str]] = []
    for t in r.json().get("topics", []) or []:
        if (t.get("score") or 0.0) < min_score:
            continue
        tid = (t.get("id") or "").rsplit("/", 1)[-1]
        if tid:
            out.append((tid, t.get("display_name") or ""))
        if len(out) >= top_k:
            break
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Runtime OpenAlex topic resolution")
    ap.add_argument("text", type=str, help="Claim / text to classify into topics")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--min_score", type=float, default=0.6)
    args = ap.parse_args()
    for tid, name in resolve_topics(args.text, args.top_k, args.min_score):
        print(f"{tid:>8}  {name}")
