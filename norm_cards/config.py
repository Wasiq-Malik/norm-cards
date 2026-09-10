"""Configuration and API-key loading for norm_cards.

Keys resolve from the process environment, then an optional `api_key.py` at the
repo root (the repo's convention). Sources/LLM read keys directly and will fail
loudly if a required key is absent — by design.
"""

import re
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


def _load_dotenv():
    """Minimal no-dependency .env loader (repo root). Does not override real env."""
    path = os.path.join(_REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()

# Polite-pool contact for OpenAlex (recommended, not a key).
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "scify-norm-cards@umbc.edu")

# Default LLM for classification / query generation.
DEFAULT_MODEL = os.environ.get("NORM_CARDS_MODEL", "gpt-5-mini")

HTTP_TIMEOUT = int(os.environ.get("NORM_CARDS_HTTP_TIMEOUT", "30"))
USER_AGENT = "scify-norm-cards/0.1 (research; paper-gathering)"


def _from_api_key_module(name):
    try:
        import api_key  # type: ignore

        return getattr(api_key, name, None)
    except Exception:
        return None


def get_key(name):
    """Return an API key by name from env, then api_key.py. None if absent."""
    return os.environ.get(name) or _from_api_key_module(name)


def openai_key():
    return get_key("OPENAI_API_KEY")


def anthropic_key():
    return get_key("ANTHROPIC_API_KEY")


def serp_key():
    return get_key("SERP_API_KEY")


def openalex_key():
    return get_key("OPENALEX_API_KEY")


def semantic_scholar_key():
    return get_key("SEMANTIC_SCHOLAR_API_KEY")


# --------------------------------------------------------------------------- #
# secret redaction
# --------------------------------------------------------------------------- #
_SECRET_NAMES = ("SERPAPI_API_KEY", "SERP_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "SEMANTIC_SCHOLAR_API_KEY", "HF_TOKEN")


def redact(text):
    """Strip credentials out of a string before it is logged, traced or shown.

    Written because they were not. `requests.Response.raise_for_status()` puts
    the FULL request URL in the exception message, query string included, and a
    SerpAPI call carries its key there — so every 429 during a judge run wrote
    the key into that run's trace.jsonl. The tool belt turns exceptions into
    strings (`{"error": str(e)}`) and the agent loop writes every tool result to
    disk, so one rate-limited search was enough to persist it.

    Redacts by value (any configured key, wherever it appears) and by shape (a
    `key=` query parameter), so a credential this function does not know about
    still does not survive a URL.
    """
    s = str(text)
    for name in _SECRET_NAMES:
        v = get_key(name)
        if v and len(v) > 6:
            s = s.replace(v, f"<{name}:redacted>")
    return re.sub(r"((?:api_|access_|auth_|secret_)?key|token|password)=[^&\s'\"]+",
                  r"\1=<redacted>", s, flags=re.I)
