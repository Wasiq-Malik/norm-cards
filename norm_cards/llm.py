"""Thin LLM wrapper for classification / query generation (litellm-backed).

Assumes an LLM is configured (key + litellm available); calling without one
raises. No silent fallbacks.
"""

import os
import re
import json

from . import config


def _litellm():
    import litellm  # raises ImportError if missing — intentional

    if config.openai_key():
        os.environ.setdefault("OPENAI_API_KEY", config.openai_key())
    if config.anthropic_key():
        os.environ.setdefault("ANTHROPIC_API_KEY", config.anthropic_key())
    return litellm


def complete(prompt: str, model: str = None, temperature: float = 0.1) -> str:
    """Single-turn completion."""
    model = model or config.DEFAULT_MODEL
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    # gpt-5 family wants temperature=1, mirroring codeagent.py.
    kwargs["temperature"] = 1 if model.startswith("gpt-5") else temperature
    resp = _litellm().completion(**kwargs)
    return resp.choices[0].message.content.strip()


def complete_json(prompt: str, model: str = None) -> dict:
    """Completion expected to return a JSON object."""
    text = complete(prompt + "\n\nReturn ONLY valid JSON, no prose.", model=model)
    return extract_json(text)


def extract_json(text: str):
    """Parse the first JSON object/array from model output (tolerant of fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("Could not parse JSON from model output:\n" + text[:500])
