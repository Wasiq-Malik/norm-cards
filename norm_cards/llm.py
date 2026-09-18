"""Thin LLM wrapper for classification / query generation (litellm-backed).

Assumes an LLM is configured (key + litellm available); calling without one
raises. No silent fallbacks.
"""

import os
import re
import json
import time

from . import config


def _litellm():
    import litellm  # raises ImportError if missing — intentional

    if config.openai_key():
        os.environ.setdefault("OPENAI_API_KEY", config.openai_key())
    if config.anthropic_key():
        os.environ.setdefault("ANTHROPIC_API_KEY", config.anthropic_key())
    return litellm


# Reasoning models reject any temperature but the default. The check is by family
# rather than an exact list because the list keeps growing: gpt-6-astra shipped
# after this code was written and failed with "temperature does not support 0.1",
# because the only guard was `startswith("gpt-5")`.
_FIXED_TEMP = ("gpt-5", "gpt-6", "gpt-7", "o1", "o3", "o4")


def wants_default_temperature(model: str) -> bool:
    """Does this model reject an explicit temperature?"""
    m = (model or "").split("/")[-1].lower()
    return m.startswith(_FIXED_TEMP)


# No call gets to hang forever. A card build once sat for FOUR HOURS on an
# established connection with nothing to show for it, because litellm's default
# is to wait indefinitely and nothing here overrode it. A batch job that stalls
# silently is worse than one that fails: the failure is at least visible.
TIMEOUT = 300
RETRIES = 3


def complete(prompt: str, model: str = None, temperature: float = 0.1,
             timeout: int = TIMEOUT, retries: int = RETRIES) -> str:
    """Single-turn completion, with a hard timeout and bounded retries."""
    model = model or config.DEFAULT_MODEL
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
              "timeout": timeout}
    kwargs["temperature"] = 1 if wants_default_temperature(model) else temperature
    last = None
    for attempt in range(retries):
        try:
            resp = _litellm().completion(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last = e
            print(f"  [llm] {type(e).__name__} on {model} "
                  f"(attempt {attempt + 1}/{retries}): {str(e)[:110]}", flush=True)
            if attempt + 1 < retries:
                time.sleep(5 * (attempt + 1))
    raise last


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
