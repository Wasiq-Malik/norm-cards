"""Generic tool-calling agent loop, used by the evaluation judge.

The loop runs until the agent calls its terminal `submit_*` tool. There is no
budget: `max_steps` is a runaway guard, and it is never mentioned to the model —
the judge is supposed to research as long as it needs to.

Two details worth knowing before changing anything here:

1. `reasoning_effort` MUST be passed explicitly on every call. The API rejects
   function tools on /v1/chat/completions when the parameter is omitted
   ("Function tools with reasoning_effort are not supported ... use /v1/responses
   or set reasoning_effort to 'none'"), but accepts every explicit level up to
   xhigh. Verified live against gpt-5.6-sol and gpt-5.6-luna.

2. Long research runs would blow the context window on raw page text, so old tool
   results get compacted. That is only safe because evidence is captured *as it is
   found* via `record_evidence`: the ledger is pinned in context and never
   compacted, so an agent can still quote a paper it read eighty steps ago. This
   is also what enforces the design's "every judgment quotes evidence" rule at the
   mechanism level rather than by asking nicely in a prompt.
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional

from .. import llm
from . import tools

# Keep the newest N tool results verbatim; older ones shrink to a head slice.
_KEEP_VERBATIM = 6
_COMPACT_AT_CHARS = 400_000     # ~100k tokens of message payload
_COMPACT_HEAD = 700

RECORD_EVIDENCE_SPEC = {
    "type": "function", "function": {
        "name": "record_evidence",
        "description": "Record a piece of evidence the moment you find it: the exact "
                       "quoted text, where it came from, and what it establishes. Do "
                       "this as you read, not at the end — raw pages you fetched may be "
                       "compacted out of context later, but recorded evidence stays "
                       "available to you and becomes the audit trail for your final "
                       "answer. Record anything you might cite.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string",
                       "description": "URL, DOI, arXiv id, or paper title"},
            "quote": {"type": "string",
                      "description": "verbatim text from that source — do not paraphrase"},
            "bearing": {"type": "string",
                        "description": "what this establishes and why it matters here"}},
            "required": ["source", "quote", "bearing"]}}}


class AgentResult(dict):
    """Submitted payload plus run metadata (`_meta`, `_evidence`)."""


def _msg_chars(messages: List[Dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _compact(messages: List[Dict], protected: set):
    """Shrink older tool results in place once the transcript gets large."""
    tool_idxs = [i for i, m in enumerate(messages)
                 if m.get("role") == "tool" and i not in protected]
    if len(tool_idxs) <= _KEEP_VERBATIM:
        return
    for i in tool_idxs[:-_KEEP_VERBATIM]:
        c = messages[i].get("content") or ""
        if len(c) > _COMPACT_HEAD:
            messages[i]["content"] = (
                c[:_COMPACT_HEAD]
                + f"\n\n[... {len(c) - _COMPACT_HEAD} chars compacted out of context. "
                  "The full result is in this run's trace.jsonl. Re-fetch if you need "
                  "it again; anything you recorded with record_evidence is still above.]")
            protected.add(i)


def _ledger_text(ledger: List[Dict]) -> str:
    if not ledger:
        return "EVIDENCE LEDGER: empty. Record evidence as you find it."
    lines = ["EVIDENCE LEDGER (your recorded evidence; cite from here):"]
    for i, e in enumerate(ledger):
        lines.append(f"[E{i}] source: {e['source']}\n     quote: \"{e['quote']}\""
                     f"\n     bearing: {e['bearing']}")
    return "\n".join(lines)


def run_agent(system: str, user: str, submit_spec: Dict, model: str,
              reasoning_effort: str = "high", max_steps: int = 200,
              trace_path: Optional[str] = None,
              extra_tools: Optional[List[Dict]] = None,
              validate: Optional[Callable[[Dict], Optional[str]]] = None,
              progress: bool = True, tag: str = "") -> AgentResult:
    """Run an agent until it calls `submit_spec`'s tool; return that payload.

    `validate` may return an error string, which is handed back to the agent so it
    can fix and resubmit rather than the run dying on a schema slip.
    """
    submit_name = submit_spec["function"]["name"]
    tool_specs = list(tools.SPECS) + [RECORD_EVIDENCE_SPEC] + list(extra_tools or [])
    tool_specs.append(submit_spec)

    ledger: List[Dict] = []
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "system", "content": _ledger_text(ledger)}]
    ledger_idx = 2
    protected = {0, 1, 2}

    trace = None
    if trace_path:
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        trace = open(trace_path, "w", encoding="utf-8")

    def log(rec):
        if trace:
            trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace.flush()

    log({"type": "start", "ts": time.time(), "model": model,
         "reasoning_effort": reasoning_effort, "system": system, "user": user})

    lt = llm._litellm()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    started = time.time()
    result: Optional[AgentResult] = None

    for step in range(max_steps):
        if _msg_chars(messages) > _COMPACT_AT_CHARS:
            _compact(messages, protected)
        messages[ledger_idx]["content"] = _ledger_text(ledger)

        resp = _completion_with_retry(lt, model, messages, tool_specs,
                                      reasoning_effort, log)
        usage["calls"] += 1
        u = getattr(resp, "usage", None)
        if u:
            usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0

        m = resp.choices[0].message
        calls = m.tool_calls or []
        assistant_msg = {"role": "assistant", "content": m.content or ""}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name,
                              "arguments": c.function.arguments}} for c in calls]
        messages.append(assistant_msg)

        if not calls:
            # No tool call and no submission: nudge rather than accept prose.
            log({"type": "assistant_text", "step": step, "text": m.content})
            messages.append({"role": "user", "content":
                             f"Continue. Use the tools to keep investigating, and call "
                             f"{submit_name} when — and only when — you have finished the "
                             f"full protocol."})
            continue

        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except Exception as e:
                out = json.dumps({"error": f"arguments were not valid JSON: {e}"})
                log({"type": "bad_args", "step": step, "name": name,
                     "raw": c.function.arguments})
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "name": name, "content": out})
                continue

            if name == submit_name:
                err = validate(args) if validate else None
                log({"type": "submit", "step": step, "ts": time.time(),
                     "payload": args, "validation_error": err})
                if err:
                    if progress:
                        print(f"    {tag}[step {step}] submit rejected: {err[:120]}")
                    messages.append({"role": "tool", "tool_call_id": c.id, "name": name,
                                     "content": json.dumps({
                                         "error": err,
                                         "note": "Not accepted. Fix the issues and call "
                                                 f"{name} again."})})
                    continue
                result = AgentResult(args)
                break

            if name == "record_evidence":
                ledger.append({"source": args.get("source", ""),
                               "quote": args.get("quote", ""),
                               "bearing": args.get("bearing", "")})
                out = json.dumps({"recorded": True, "id": f"E{len(ledger) - 1}",
                                  "total_recorded": len(ledger)})
            else:
                out = tools.dispatch(name, args)

            log({"type": "tool", "step": step, "ts": time.time(), "name": name,
                 "args": args, "result_chars": len(out), "result": out})
            if progress:
                print(f"    {tag}[step {step}] {name}({_brief(args)}) -> {len(out)} chars")
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "name": name, "content": out})

        if result is not None:
            break

    if trace:
        trace.close()

    if result is None:
        raise RuntimeError(
            f"agent hit max_steps={max_steps} without calling {submit_name}"
            + (f" (trace: {trace_path})" if trace_path else ""))

    result["_meta"] = {"model": model, "reasoning_effort": reasoning_effort,
                       "steps": step + 1, "usage": usage,
                       "seconds": round(time.time() - started, 1),
                       "trace": trace_path}
    result["_evidence"] = ledger
    return result


def _brief(args: Dict, n: int = 60) -> str:
    s = ", ".join(f"{k}={str(v)[:n]!r}" for k, v in args.items())
    return s[:140]


def _completion_with_retry(lt, model, messages, tool_specs, reasoning_effort, log,
                           retries: int = 5):
    """Retry transient API failures (rate limits, timeouts, 5xx) with backoff."""
    for attempt in range(retries):
        try:
            return lt.completion(
                model=model, messages=messages, tools=tool_specs,
                temperature=1,                    # gpt-5 family requires 1
                reasoning_effort=reasoning_effort)  # required whenever tools are sent
        except Exception as e:
            transient = any(s in type(e).__name__.lower()
                            for s in ("ratelimit", "timeout", "apiconnection",
                                      "internalserver", "serviceunavailable"))
            if not transient or attempt == retries - 1:
                raise
            wait = min(2 ** attempt * 5, 60)
            log({"type": "api_retry", "error": f"{type(e).__name__}: {str(e)[:200]}",
                 "wait": wait, "attempt": attempt + 1})
            print(f"    [api] {type(e).__name__}; retry in {wait}s "
                  f"({attempt + 1}/{retries})")
            time.sleep(wait)
