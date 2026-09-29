"""Submit-tool loop, used by the evaluation judge.

The judge has exactly one tool, its `submit_*` tool, and the loop runs until it calls
it with a payload that passes `validate`. A rejected payload goes back to the model
with the error so it can fix and resubmit rather than the run dying on a schema slip.
`max_steps` is a runaway guard and is never mentioned to the model.

The judge once had a research belt (web search, OpenAlex, full-text fetch) and an
evidence ledger. They were removed: in the old judge stress suite the belt made an
order-invariance control drift by 0.165 instead of 0.069, because rate-limited
searches hand the judge different evidence on different passes, and the judgement we
want to measure is the judge's own reading of the claim and the two lists.

`reasoning_effort` MUST be passed explicitly on every call. The API rejects function
tools on /v1/chat/completions when the parameter is omitted ("Function tools with
reasoning_effort are not supported ... use /v1/responses or set reasoning_effort to
'none'"), but accepts every explicit level up to xhigh. Verified live against
gpt-5.6-sol and gpt-5.6-luna.

litellm gates the parameter on its own per-model support table, which lags new
releases: gpt-6-luna was rejected client-side with "openai does not support
parameters: ['reasoning_effort']" before any request was sent. The call passes
`allowed_openai_params` to override that table rather than dropping the parameter,
because dropping it hits the server-side rejection above and a judge silently running
at a different effort is worse than a crash.

The GPT-6 family refuses function tools on /v1/chat/completions at any reasoning
effort, accepting them only at effort 'none' — which would leave the judge doing no
reasoning at all. It works at full effort through /v1/responses, so those models go
through a second transport rather than being locked out. gpt-6 models are unaffected
as PROPOSERS, which send no tools.
"""

import json
import os
import time
from typing import Callable, Dict, List, Optional

from .. import llm


class AgentResult(dict):
    """Submitted payload plus run metadata (`_meta`)."""


# ---------------------------------------------------------------------------
# Transport. Two wire formats, one loop.
#
# /v1/chat/completions takes `messages` and nested {"type":"function","function":…}
# tool specs, and answers in `choices[0].message.tool_calls`. /v1/responses takes
# `input` and FLAT {"type":"function","name":…} specs, and answers with a list of
# output items. The loop below should not have to care, so each transport keeps its
# own conversation state and exposes the same three operations.
#
# The responses transport must echo back every output item it received — reasoning
# items included — before the matching `function_call_output`, or the model loses
# the thread of its own tool call. That is why it stores raw items rather than
# rebuilding them from the loop's view.
#
# Neither transport may wait forever. A judge call that hangs looks identical to a
# judge call that is thinking hard: 0% CPU, no output, no error. One hung here for
# 32 minutes on 1.5 seconds of CPU before timeouts were added, and a card build once
# sat for four hours the same way. A visible failure beats a silent stall.
# ---------------------------------------------------------------------------

_RESPONSES_FAMILIES = ("gpt-6", "gpt-7")


def uses_responses_api(model: str) -> bool:
    """Does this model need /v1/responses to accept tools with reasoning effort?"""
    return (model or "").split("/")[-1].lower().startswith(_RESPONSES_FAMILIES)


class _Call:
    """One tool call, in whichever wire format it arrived."""

    def __init__(self, cid: str, name: str, arguments: str):
        self.id, self.name, self.arguments = cid, name, arguments


class _ChatSession:
    """/v1/chat/completions."""

    def __init__(self, system: str, user: str):
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": user}]

    def send(self, lt, model, tool_specs, effort, log):
        resp = _with_retry(lambda: lt.completion(
            model=model, messages=self.messages, tools=tool_specs,
            temperature=1,                        # gpt-5 family requires 1
            reasoning_effort=effort,              # required whenever tools are sent
            allowed_openai_params=["reasoning_effort"],
            timeout=llm.TIMEOUT), log)
        m = resp.choices[0].message
        calls = [_Call(c.id, c.function.name, c.function.arguments)
                 for c in (m.tool_calls or [])]
        msg = {"role": "assistant", "content": m.content or ""}
        if calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in calls]
        self.messages.append(msg)
        u = getattr(resp, "usage", None)
        return (m.content or ""), calls, (getattr(u, "prompt_tokens", 0) or 0,
                                          getattr(u, "completion_tokens", 0) or 0)

    def add_tool_result(self, call: "_Call", content: str):
        self.messages.append({"role": "tool", "tool_call_id": call.id,
                              "name": call.name, "content": content})

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})


class _ResponsesSession:
    """/v1/responses — for models that refuse tools on chat/completions."""

    def __init__(self, system: str, user: str):
        self.input = [{"role": "system", "content": system},
                      {"role": "user", "content": user}]

    def send(self, lt, model, tool_specs, effort, log):
        flat = [{"type": "function", "name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "parameters": t["function"].get("parameters", {})}
                for t in tool_specs]
        resp = _with_retry(lambda: lt.responses(
            model=model, input=self.input, tools=flat,
            reasoning={"effort": effort}, timeout=llm.TIMEOUT), log)
        text, calls = "", []
        for it in resp.output:
            # echo every item back, verbatim, before any tool result
            self.input.append(it.model_dump() if hasattr(it, "model_dump") else it)
            kind = getattr(it, "type", None)
            if kind == "function_call":
                calls.append(_Call(it.call_id, it.name, it.arguments))
            elif kind == "message":
                for part in (getattr(it, "content", None) or []):
                    text += getattr(part, "text", "") or ""
        u = getattr(resp, "usage", None)
        return text, calls, (getattr(u, "input_tokens", 0) or 0,
                             getattr(u, "output_tokens", 0) or 0)

    def add_tool_result(self, call: "_Call", content: str):
        self.input.append({"type": "function_call_output",
                           "call_id": call.id, "output": content})

    def add_user(self, text: str):
        self.input.append({"role": "user", "content": text})


def _with_retry(call: Callable, log, retries: int = 5):
    """Retry transient API failures (rate limits, timeouts, 5xx) with backoff."""
    for attempt in range(retries):
        try:
            return call()
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


def run_agent(system: str, user: str, submit_spec: Dict, model: str,
              reasoning_effort: str = "high", max_steps: int = 200,
              trace_path: Optional[str] = None,
              validate: Optional[Callable[[Dict], Optional[str]]] = None,
              progress: bool = True, tag: str = "") -> AgentResult:
    """Run the model until it calls `submit_spec`'s tool with a valid payload."""
    submit_name = submit_spec["function"]["name"]
    tool_specs = [submit_spec]
    session = (_ResponsesSession if uses_responses_api(model) else _ChatSession)(
        system, user)

    trace = None
    if trace_path:
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        trace = open(trace_path, "w", encoding="utf-8")

    def log(rec):
        if trace:
            trace.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace.flush()

    log({"type": "start", "ts": time.time(), "model": model,
         "transport": "responses" if uses_responses_api(model) else "chat_completions",
         "reasoning_effort": reasoning_effort, "system": system, "user": user})

    lt = llm._litellm()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    started = time.time()
    result: Optional[AgentResult] = None

    for step in range(max_steps):
        text, calls, (pt, ct) = session.send(lt, model, tool_specs,
                                             reasoning_effort, log)
        usage["calls"] += 1
        usage["prompt_tokens"] += pt
        usage["completion_tokens"] += ct

        if not calls:
            # Prose instead of a submission: nudge rather than accept it.
            log({"type": "assistant_text", "step": step, "text": text})
            session.add_user(f"Call {submit_name} with your evaluation.")
            continue

        for c in calls:
            if c.name != submit_name:
                session.add_tool_result(c, json.dumps(
                    {"error": f"unknown tool {c.name!r}; call {submit_name}"}))
                continue
            try:
                args = json.loads(c.arguments or "{}")
            except Exception as e:
                log({"type": "bad_args", "step": step, "raw": c.arguments})
                session.add_tool_result(c, json.dumps(
                    {"error": f"arguments were not valid JSON: {e}"}))
                continue
            err = validate(args) if validate else None
            log({"type": "submit", "step": step, "ts": time.time(),
                 "payload": args, "validation_error": err})
            if err:
                if progress:
                    print(f"    {tag}[step {step}] submit rejected: {err[:120]}")
                session.add_tool_result(c, json.dumps({
                    "error": err,
                    "note": f"Not accepted. Fix the issues and call {submit_name} again."}))
                continue
            result = AgentResult(args)
            break

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
    return result
