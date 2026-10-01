"""Shared LLM factory with a coordinated rate limiter.

Every agent used to build its own ChatGroq instance via a locally-duplicated
``_make_llm()``, so nothing paced the *overall* request rate across
profiler -> STTM -> bronze -> silver -> gold -> reporter -> the Supervisor's
own calls. A single pipeline run fires many LLM calls in quick succession,
which breaches a free-tier rate limit fast. This module centralises LLM
construction so every agent shares one InMemoryRateLimiter instance that
proactively spaces out requests, instead of firing as fast as Python allows
and only reacting after a 429.

Tune the pace via the LLM_REQUESTS_PER_MINUTE env var to match your actual
Groq rate limit (check your Groq console for the real number).
"""

import inspect
import json
import os
import re
from types import SimpleNamespace
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY, GROQ_MODEL

_REQUESTS_PER_MINUTE = float(os.getenv("LLM_REQUESTS_PER_MINUTE", "20"))

# One shared limiter instance for the whole process: every agent's make_llm() call
# draws from the same bucket, so pacing is coordinated across the entire pipeline
# rather than each agent independently allowing its own first call through.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=_REQUESTS_PER_MINUTE / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=1,  # no burst allowance -- always wait the full interval between calls
)


def make_llm():
    """Build the configured chat model, sharing one rate limiter across the whole pipeline."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        rate_limiter=_rate_limiter,
        max_retries=5,
    )


# ---------------------------------------------------------------------------
# Tool-call corruption recovery
# ---------------------------------------------------------------------------
# Groq-hosted gpt-oss models occasionally emit a malformed tool call that Groq's
# own server rejects with a 400 before it ever reaches LangChain as a normal tool
# call. Three shapes observed so far:
#   - A phantom tool name invented out of thin air (e.g. "json") when the model
#     should have written its final answer as plain text.
#   - A real, correctly-named tool corrupted with a trailing Harmony response-
#     format channel tag (e.g. "bronze_ingestion_tool<|channel|>commentary").
#   - A real, correctly-named tool called with bloated/invalid arguments (e.g.
#     re-forwarding an entire inspect-tool result as arguments instead of the
#     single `confirmation` field the tool actually takes), malformed enough
#     that even Groq's own `failed_generation` echo of the attempt is truncated,
#     unparseable JSON.
# The first two cases parse cleanly out of `failed_generation`; the third does
# not, so name recovery falls back to a regex pull of the (reliably intact,
# always-first) "name" field, and argument recovery falls back to invoking the
# tool with no arguments at all -- safe only when every one of the tool's own
# parameters has a default, which is true for every tool in this codebase
# except the handful that take a real required argument (e.g. SQL text or a
# goal string), where a safe empty-args retry isn't possible.

_HARMONY_ARTIFACT_RE = re.compile(r"<\|.*$", re.DOTALL)
_FAILED_GENERATION_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _tool_accepts_empty_args(tool) -> bool:
    """True if every parameter of the wrapped tool function has a default --
    meaning it's safe to invoke the tool with no arguments at all."""
    func = getattr(tool, "func", None)
    if func is None:
        return False
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return all(p.default is not inspect.Parameter.empty for p in sig.parameters.values())


def invoke_agent_with_tool_recovery(agent, messages_input: dict, tools: list) -> dict:
    """Invoke a LangChain agent, recovering from Groq/gpt-oss tool-call corruption.

    If the recovered (cleaned) tool name matches one of `tools`, that tool is
    called directly -- with the recovered arguments if they parsed, or with no
    arguments at all if they didn't but the tool allows it (see module docstring)
    -- and its real result becomes the final message, exactly as if the agent
    had called it normally. If the name matches no real tool, the recovered
    arguments are treated as the model's final answer (its JSON content).

    If nothing usable can be recovered at all, raises a RuntimeError carrying
    Groq's own error message rather than letting a secondary JSON-parsing
    failure mask what actually went wrong.

    Known limitation: this only recovers the single call that failed. For an
    agent that must make more than one tool call in sequence (e.g. the Phase 1
    Supervisor: profile, then generate Bronze STTM), recovering a failure on an
    earlier required call still ends the loop there -- later calls never run.
    In every occurrence observed so far, the corruption has hit the last call
    needed, after earlier calls already succeeded, which this handles correctly.
    """
    try:
        return agent.invoke(messages_input)
    except Exception as e:
        body = getattr(e, "body", None)
        if not isinstance(body, dict):
            raise
        error = body.get("error", {})
        if error.get("code") != "tool_use_failed":
            raise

        raw_failed_generation = error.get("failed_generation", "")
        attempted_name = None
        arguments = None
        try:
            recovered = json.loads(raw_failed_generation)
            attempted_name = recovered.get("name")
            arguments = recovered.get("arguments")
        except (json.JSONDecodeError, ValueError, AttributeError):
            # failed_generation itself is truncated/invalid -- the arguments are
            # unrecoverable, but the tool name reliably appears intact near the
            # start of the string even when the rest got cut off mid-generation.
            name_match = _FAILED_GENERATION_NAME_RE.search(raw_failed_generation)
            attempted_name = name_match.group(1) if name_match else None

        if attempted_name is None:
            raise RuntimeError(
                f"Groq rejected a tool call and the failure couldn't be recovered: "
                f"{error.get('message', 'unknown error')}"
            ) from e

        cleaned_name = _HARMONY_ARTIFACT_RE.sub("", attempted_name)
        matched_tool = next((t for t in tools if getattr(t, "name", None) == cleaned_name), None)

        if matched_tool is not None:
            if not isinstance(arguments, dict) or not arguments:
                if not _tool_accepts_empty_args(matched_tool):
                    raise RuntimeError(
                        f"Groq corrupted the call to '{cleaned_name}' and its arguments "
                        f"could not be recovered. It requires real arguments, so a safe "
                        f"empty-args retry isn't possible: {error.get('message', '')}"
                    ) from e
                arguments = {}
            print(f"[RECOVERY] Groq corrupted a real tool call ('{attempted_name}') -- "
                  f"invoking the real tool '{cleaned_name}' directly with "
                  f"{'recovered' if arguments else 'default'} arguments.")
            tool_result = matched_tool.invoke(arguments)
            return {"messages": [SimpleNamespace(content=tool_result)]}

        if not isinstance(arguments, dict):
            raise RuntimeError(
                f"Groq rejected a phantom tool call ('{attempted_name}') and its "
                f"arguments could not be recovered: {error.get('message', '')}"
            ) from e
        print(f"[RECOVERY] Recovered final answer from a rejected phantom tool call ('{attempted_name}')")
        return {"messages": [SimpleNamespace(content=json.dumps(arguments))]}
