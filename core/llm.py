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
# call. Two shapes observed so far, both from the model's internal "Harmony"
# response format leaking into the tool-call name:
#   - A phantom tool name invented out of thin air (e.g. "json") when the model
#     should have written its final answer as plain text.
#   - A real, correctly-named tool corrupted with a trailing channel tag (e.g.
#     "bronze_ingestion_tool<|channel|>commentary").
# In both cases Groq's error body still contains the model's actual intended
# call in `failed_generation` -- recover it rather than failing the whole run.

_HARMONY_ARTIFACT_RE = re.compile(r"<\|.*$", re.DOTALL)


def invoke_agent_with_tool_recovery(agent, messages_input: dict, tools: list) -> dict:
    """Invoke a LangChain agent, recovering from Groq/gpt-oss tool-call corruption.

    If the recovered (cleaned) tool name matches one of `tools`, that tool is
    called directly with the recovered arguments and its real result becomes the
    final message -- exactly as if the agent had called it normally. Otherwise,
    the recovered arguments are treated as the model's final answer (its JSON
    content), matching the phantom-tool-name case.

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
        try:
            recovered = json.loads(error.get("failed_generation", ""))
            attempted_name = recovered["name"]
            arguments = recovered["arguments"]
        except (json.JSONDecodeError, KeyError, TypeError):
            raise

        cleaned_name = _HARMONY_ARTIFACT_RE.sub("", attempted_name)
        matched_tool = next((t for t in tools if getattr(t, "name", None) == cleaned_name), None)

        if matched_tool is not None:
            print(f"[RECOVERY] Groq corrupted a real tool call ('{attempted_name}') -- "
                  f"invoking the real tool '{cleaned_name}' directly.")
            tool_result = matched_tool.invoke(arguments)
            return {"messages": [SimpleNamespace(content=tool_result)]}

        if not isinstance(arguments, dict):
            raise
        print(f"[RECOVERY] Recovered final answer from a rejected phantom tool call ('{attempted_name}')")
        return {"messages": [SimpleNamespace(content=json.dumps(arguments))]}
