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

import os
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
