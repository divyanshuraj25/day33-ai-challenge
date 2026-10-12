"""LangSmith tracing: one LangChainTracer callback attached to every pipeline run."""
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

_tracer = None


def tracing_enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY"))


def get_tracer():
    """Return a LangChainTracer (LangSmith callback), or None if no API key is set."""
    global _tracer
    if not tracing_enabled():
        warnings.warn("LANGSMITH_API_KEY not set - running WITHOUT tracing.")
        return None
    if _tracer is None:
        from langchain_core.tracers.langchain import LangChainTracer

        _tracer = LangChainTracer(project_name=os.getenv("LANGSMITH_PROJECT", "ai-debugging-day30"))
    return _tracer


def flush():
    """Block until all traces are uploaded. Call at the end of every script."""
    if not tracing_enabled():
        return
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()
