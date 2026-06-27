"""
AgentOps Reliability Platform

A production-oriented toolkit for building, tracing, evaluating, and
improving the reliability of tool-using AI agents. Provides:

- LangGraph agent orchestration with typed tools and failure handling
- Hybrid retrieval with verifiable citations
- SQLite-backed trace store with replay and failure classification
- Systematic evaluation harness with reliability metrics
- FastAPI server and CLI for trace/eval inspection
- SDK/client library for agent instrumentation (v0.14)

Designed to bridge the gap between research prototypes and production
agent systems by making observability and evaluation first-class concerns.
"""

__version__ = "0.14.0"

# Lazy-import SDK on first access so the module is available without
# pulling in the full SDK deps at import time.
_sdk = None


def _get_sdk():
    global _sdk
    if _sdk is None:
        from agentops.sdk.tracer import AgentOps as _AgentOps
        _sdk = _AgentOps()
    return _sdk


def init(endpoint="http://localhost:8000", **kwargs):
    """Initialize the AgentOps SDK. See agentops.sdk.init for full docs."""
    from agentops.sdk.tracer import init as _init
    return _init(endpoint=endpoint, **kwargs)


def trace(*args, **kwargs):
    """Decorator to trace a function as an agent run. See agentops.sdk.trace."""
    from agentops.sdk.tracer import trace as _trace
    return _trace(*args, **kwargs)


def start_run(*args, **kwargs):
    """Start a new agent run context manager. See agentops.sdk.start_run."""
    from agentops.sdk.tracer import start_run as _start_run
    return _start_run(*args, **kwargs)


def log_tool_call(*args, **kwargs):
    """Log a tool call. See agentops.sdk.log_tool_call."""
    from agentops.sdk.tracer import log_tool_call as _log_tool_call
    return _log_tool_call(*args, **kwargs)


def log_retrieval(*args, **kwargs):
    """Log a retrieval operation. See agentops.sdk.log_retrieval."""
    from agentops.sdk.tracer import log_retrieval as _log_retrieval
    return _log_retrieval(*args, **kwargs)


def log_verification(*args, **kwargs):
    """Log a verification decision. See agentops.sdk.log_verification."""
    from agentops.sdk.tracer import log_verification as _log_verification
    return _log_verification(*args, **kwargs)


def get_current_run(*args, **kwargs):
    """Get the current active run context. See agentops.sdk.get_current_run."""
    from agentops.sdk.tracer import get_current_run as _get_current_run
    return _get_current_run(*args, **kwargs)
