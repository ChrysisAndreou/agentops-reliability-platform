"""
AgentOps SDK — client library for instrumenting AI agents with observability.

Install: pip install agentops
Usage:
    import agentops
    agentops.init(endpoint="http://localhost:8000")

    @agentops.trace()
    async def my_agent(task: str) -> str:
        agentops.log_retrieval(query=task, chunks=["doc1", "doc2"])
        return "answer"

    # Query traces from the server
    traces = agentops.list_traces()
"""

from .client import AgentOpsHTTPClient
from .state import (
    AgentOpsClient,
    RetrievalRecord,
    RunContext,
    SDKConfig,
    SpanKind,
    SpanStatus,
    ToolCallRecord,
    TraceSpan,
    TraceStatus,
)
from .tracer import (
    AgentOps,
    get_current_run,
    log_retrieval,
    log_tool_call,
    log_verification,
    start_run,
    trace,
)

__all__ = [
    # Main API
    "AgentOps",
    "trace",
    "start_run",
    "log_tool_call",
    "log_retrieval",
    "log_verification",
    "get_current_run",
    # Types
    "SDKConfig",
    "TraceSpan",
    "SpanKind",
    "SpanStatus",
    "RunContext",
    "AgentOpsClient",
    "AgentOpsHTTPClient",
    "ToolCallRecord",
    "RetrievalRecord",
    "TraceStatus",
]
