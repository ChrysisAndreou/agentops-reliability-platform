"""
Multi-agent coordination module for AgentOps Reliability Platform.

Provides supervisor-worker agent topologies with full tracing,
enabling systematic evaluation of multi-agent coordination patterns.
"""

from .state import (
    MultiAgentState,
    WorkerRole,
    WorkerAssignment,
    WorkerResult,
    InterAgentMessage,
    DEFAULT_WORKER_ROLES,
    create_multi_agent_state,
)
from .topology import build_multi_agent_graph
from .coordinator import (
    MultiAgentCoordinator,
    MultiAgentRunResult,
    MultiAgentConfig,
    extend_trace_store_for_multi_agent,
    save_multi_agent_run,
)

__all__ = [
    "MultiAgentState",
    "WorkerRole",
    "WorkerAssignment",
    "WorkerResult",
    "InterAgentMessage",
    "DEFAULT_WORKER_ROLES",
    "create_multi_agent_state",
    "build_multi_agent_graph",
    "MultiAgentCoordinator",
    "MultiAgentRunResult",
    "MultiAgentConfig",
    "extend_trace_store_for_multi_agent",
    "save_multi_agent_run",
]
