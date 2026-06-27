"""
Multi-agent coordination module for AgentOps Reliability Platform.

Provides supervisor-worker agent topologies with full tracing,
enabling systematic evaluation of multi-agent coordination patterns.
"""

from .coordinator import (
    MultiAgentConfig,
    MultiAgentCoordinator,
    MultiAgentRunResult,
    extend_trace_store_for_multi_agent,
    save_multi_agent_run,
)
from .state import (
    DEFAULT_WORKER_ROLES,
    InterAgentMessage,
    MultiAgentState,
    WorkerAssignment,
    WorkerResult,
    WorkerRole,
    create_multi_agent_state,
)
from .topology import build_multi_agent_graph

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
