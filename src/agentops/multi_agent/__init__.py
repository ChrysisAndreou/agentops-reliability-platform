"""
Multi-agent coordination module for AgentOps Reliability Platform.

Provides supervisor-worker agent topologies with full tracing,
systematic evaluation of multi-agent coordination patterns,
and benchmark tasks requiring genuine inter-agent coordination.

v0.7 — Core multi-agent topology (supervisor-worker with 4 specialized roles)
v0.29 — Multi-agent evaluation framework (benchmarks, coordination metrics, reporting)
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
from .benchmarks import (
    CoordinationPattern,
    BenchmarkDifficulty,
    MultiAgentBenchmarkTask,
    MultiAgentBenchmark,
    MULTI_AGENT_BENCHMARK,
    get_multi_agent_benchmark,
    get_tasks_by_difficulty,
    get_tasks_by_pattern,
)
from .eval import (
    CoordinationMetrics,
    MultiAgentEvalReport,
    MultiAgentEvaluator,
    evaluate_multi_agent_system,
    format_multi_agent_eval_report,
    # Metric functions
    decomposition_quality,
    routing_accuracy,
    coordination_efficiency,
    conflict_resolution_quality,
    synthesis_quality,
    load_balance_score,
    scalability_score,
)

__all__ = [
    # State
    "MultiAgentState",
    "WorkerRole",
    "WorkerAssignment",
    "WorkerResult",
    "InterAgentMessage",
    "DEFAULT_WORKER_ROLES",
    "create_multi_agent_state",
    # Topology
    "build_multi_agent_graph",
    # Coordinator
    "MultiAgentCoordinator",
    "MultiAgentRunResult",
    "MultiAgentConfig",
    "extend_trace_store_for_multi_agent",
    "save_multi_agent_run",
    # Benchmarks (v0.29)
    "CoordinationPattern",
    "BenchmarkDifficulty",
    "MultiAgentBenchmarkTask",
    "MultiAgentBenchmark",
    "MULTI_AGENT_BENCHMARK",
    "get_multi_agent_benchmark",
    "get_tasks_by_difficulty",
    "get_tasks_by_pattern",
    # Evaluation (v0.29)
    "CoordinationMetrics",
    "MultiAgentEvalReport",
    "MultiAgentEvaluator",
    "evaluate_multi_agent_system",
    "format_multi_agent_eval_report",
    "decomposition_quality",
    "routing_accuracy",
    "coordination_efficiency",
    "conflict_resolution_quality",
    "synthesis_quality",
    "load_balance_score",
    "scalability_score",
]
