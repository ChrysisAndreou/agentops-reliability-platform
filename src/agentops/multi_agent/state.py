"""
Multi-agent state schemas for supervisor-worker agent topologies.

Extends the single-agent ReliabilityState with multi-agent concepts:
supervisor task decomposition, worker assignments, inter-agent
messages, and consensus/reconciliation tracking.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage

from ..agent.state import ReliabilityState, ReliabilityStep, RetrievalResult


class WorkerRole(TypedDict):
    """Definition of a worker agent's specialization."""
    name: str
    description: str
    capabilities: list[str]  # e.g., ["retrieval", "tool_use", "code", "verification"]
    tools: list[str]  # tool names this worker has access to


class WorkerAssignment(TypedDict):
    """A subtask assigned to a specific worker agent."""
    assignment_id: str
    worker_role: str
    subtask: str
    context: str  # additional context passed to the worker
    priority: int  # 0 = highest


class InterAgentMessage(TypedDict):
    """A message passed between agents in the multi-agent system."""
    msg_id: str
    from_agent: str  # "supervisor" or worker name
    to_agent: str
    msg_type: Literal["assignment", "result", "query", "clarification", "consensus"]
    content: str
    timestamp_ms: float
    metadata: dict[str, Any]


class WorkerResult(TypedDict):
    """Result from a worker agent completing a subtask."""
    assignment_id: str
    worker_role: str
    subtask: str
    answer: str
    grounded_claims: list[str]
    ungrounded_claims: list[str]
    citations_used: list[str]
    verification_passed: bool
    tool_calls_count: int
    retrieved_chunks_count: int
    latency_ms: float
    error: str | None


class MultiAgentState(TypedDict, total=False):
    """State for a supervisor-worker multi-agent workflow.

    The supervisor receives a complex task, decomposes it into subtasks,
    routes them to specialized worker agents, aggregates results, and
    produces a final verified answer.

    Flow: decompose → assign → workers_execute → aggregate → verify → respond
    """
    messages: Annotated[list[BaseMessage], operator.add]
    task: str
    task_context: str

    # Decomposition
    subtasks: list[str]
    decomposition_rationale: str

    # Worker pool
    worker_roles: list[WorkerRole]
    assignments: list[WorkerAssignment]
    worker_results: Annotated[list[WorkerResult], operator.add]

    # Inter-agent communication trace
    inter_agent_messages: Annotated[list[InterAgentMessage], operator.add]

    # Aggregation
    aggregated_answer: str
    aggregation_notes: str

    # Verification (reuses single-agent verification logic)
    verification_passed: bool
    verification_notes: str
    grounded_claims: list[str]
    ungrounded_claims: list[str]

    # Response
    final_answer: str
    citations_used: list[str]

    # Control
    step_count: int
    current_phase: str  # "decompose", "assign", "execute", "aggregate", "verify", "respond", "done"
    done: bool
    error: str

    # Tracing
    reliability_trace: Annotated[list[ReliabilityStep], operator.add]
    coordination_trace: Annotated[list[dict[str, Any]], operator.add]


# ── Default worker roles ─────────────────────────────────────────────

DEFAULT_WORKER_ROLES: list[WorkerRole] = [
    WorkerRole(
        name="retrieval_specialist",
        description="Specializes in finding and synthesizing information from documentation",
        capabilities=["retrieval", "synthesis"],
        tools=[],
    ),
    WorkerRole(
        name="tool_executor",
        description="Executes calculations, API calls, and data transformations",
        capabilities=["tool_use", "computation"],
        tools=["calculator"],
    ),
    WorkerRole(
        name="code_analyst",
        description="Analyzes code, configuration files, and system logs",
        capabilities=["code_analysis", "debugging", "configuration"],
        tools=[],
    ),
    WorkerRole(
        name="verifier",
        description="Fact-checks claims against evidence and validates outputs",
        capabilities=["verification", "fact_checking"],
        tools=[],
    ),
]

# ── Factory functions ────────────────────────────────────────────────


def create_multi_agent_state(
    task: str,
    context: str = "",
    worker_roles: list[WorkerRole] | None = None,
) -> MultiAgentState:
    """Create the initial state for a multi-agent run."""
    return MultiAgentState(
        messages=[HumanMessage(content=task)],
        task=task,
        task_context=context,
        subtasks=[],
        decomposition_rationale="",
        worker_roles=worker_roles or DEFAULT_WORKER_ROLES,
        assignments=[],
        worker_results=[],
        inter_agent_messages=[],
        aggregated_answer="",
        aggregation_notes="",
        verification_passed=False,
        verification_notes="",
        grounded_claims=[],
        ungrounded_claims=[],
        final_answer="",
        citations_used=[],
        step_count=0,
        current_phase="decompose",
        done=False,
        error="",
        reliability_trace=[],
        coordination_trace=[],
    )
