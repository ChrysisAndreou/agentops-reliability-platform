"""
Multi-agent coordinator — orchestrates supervisor-worker agent workflows.

The coordinator provides a high-level API for running multi-agent tasks:
- Creates worker agents from the existing ReliabilityAgent
- Wires them into the supervisor-worker topology
- Manages trace persistence and result aggregation
- Supports both real LLM-backed and simulated workers
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import (
    MultiAgentState,
    WorkerAssignment,
    WorkerResult,
    InterAgentMessage,
    WorkerRole,
    DEFAULT_WORKER_ROLES,
    create_multi_agent_state,
)
from .topology import build_multi_agent_graph


@dataclass
class MultiAgentRunResult:
    """Result of a complete multi-agent workflow run."""
    run_id: str
    task: str
    subtasks: list[str]
    assignments: list[dict[str, Any]]
    worker_results: list[WorkerResult]
    inter_agent_messages: list[InterAgentMessage]
    aggregated_answer: str
    final_answer: str
    verification_passed: bool
    grounded_claims: list[str]
    ungrounded_claims: list[str]
    citations_used: list[str]
    coordination_trace: list[dict[str, Any]]
    error: str | None
    total_latency_ms: float
    worker_count: int
    success: bool


@dataclass
class MultiAgentConfig:
    """Configuration for a multi-agent deployment."""
    worker_roles: list[WorkerRole] = field(default_factory=lambda: DEFAULT_WORKER_ROLES)
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_worker_concurrency: int = 3  # max parallel workers


class MultiAgentCoordinator:
    """Orchestrates multi-agent workflows.

    Usage:
        coordinator = MultiAgentCoordinator(
            worker_fn=my_worker_function,
            config=MultiAgentConfig(),
        )
        result = await coordinator.run("Complex task requiring multiple agents")
    """

    def __init__(
        self,
        worker_fn: Callable[[str, str, str], WorkerResult],
        config: MultiAgentConfig | None = None,
    ):
        """
        Args:
            worker_fn: Function (role, subtask, context) -> WorkerResult.
                      Each call should run the appropriate specialized agent
                      on the given subtask.
            config: Multi-agent configuration.
        """
        self._worker_fn = worker_fn
        self.config = config or MultiAgentConfig()

    async def run(
        self,
        task: str,
        context: str = "",
        run_id: str | None = None,
    ) -> MultiAgentRunResult:
        """Run a multi-agent workflow on a task.

        Args:
            task: The complex task to solve.
            context: Additional context/background.
            run_id: Optional run identifier (auto-generated if not provided).

        Returns:
            MultiAgentRunResult with full trace and results.
        """
        import uuid
        run_id = run_id or f"multi-{uuid.uuid4().hex[:8]}"

        t0 = time.time()

        initial_state = create_multi_agent_state(
            task=task,
            context=context,
            worker_roles=self.config.worker_roles,
        )

        # Auto-detect: use simulated supervisor if no LLM API key available
        import os
        has_llm = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        if has_llm:
            from .topology import build_multi_agent_graph
            graph = build_multi_agent_graph(
                worker_fn=self._worker_fn,
                model_name=self.config.model,
                temperature=self.config.temperature,
                worker_roles=self.config.worker_roles,
            )
        else:
            from .topology import build_simulated_multi_agent_graph
            graph = build_simulated_multi_agent_graph(
                worker_fn=self._worker_fn,
                worker_roles=self.config.worker_roles,
            )

        # Execute the graph
        config_dict = {"configurable": {"thread_id": run_id}}

        async for event in graph.astream(initial_state, config_dict):
            pass  # Stream to completion; state is accumulated in checkpointer

        # Get full accumulated state from checkpointer
        snapshot = graph.get_state(config_dict)
        if snapshot and snapshot.values:
            final_state = snapshot.values
        else:
            final_state = initial_state

        total_latency = (time.time() - t0) * 1000

        # Extract all inter-agent messages from the state
        all_msgs = final_state.get("inter_agent_messages", [])

        # Extract worker results
        worker_results = final_state.get("worker_results", [])

        return MultiAgentRunResult(
            run_id=run_id,
            task=task,
            subtasks=final_state.get("subtasks", []),
            assignments=[dict(a) for a in final_state.get("assignments", [])],
            worker_results=worker_results,
            inter_agent_messages=all_msgs,
            aggregated_answer=final_state.get("aggregated_answer", ""),
            final_answer=final_state.get("final_answer", ""),
            verification_passed=final_state.get("verification_passed", False),
            grounded_claims=final_state.get("grounded_claims", []),
            ungrounded_claims=final_state.get("ungrounded_claims", []),
            citations_used=final_state.get("citations_used", []),
            coordination_trace=final_state.get("coordination_trace", []),
            error=final_state.get("error"),
            total_latency_ms=total_latency,
            worker_count=len(worker_results),
            success=final_state.get("verification_passed", False) and not final_state.get("error"),
        )

    def run_sync(self, task: str, context: str = "", run_id: str | None = None) -> MultiAgentRunResult:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(task, context, run_id))

    # ── Static helpers for building worker functions ─────────────────

    @staticmethod
    def make_simulated_worker_fn(
        profile_name: str = "production",
    ) -> Callable[[str, str, str], WorkerResult]:
        """Create a worker function backed by the simulated agent.

        Args:
            profile_name: SimConfig profile ('perfect', 'production', 'development', 'unreliable').

        Returns:
            A callable suitable for use as worker_fn in MultiAgentCoordinator.
        """
        from ..evals.simulator import get_profile, SimulatedAgent

        sim_config = get_profile(profile_name)
        if sim_config is None:
            raise ValueError(f"Unknown profile: {profile_name}")

        def _worker(role: str, subtask: str, context: str) -> WorkerResult:
            import uuid, asyncio
            agent = SimulatedAgent(config=sim_config, seed=hash(f"{role}:{subtask}") & 0xFFFFFFFF)

            full_task = subtask
            if context:
                full_task = f"{subtask}\n\nContext: {context}"

            result = asyncio.run(agent.run(full_task, task_id=f"worker-{uuid.uuid4().hex[:6]}"))

            return WorkerResult(
                assignment_id=f"asgn-{uuid.uuid4().hex[:6]}",
                worker_role=role,
                subtask=subtask,
                answer=result.final_answer,
                grounded_claims=result.grounded_claims,
                ungrounded_claims=result.ungrounded_claims,
                citations_used=result.citations_used,
                verification_passed=result.verification_passed,
                tool_calls_count=result.tool_calls_count,
                retrieved_chunks_count=result.retrieved_chunks_count,
                latency_ms=result.total_latency_ms,
                error=result.error,
            )

        return _worker

    @staticmethod
    def make_relability_worker_fn(
        tool_registry,
        retrieval_fn,
        model: str = "gpt-4o",
    ) -> Callable[[str, str, str], WorkerResult]:
        """Create a worker function backed by the real ReliabilityAgent.

        Args:
            tool_registry: ToolRegistry instance.
            retrieval_fn: Retrieval function (query, k) -> list[RetrievalResult].
            model: LLM model name.

        Returns:
            A callable suitable for use as worker_fn in MultiAgentCoordinator.
        """
        from ..agent.implementations import ReliabilityAgent

        def _worker(role: str, subtask: str, context: str) -> WorkerResult:
            import uuid

            agent = ReliabilityAgent(
                tool_registry=tool_registry,
                retrieval_fn=retrieval_fn,
                model=model,
            )

            full_task = subtask
            if context:
                full_task = f"{subtask}\n\nContext: {context}"

            result = agent.run_sync(full_task, task_id=f"worker-{role}-{uuid.uuid4().hex[:6]}")

            return WorkerResult(
                assignment_id=f"asgn-{uuid.uuid4().hex[:6]}",
                worker_role=role,
                subtask=subtask,
                answer=result.final_answer,
                grounded_claims=result.grounded_claims,
                ungrounded_claims=result.ungrounded_claims,
                citations_used=result.citations_used,
                verification_passed=result.verification_passed,
                tool_calls_count=result.tool_calls_count,
                retrieved_chunks_count=result.retrieved_chunks_count,
                latency_ms=result.total_latency_ms,
                error=result.error,
            )

        return _worker


# ── Trace store extensions for multi-agent ───────────────────────────


def extend_trace_store_for_multi_agent(store) -> None:
    """Add multi-agent tracing tables to an existing TraceStore.

    Extends the SQLite schema with:
    - multi_agent_runs: top-level multi-agent workflow runs
    - inter_agent_messages: message passing between agents
    - worker_assignments: subtask routing details

    Args:
        store: TraceStore instance from agentops.tracing.store.
    """
    conn = store._get_conn()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS multi_agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            task TEXT NOT NULL,
            subtask_count INTEGER DEFAULT 0,
            worker_count INTEGER DEFAULT 0,
            verification_passed INTEGER DEFAULT 0,
            total_latency_ms REAL DEFAULT 0,
            coordination_trace TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_ma_runs_id ON multi_agent_runs(run_id);
        CREATE INDEX IF NOT EXISTS idx_ma_runs_verified ON multi_agent_runs(verification_passed);

        CREATE TABLE IF NOT EXISTS inter_agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            msg_id TEXT NOT NULL,
            from_agent TEXT NOT NULL,
            to_agent TEXT NOT NULL,
            msg_type TEXT NOT NULL,
            content TEXT DEFAULT '',
            timestamp_ms REAL DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (run_id) REFERENCES multi_agent_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_iam_run_id ON inter_agent_messages(run_id);
        CREATE INDEX IF NOT EXISTS idx_iam_msg_type ON inter_agent_messages(msg_type);
        CREATE INDEX IF NOT EXISTS idx_iam_agents ON inter_agent_messages(from_agent, to_agent);

        CREATE TABLE IF NOT EXISTS worker_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            assignment_id TEXT NOT NULL,
            worker_role TEXT NOT NULL,
            subtask TEXT NOT NULL,
            verification_passed INTEGER DEFAULT 0,
            tool_calls_count INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            error TEXT,
            FOREIGN KEY (run_id) REFERENCES multi_agent_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_wa_run_id ON worker_assignments(run_id);
    """)
    conn.commit()


def save_multi_agent_run(store, result: MultiAgentRunResult) -> str:
    """Save a complete multi-agent run to the trace store.

    Args:
        store: TraceStore instance (must have multi-agent tables).
        result: MultiAgentRunResult from a coordinator run.

    Returns:
        The run_id.
    """
    extend_trace_store_for_multi_agent(store)

    conn = store._get_conn()

    # Save the top-level run
    conn.execute(
        """INSERT OR REPLACE INTO multi_agent_runs
           (run_id, task, subtask_count, worker_count, verification_passed,
            total_latency_ms, coordination_trace)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            result.run_id,
            result.task,
            len(result.subtasks),
            result.worker_count,
            1 if result.verification_passed else 0,
            result.total_latency_ms,
            json.dumps(result.coordination_trace),
        ),
    )

    # Save inter-agent messages
    for msg in result.inter_agent_messages:
        conn.execute(
            """INSERT OR REPLACE INTO inter_agent_messages
               (run_id, msg_id, from_agent, to_agent, msg_type, content, timestamp_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.run_id,
                msg["msg_id"],
                msg["from_agent"],
                msg["to_agent"],
                msg["msg_type"],
                msg["content"],
                msg["timestamp_ms"],
                json.dumps(msg.get("metadata", {})),
            ),
        )

    # Save worker assignments
    for wr in result.worker_results:
        conn.execute(
            """INSERT OR REPLACE INTO worker_assignments
               (run_id, assignment_id, worker_role, subtask, verification_passed,
                tool_calls_count, latency_ms, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.run_id,
                wr.get("assignment_id", ""),
                wr.get("worker_role", ""),
                wr.get("subtask", ""),
                1 if wr.get("verification_passed", False) else 0,
                wr.get("tool_calls_count", 0),
                wr.get("latency_ms", 0),
                wr.get("error"),
            ),
        )

    conn.commit()
    return result.run_id
