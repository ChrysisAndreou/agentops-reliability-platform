"""
ReliabilityAgent — the primary agent implementation wrapping the
reliability graph with trace integration.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .graphs import build_reliability_graph
from .state import ReliabilityState, create_initial_state
from .tool_registry import ToolRegistry


@contextmanager
def _null_context():
    yield


@dataclass
class AgentRunResult:
    """Complete result of a reliability agent run."""
    task_id: str
    task: str
    final_answer: str
    success: bool
    error: str | None
    total_latency_ms: float
    verification_passed: bool
    verification_notes: str
    grounded_claims: list[str]
    ungrounded_claims: list[str]
    citations_used: list[str]
    plan: list[str]
    tool_calls_count: int
    retrieved_chunks_count: int
    reliability_trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "final_answer": self.final_answer,
            "success": self.success,
            "error": self.error,
            "total_latency_ms": self.total_latency_ms,
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
            "grounded_claims": self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "citations_used": self.citations_used,
            "plan": self.plan,
            "tool_calls_count": self.tool_calls_count,
            "retrieved_chunks_count": self.retrieved_chunks_count,
            "reliability_trace_length": len(self.reliability_trace),
        }


class ReliabilityAgent:
    """An AI agent that follows a reliability-oriented workflow.

    Uses the plan → retrieve → execute → verify → respond graph
    with full traceability and verification gating.

    Usage:
        agent = ReliabilityAgent(
            tool_registry=registry,
            retrieval_fn=my_retrieval_function,
            model="gpt-4o",
        )
        result = await agent.run("How do I reset my password?")
        print(f"Verified: {result.verification_passed}")
        print(f"Answer: {result.final_answer}")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        retrieval_fn,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        otel_observer: Any = None,
    ):
        self.tool_registry = tool_registry
        self.retrieval_fn = retrieval_fn
        self.model = model
        self.temperature = temperature
        self._otel = otel_observer
        self._graph = build_reliability_graph(
            tool_registry=tool_registry,
            retrieval_fn=retrieval_fn,
            model_name=model,
            temperature=temperature,
        )

    async def run(self, task: str, task_id: str = "default", context: str = "") -> AgentRunResult:
        """Execute the agent on a task and return a structured result."""
        start_time = time.time()

        state = create_initial_state(task, context)
        config = {"configurable": {"thread_id": task_id}}

        final_state = None
        error = None

        # OTEL trace wrapper
        otel = self._otel
        trace_ctx = (
            otel.trace_run(task=task, task_id=task_id, model=self.model)
            if otel and otel.enabled
            else None
        )
        if trace_ctx is None:
            trace_ctx = _null_context()

        with trace_ctx:
            try:
                for event in self._graph.stream(state, config):
                    if "__end__" in event:
                        continue
                    final_state = list(event.values())[-1] if event else None
            except Exception as e:
                error = str(e)

        total_latency = (time.time() - start_time) * 1000

        if final_state is None:
            result = AgentRunResult(
                task_id=task_id,
                task=task,
                final_answer="",
                success=False,
                error=error or "No final state reached",
                total_latency_ms=total_latency,
                verification_passed=False,
                verification_notes="",
                grounded_claims=[],
                ungrounded_claims=[],
                citations_used=[],
                plan=[],
                tool_calls_count=0,
                retrieved_chunks_count=0,
                reliability_trace=[],
            )
            self._record_otel_result(otel, result)
            return result

        result = AgentRunResult(
            task_id=task_id,
            task=task,
            final_answer=final_state.get("final_answer", ""),
            success=final_state.get("done", False) and not error,
            error=error or final_state.get("error"),
            total_latency_ms=total_latency,
            verification_passed=final_state.get("verification_passed", False),
            verification_notes=final_state.get("verification_notes", ""),
            grounded_claims=final_state.get("grounded_claims", []),
            ungrounded_claims=final_state.get("ungrounded_claims", []),
            citations_used=final_state.get("citations_used", []),
            plan=final_state.get("plan", []),
            tool_calls_count=len(final_state.get("tool_calls_made", [])),
            retrieved_chunks_count=len(final_state.get("retrieved_chunks", [])),
            reliability_trace=[
                dict(t) if hasattr(t, "__iter__") and not isinstance(t, str) else t
                for t in final_state.get("reliability_trace", [])
            ],
        )

        # Record step spans from reliability_trace
        if otel and otel.enabled:
            for step in result.reliability_trace:
                if isinstance(step, dict):
                    otel.record_step(
                        step_name=step.get("step_name", "unknown"),
                        step_type=step.get("step_type", "unknown"),
                        latency_ms=step.get("latency_ms", 0),
                        extra={
                            k: v for k, v in step.items()
                            if k not in ("step_name", "step_type", "latency_ms")
                        },
                    )

        self._record_otel_result(otel, result)
        return result

    def _record_otel_result(self, otel: Any, result: AgentRunResult) -> None:
        if otel is None or not otel.enabled:
            return
        grounded = len(result.grounded_claims)
        ungrounded = len(result.ungrounded_claims)
        grounded_ratio = grounded / max(grounded + ungrounded, 1)
        otel.record_result(
            success=result.success,
            verification_passed=result.verification_passed,
            grounded_ratio=grounded_ratio,
            tool_calls=result.tool_calls_count,
            total_latency_ms=result.total_latency_ms,
            retrieved_chunks=result.retrieved_chunks_count,
            failure_mode="" if result.success else
                ("verification" if not result.verification_passed else "error"),
            error=result.error or "",
        )

    def reset(self) -> None:
        """Reset agent state for a fresh run."""
        self._graph = build_reliability_graph(
            tool_registry=self.tool_registry,
            retrieval_fn=self.retrieval_fn,
            model_name=self.model,
            temperature=self.temperature,
        )
        self.tool_registry.reset()
