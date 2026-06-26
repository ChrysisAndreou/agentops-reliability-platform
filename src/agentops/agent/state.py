"""
Typed state schemas for agent orchestration graphs.

Defines the state containers used by LangGraph workflows, including
the core ReliabilityState that drives the reliability agent's
plan → retrieve → execute → verify → respond pipeline.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage


class AgentState(TypedDict, total=False):
    """State for a single-agent graph with tool use."""
    messages: Annotated[list[BaseMessage], operator.add]
    task: str
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    final_answer: str
    step_count: int
    done: bool
    error: str


class RetrievalResult(TypedDict):
    """A single retrieved document chunk with citation metadata."""
    chunk_id: str
    content: str
    source: str
    score: float
    retrieval_method: Literal["bm25", "dense", "hybrid"]


class ReliabilityStep(TypedDict):
    """A single step in the reliability trace."""
    step_name: str
    step_type: Literal["plan", "retrieve", "tool_call", "verify", "respond"]
    input_summary: str
    output_summary: str
    tool_calls: list[dict[str, Any]]
    retrieved_chunks: list[str]
    verification_passed: bool | None
    latency_ms: float
    error: str | None


class ReliabilityState(TypedDict, total=False):
    """State for the reliability agent workflow.

    The agent goes through: plan → retrieve → tool_execute → verify → respond,
    with full traceability at each step.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    task: str
    task_context: str

    # Planning
    plan: list[str]

    # Retrieval
    retrieved_chunks: list[RetrievalResult]
    citation_map: dict[str, str]

    # Tool execution
    tool_calls_made: list[dict[str, Any]]
    tool_results_raw: list[str]

    # Verification
    verification_passed: bool
    verification_notes: str
    grounded_claims: list[str]
    ungrounded_claims: list[str]

    # Response
    final_answer: str
    citations_used: list[str]

    # Control
    step_count: int
    current_step: str
    done: bool
    error: str

    # Trace
    reliability_trace: Annotated[list[ReliabilityStep], operator.add]


def create_initial_state(task: str, context: str = "") -> ReliabilityState:
    """Create the initial state for a reliability agent run."""
    return ReliabilityState(
        messages=[HumanMessage(content=task)],
        task=task,
        task_context=context,
        plan=[],
        retrieved_chunks=[],
        citation_map={},
        tool_calls_made=[],
        tool_results_raw=[],
        verification_passed=False,
        verification_notes="",
        grounded_claims=[],
        ungrounded_claims=[],
        final_answer="",
        citations_used=[],
        step_count=0,
        current_step="init",
        done=False,
        error="",
        reliability_trace=[],
    )
