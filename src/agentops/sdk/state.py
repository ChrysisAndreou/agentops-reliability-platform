"""
SDK state models — typed schemas for the AgentOps client library.

All models are Pydantic-free to keep the SDK lightweight. Plain dataclasses
with serialization methods for JSON transport.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    """Kind of span in an agent trace."""
    RUN = "run"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    TOOL = "tool"
    VERIFY = "verify"
    RESPOND = "respond"
    LLM = "llm"


class SpanStatus(str, Enum):
    """Status of a span."""
    OK = "ok"
    ERROR = "error"
    WARNING = "warning"


class TraceStatus(str, Enum):
    """Overall trace status."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class ToolCallRecord:
    """Record of a tool invocation."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any = None
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class RetrievalRecord:
    """Record of a retrieval operation."""
    query: str
    chunks_retrieved: int = 0
    top_chunk_scores: list[float] = field(default_factory=list)
    retrieval_method: str = "hybrid"
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks_retrieved": self.chunks_retrieved,
            "top_chunk_scores": self.top_chunk_scores,
            "retrieval_method": self.retrieval_method,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class TraceSpan:
    """A single span in an agent trace — represents one step in the pipeline."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.RUN
    name: str = ""
    status: SpanStatus = SpanStatus.OK
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    children: list[TraceSpan] = field(default_factory=list)
    latency_ms: float = 0.0

    def finish(self, status: SpanStatus | None = None):
        """Mark the span as finished."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        if status:
            self.status = status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add a timestamped event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "attributes": self.attributes,
            "events": self.events,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class RunContext:
    """Context for an active agent run — accumulates spans and metadata."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task: str = ""
    task_id: str = ""
    model: str = ""
    status: TraceStatus = TraceStatus.RUNNING
    start_time: float = field(default_factory=time.time)
    root_span: TraceSpan | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    retrievals: list[RetrievalRecord] = field(default_factory=list)
    verification_passed: bool | None = None
    verification_notes: str = ""
    final_answer: str = ""
    error: str | None = None
    grounded_claims: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    citations_used: list[str] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, success: bool = True, error: str | None = None):
        """Mark the run as finished."""
        self.status = TraceStatus.SUCCESS if success else TraceStatus.FAILED
        self.error = error
        if self.root_span:
            self.root_span.finish(
                SpanStatus.OK if success else SpanStatus.ERROR
            )

    @property
    def latency_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "task_id": self.task_id,
            "model": self.model,
            "status": self.status.value,
            "start_time": self.start_time,
            "latency_ms": self.latency_ms,
            "root_span": self.root_span.to_dict() if self.root_span else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "retrievals": [r.to_dict() for r in self.retrievals],
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
            "final_answer": self.final_answer,
            "error": self.error,
            "grounded_claims": self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "citations_used": self.citations_used,
            "plan_steps": self.plan_steps,
            "metadata": self.metadata,
        }


@dataclass
class SDKConfig:
    """Configuration for the AgentOps SDK client."""
    endpoint: str = "http://localhost:8000"
    api_key: str | None = None
    project_name: str = "default"
    enabled: bool = True
    flush_interval_ms: int = 5000
    max_buffer_size: int = 100
    timeout_seconds: float = 30.0
    max_retries: int = 3
    verify_ssl: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOpsClient:
    """Typed response wrapper for API client calls."""
    endpoint: str
    connected: bool = False
    server_version: str = ""
    project_name: str = "default"

    def check_health(self) -> dict[str, Any]:
        """Stub — actual HTTP call made by AgentOpsHTTPClient."""
        return {"status": "unknown"}
