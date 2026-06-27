"""
AgentOps tracer — high-level instrumentation API for AI agents.

Provides decorators, context managers, and logging functions that
instrument agent code with tracing and send results to an AgentOps server.

Usage:
    import agentops

    agentops.init(endpoint="http://localhost:8000")

    @agentops.trace()
    async def my_agent(task: str) -> str:
        agentops.log_retrieval(query=task, chunks=["doc1"])
        agentops.log_tool_call("search", {"query": task}, output="results")
        return "answer"

    # Or with explicit context manager:
    with agentops.start_run(task="Analyze deployment logs") as run:
        agentops.log_tool_call("fetch_logs", {"service": "api"})
        run.final_answer = "3 errors found"
        run.verification_passed = True
"""

from __future__ import annotations

import contextvars
import functools
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable

from .state import (
    SDKConfig,
    TraceSpan,
    SpanKind,
    SpanStatus,
    RunContext,
    ToolCallRecord,
    RetrievalRecord,
    TraceStatus,
)
from .client import AgentOpsHTTPClient


# Context variable for the currently active run — enables log_*() helpers
# without passing the context explicitly through every function.
_active_run: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "agentops_active_run", default=None
)

# Global SDK instance (singleton pattern for simplicity)
_global_agentops: AgentOps | None = None


def _get_client() -> AgentOps | None:
    """Get the global AgentOps client, if initialized."""
    return _global_agentops


class AgentOps:
    """Main entry point for the AgentOps SDK.

    Initialize once at application startup, then use decorators and helpers
    throughout your agent code.

    Usage:
        agentops = AgentOps()
        agentops.init(endpoint="http://localhost:8000", project_name="my-agent")

        @agentops.trace()
        async def my_agent(task: str) -> str:
            ...
    """

    def __init__(self):
        self.config: SDKConfig | None = None
        self.client: AgentOpsHTTPClient | None = None
        self._initialized = False

    def init(
        self,
        endpoint: str = "http://localhost:8000",
        api_key: str | None = None,
        project_name: str = "default",
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> bool:
        """Initialize the AgentOps SDK and connect to the server.

        Args:
            endpoint: URL of the AgentOps API server.
            api_key: Optional API key for authenticated servers.
            project_name: Name of the project for trace grouping.
            enabled: If False, all tracing becomes no-ops (for dev/prod toggle).
            metadata: Key-value metadata attached to all runs.
            **kwargs: Additional config overrides (timeout_seconds, max_retries, etc.)

        Returns:
            True if the server is reachable and healthy.
        """
        self.config = SDKConfig(
            endpoint=endpoint,
            api_key=api_key,
            project_name=project_name,
            enabled=enabled,
            metadata=metadata or {},
            **{
                k: v
                for k, v in kwargs.items()
                if k in SDKConfig.__dataclass_fields__
            },
        )
        self.client = AgentOpsHTTPClient(self.config)

        if enabled:
            self._initialized = self.client.health_check()
        else:
            self._initialized = False
            self.client = None

        # Register as global
        global _global_agentops
        _global_agentops = self

        return self._initialized

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.client is not None

    def trace(
        self,
        task: str | None = None,
        task_id: str | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """Decorator to trace a function as an agent run.

        Usage:
            @agentops.trace()
            async def my_agent(task: str) -> str:
                ...

            @agentops.trace(model="gpt-4o", metadata={"version": "1.2"})
            def handler(query: str) -> str:
                ...

        The wrapped function's first argument is used as the task description
        if not explicitly provided. The return value becomes the final answer.
        Exceptions are captured as errors.
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Determine task from args/kwargs or explicit parameter
                resolved_task = task
                if resolved_task is None:
                    if args:
                        resolved_task = str(args[0])
                    elif "task" in kwargs:
                        resolved_task = str(kwargs["task"])
                    else:
                        resolved_task = func.__name__

                run = start_run(
                    task=resolved_task,
                    task_id=task_id or str(uuid.uuid4())[:8],
                    model=model,
                    metadata=metadata,
                )

                try:
                    with run as ctx:
                        result = func(*args, **kwargs)
                        ctx.final_answer = str(result)
                    return result
                except Exception as e:
                    # run might be partially set up
                    try:
                        with run as ctx:
                            ctx.finish(success=False, error=str(e))
                    except Exception:
                        pass
                    raise

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                resolved_task = task
                if resolved_task is None:
                    if args:
                        resolved_task = str(args[0])
                    elif "task" in kwargs:
                        resolved_task = str(kwargs["task"])
                    else:
                        resolved_task = func.__name__

                run = start_run(
                    task=resolved_task,
                    task_id=task_id or str(uuid.uuid4())[:8],
                    model=model,
                    metadata=metadata,
                )

                try:
                    with run as ctx:
                        result = await func(*args, **kwargs)
                        ctx.final_answer = str(result)
                    return result
                except Exception as e:
                    try:
                        with run as ctx:
                            ctx.finish(success=False, error=str(e))
                    except Exception:
                        pass
                    raise

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return decorator

    def start_run(
        self,
        task: str = "",
        task_id: str | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RunContext:
        """Start a new agent run and return a RunContext.

        Use as a context manager for explicit trace boundaries:

            with agentops.start_run(task="Analyze logs") as run:
                agentops.log_tool_call("fetch_logs", {"service": "api"})
                run.final_answer = "analysis result"
                run.verification_passed = True

        The run is automatically finished and submitted when the context exits.
        """
        return start_run(
            task=task,
            task_id=task_id,
            model=model,
            metadata=metadata,
        )

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any = None,
        success: bool = True,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Log a tool call to the current active run."""
        log_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            success=success,
            error=error,
            latency_ms=latency_ms,
        )

    def log_retrieval(
        self,
        query: str,
        chunks: list[str] | None = None,
        scores: list[float] | None = None,
        method: str = "hybrid",
        latency_ms: float = 0.0,
    ) -> None:
        """Log a retrieval operation to the current active run."""
        log_retrieval(
            query=query,
            chunks=chunks,
            scores=scores,
            method=method,
            latency_ms=latency_ms,
        )

    def log_verification(
        self,
        passed: bool,
        notes: str = "",
        grounded_claims: list[str] | None = None,
        ungrounded_claims: list[str] | None = None,
    ) -> None:
        """Log a verification decision to the current active run."""
        log_verification(
            passed=passed,
            notes=notes,
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
        )

    # ── Query API ──────────────────────────────────────────────────────

    def list_traces(
        self,
        verification_passed: bool | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query traces from the server."""
        if not self.client:
            return []
        return self.client.list_traces(
            verification_passed=verification_passed, limit=limit
        )

    def get_trace(self, run_id: str) -> dict[str, Any] | None:
        """Get full trace detail."""
        if not self.client:
            return None
        return self.client.get_trace(run_id)

    def get_replay(self, run_id: str) -> dict[str, Any] | None:
        """Get replay data for a trace."""
        if not self.client:
            return None
        return self.client.get_replay(run_id)

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics."""
        if not self.client:
            return {"error": "Not connected"}
        return self.client.get_stats()

    def list_evals(self) -> dict[str, Any]:
        """List evaluation runs."""
        if not self.client:
            return {"eval_runs": [], "count": 0}
        return self.client.list_evals()

    def get_eval(self, eval_id: str) -> dict[str, Any] | None:
        """Get an evaluation report."""
        if not self.client:
            return None
        return self.client.get_eval(eval_id)


# ── Module-level convenience functions ─────────────────────────────────

def init(
    endpoint: str = "http://localhost:8000",
    api_key: str | None = None,
    project_name: str = "default",
    enabled: bool = True,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> bool:
    """Initialize the AgentOps SDK globally (module-level convenience).

    After calling this, use the module-level trace(), start_run(),
    log_tool_call(), etc.

    Usage:
        import agentops
        agentops.init(endpoint="http://localhost:8000")

        @agentops.trace()
        async def my_agent(task: str) -> str:
            ...
    """
    aops = AgentOps()
    return aops.init(
        endpoint=endpoint,
        api_key=api_key,
        project_name=project_name,
        enabled=enabled,
        metadata=metadata,
        **kwargs,
    )


def trace(
    task: str | None = None,
    task_id: str | None = None,
    model: str = "",
    metadata: dict[str, Any] | None = None,
):
    """Module-level trace decorator. Requires agentops.init() first."""
    aops = _get_client()
    if aops is None:
        # No-op: return a decorator that just passes through
        def noop_decorator(func):
            return func
        return noop_decorator
    return aops.trace(task=task, task_id=task_id, model=model, metadata=metadata)


@contextmanager
def start_run(
    task: str = "",
    task_id: str | None = None,
    model: str = "",
    metadata: dict[str, Any] | None = None,
):
    """Start a new agent run context.

    Usage:
        with agentops.start_run(task="Query the database") as run:
            agentops.log_tool_call("db_query", {"sql": "SELECT ..."})
            run.final_answer = "42 rows"
    """
    aops = _get_client()
    ctx = RunContext(
        run_id=task_id or str(uuid.uuid4())[:8],
        task=task,
        task_id=task_id or str(uuid.uuid4())[:8],
        model=model,
        metadata=metadata or {},
    )

    # Create root span
    ctx.root_span = TraceSpan(
        span_id=ctx.run_id,
        kind=SpanKind.RUN,
        name=f"agentops.run:{task[:40] if task else 'unnamed'}",
    )

    # Set as active run
    token = _active_run.set(ctx)

    try:
        yield ctx
    except Exception:
        ctx.finish(success=False)
        raise
    finally:
        ctx.finish(success=ctx.status != TraceStatus.FAILED)
        _active_run.reset(token)
        # Submit to server if connected
        _flush_run(ctx)


def log_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any = None,
    success: bool = True,
    error: str | None = None,
    latency_ms: float = 0.0,
) -> None:
    """Log a tool call to the current active run.

    Safe to call from anywhere — becomes a no-op if no run is active.
    """
    ctx = _active_run.get()
    if ctx is None:
        return

    record = ToolCallRecord(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        success=success,
        error=error,
        latency_ms=latency_ms,
    )
    ctx.tool_calls.append(record)

    # Add as child span to root
    if ctx.root_span:
        span = TraceSpan(
            kind=SpanKind.TOOL,
            name=f"tool:{tool_name}",
            status=SpanStatus.OK if success else SpanStatus.ERROR,
            attributes={
                "tool_name": tool_name,
                "success": success,
                "error": error,
            },
        )
        span.finish()
        ctx.root_span.children.append(span)


def log_retrieval(
    query: str,
    chunks: list[str] | None = None,
    scores: list[float] | None = None,
    method: str = "hybrid",
    latency_ms: float = 0.0,
) -> None:
    """Log a retrieval operation to the current active run."""
    ctx = _active_run.get()
    if ctx is None:
        return

    record = RetrievalRecord(
        query=query,
        chunks_retrieved=len(chunks) if chunks else 0,
        top_chunk_scores=scores or [],
        retrieval_method=method,
        latency_ms=latency_ms,
    )
    ctx.retrievals.append(record)

    if ctx.root_span:
        span = TraceSpan(
            kind=SpanKind.RETRIEVE,
            name=f"retrieve:{query[:40]}",
            attributes={
                "query": query,
                "chunks": len(chunks) if chunks else 0,
                "method": method,
            },
        )
        span.finish()
        ctx.root_span.children.append(span)


def log_verification(
    passed: bool,
    notes: str = "",
    grounded_claims: list[str] | None = None,
    ungrounded_claims: list[str] | None = None,
) -> None:
    """Log a verification decision to the current active run."""
    ctx = _active_run.get()
    if ctx is None:
        return

    ctx.verification_passed = passed
    ctx.verification_notes = notes
    if grounded_claims:
        ctx.grounded_claims = grounded_claims
    if ungrounded_claims:
        ctx.ungrounded_claims = ungrounded_claims

    if ctx.root_span:
        span = TraceSpan(
            kind=SpanKind.VERIFY,
            name="verify",
            status=SpanStatus.OK if passed else SpanStatus.WARNING,
            attributes={
                "passed": passed,
                "notes": notes,
            },
        )
        span.finish()
        ctx.root_span.children.append(span)


def get_current_run() -> RunContext | None:
    """Get the currently active run context, or None."""
    return _active_run.get()


def _flush_run(ctx: RunContext) -> None:
    """Submit a completed run to the server if connected."""
    aops = _get_client()
    if aops is None or not aops.is_ready or aops.client is None:
        return

    try:
        aops.client.submit_trace(ctx)
    except Exception:
        # Trace submission is best-effort — never crash agent code
        pass
