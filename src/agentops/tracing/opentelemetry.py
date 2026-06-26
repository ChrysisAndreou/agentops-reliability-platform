"""
OpenTelemetry integration for AgentOps reliability platform.

Exports agent execution traces as OTLP spans and metrics to any
OTEL-compatible collector (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc.).

Each agent graph step (plan, retrieve, execute, verify, respond) becomes a
child span under a parent agent-run span. Span attributes carry verification
status, groundedness ratios, tool counts, latency, and failure modes —
enabling production observability dashboards and alerting.

Metrics exported:
- agentops.runs.total (counter) — total agent runs
- agentops.runs.verification_pass (counter) — runs that passed verification
- agentops.runs.failed (counter) — runs that failed
- agentops.latency_ms (histogram) — agent run latency distribution
- agentops.step.latency_ms (histogram) — per-step latency distribution
- agentops.retrieved_chunks (histogram) — chunks retrieved per run
- agentops.tool_calls (histogram) — tool calls per run

Configuration via environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT  — collector endpoint (default: http://localhost:4317)
    OTEL_SERVICE_NAME            — service name (default: agentops-reliability)
    AGENTOPS_OTEL_ENABLED        — set to "1" to enable (default: disabled)

If OpenTelemetry SDK is not installed or AGENTOPS_OTEL_ENABLED is not set,
all operations are no-ops with zero overhead — no spans created, no metrics
recorded. The trace store (SQLite) continues to work normally regardless.

Usage:
    from agentops.tracing.opentelemetry import OTelObserver

    observer = OTelObserver()
    observer.start()

    with observer.trace_run(task="How do I enable 2FA?") as span:
        # ... agent execution ...
        observer.record_step("planner", "plan", latency_ms=45.2)
        observer.record_step("retriever", "retrieve", latency_ms=120.5,
                             extra={"chunks_retrieved": 5})
        observer.record_step("verifier", "verify", latency_ms=230.1,
                             extra={"grounded_claims": 3, "ungrounded_claims": 1})
        observer.record_result(success=True, verification_passed=True,
                               grounded_ratio=0.75, tool_calls=2,
                               total_latency_ms=450.0)
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

# ── Optional imports — OTEL may not be installed ──────────────────────

OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
        OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import \
        PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import \
        OTLPMetricExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.trace import Status, StatusCode, SpanKind
    OTEL_AVAILABLE = True
except ImportError:
    pass


# ── Span attribute keys ──────────────────────────────────────────────

ATTR_TASK = "agentops.task"
ATTR_TASK_ID = "agentops.task_id"
ATTR_RUN_ID = "agentops.run_id"
ATTR_STEP_NAME = "agentops.step.name"
ATTR_STEP_TYPE = "agentops.step.type"
ATTR_VERIFICATION_PASSED = "agentops.verification_passed"
ATTR_SUCCESS = "agentops.success"
ATTR_GROUNDED_RATIO = "agentops.grounded_ratio"
ATTR_GROUNDED_CLAIMS = "agentops.grounded_claims"
ATTR_UNGROUNDED_CLAIMS = "agentops.ungrounded_claims"
ATTR_TOOL_CALLS = "agentops.tool_calls"
ATTR_RETRIEVED_CHUNKS = "agentops.retrieved_chunks"
ATTR_FAILURE_MODE = "agentops.failure_mode"
ATTR_ERROR = "agentops.error"
ATTR_MODEL = "agentops.model"


@dataclass
class _NoOpSpan:
    """Stand-in when OTEL is disabled. Supports context manager protocol."""

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass


class OTelObserver:
    """Observes agent runs and exports traces + metrics via OpenTelemetry.

    When AGENTOPS_OTEL_ENABLED is not "1", or the OpenTelemetry SDK is not
    installed, this becomes a zero-overhead no-op. All methods return safely
    and the trace store (SQLite) works as before.
    """

    def __init__(
        self,
        service_name: str | None = None,
        otlp_endpoint: str | None = None,
        enabled: bool | None = None,
    ):
        self._enabled = enabled if enabled is not None else (
            os.environ.get("AGENTOPS_OTEL_ENABLED", "") == "1"
        )
        self._service_name = (
            service_name
            or os.environ.get("OTEL_SERVICE_NAME", "agentops-reliability")
        )
        self._otlp_endpoint = (
            otlp_endpoint
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        )

        self._tracer_provider: Any = None
        self._meter_provider: Any = None
        self._tracer: Any = None
        self._meter: Any = None

        # Metric instruments (set up on start())
        self._run_counter: Any = None
        self._verification_counter: Any = None
        self._failure_counter: Any = None
        self._latency_histogram: Any = None
        self._step_latency_histogram: Any = None
        self._chunks_histogram: Any = None
        self._tool_calls_histogram: Any = None

        self._started = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialize OTEL providers and exporters.

        Call once at application startup. Safe to call multiple times —
        subsequent calls are no-ops.
        """
        if not self._enabled or not OTEL_AVAILABLE:
            return
        if self._started:
            return

        resource = Resource(attributes={SERVICE_NAME: self._service_name})

        # Trace provider
        self._tracer_provider = TracerProvider(resource=resource)
        try:
            span_exporter = OTLPSpanExporter(endpoint=self._otlp_endpoint, insecure=True)
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(span_exporter)
            )
        except Exception:
            # Collector not reachable — traces queued in memory, no crash
            pass
        otel_trace.set_tracer_provider(self._tracer_provider)
        self._tracer = otel_trace.get_tracer(__name__)

        # Meter provider
        try:
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=self._otlp_endpoint, insecure=True),
                export_interval_millis=15_000,
            )
        except Exception:
            metric_reader = PeriodicExportingMetricReader(
                export_interval_millis=15_000,
            )
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        otel_metrics.set_meter_provider(self._meter_provider)
        self._meter = otel_metrics.get_meter(__name__)

        # Counters
        self._run_counter = self._meter.create_counter(
            "agentops.runs.total",
            description="Total agent runs executed",
        )
        self._verification_counter = self._meter.create_counter(
            "agentops.runs.verification_pass",
            description="Agent runs that passed verification",
        )
        self._failure_counter = self._meter.create_counter(
            "agentops.runs.failed",
            description="Agent runs that failed (error or verification)",
        )

        # Histograms
        self._latency_histogram = self._meter.create_histogram(
            "agentops.latency_ms",
            description="Agent run latency distribution (milliseconds)",
            unit="ms",
        )
        self._step_latency_histogram = self._meter.create_histogram(
            "agentops.step.latency_ms",
            description="Per-step latency distribution (milliseconds)",
            unit="ms",
        )
        self._chunks_histogram = self._meter.create_histogram(
            "agentops.retrieved_chunks",
            description="Chunks retrieved per agent run",
        )
        self._tool_calls_histogram = self._meter.create_histogram(
            "agentops.tool_calls",
            description="Tool calls per agent run",
        )

        self._started = True

    def shutdown(self) -> None:
        """Flush and shutdown OTEL providers. Call at application shutdown."""
        if not self._started:
            return
        try:
            if self._tracer_provider:
                self._tracer_provider.shutdown()
        except Exception:
            pass
        try:
            if self._meter_provider:
                self._meter_provider.shutdown()
        except Exception:
            pass
        self._started = False

    # ── Tracing ────────────────────────────────────────────────────────

    @contextmanager
    def trace_run(
        self,
        task: str,
        task_id: str = "",
        run_id: str = "",
        model: str = "",
    ) -> Generator[Any, None, None]:
        """Context manager that creates a parent span for an agent run.

        All step spans should be created as children of this span.
        """
        if not self._started or self._tracer is None:
            yield _NoOpSpan()
            return

        span_name = f"agent.run: {task[:80]}"
        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.INTERNAL,
        ) as span:
            start_time = time.time()
            span.set_attribute(ATTR_TASK, task)
            if task_id:
                span.set_attribute(ATTR_TASK_ID, task_id)
            if run_id:
                span.set_attribute(ATTR_RUN_ID, run_id)
            if model:
                span.set_attribute(ATTR_MODEL, model)

            try:
                yield span
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.set_attribute(ATTR_ERROR, str(exc))
                span.set_attribute(ATTR_SUCCESS, False)
                raise
            finally:
                elapsed = (time.time() - start_time) * 1000
                span.set_attribute("agentops.total_latency_ms", elapsed)

    def record_step(
        self,
        step_name: str,
        step_type: str,
        latency_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a step span within the current agent run.

        Args:
            step_name: Human-readable step name (e.g., 'planner', 'retriever').
            step_type: Step type (plan, retrieve, tool_call, verify, respond).
            latency_ms: Step execution latency in milliseconds.
            extra: Additional span attributes (chunks_retrieved, tool_calls, etc.).
        """
        if not self._started or self._tracer is None:
            return

        span = self._tracer.start_span(
            f"agent.step.{step_type}: {step_name}",
            kind=SpanKind.INTERNAL,
            start_time=None,
        )
        span.set_attribute(ATTR_STEP_NAME, step_name)
        span.set_attribute(ATTR_STEP_TYPE, step_type)
        span.set_attribute("agentops.step.latency_ms", latency_ms)

        if extra:
            for key, value in extra.items():
                span.set_attribute(f"agentops.step.{key}", value)

        # Record latency in histogram
        if self._step_latency_histogram:
            self._step_latency_histogram.record(
                latency_ms,
                attributes={"step_type": step_type, "step_name": step_name},
            )

        span.end()

    def record_result(
        self,
        success: bool,
        verification_passed: bool,
        grounded_ratio: float = 0.0,
        tool_calls: int = 0,
        total_latency_ms: float = 0.0,
        retrieved_chunks: int = 0,
        failure_mode: str = "",
        error: str = "",
    ) -> None:
        """Record the outcome of an agent run as metrics.

        Call this after the agent run completes (inside or after trace_run).
        """
        if not self._started:
            return

        # Counters
        if self._run_counter:
            self._run_counter.add(1)
        if verification_passed and self._verification_counter:
            self._verification_counter.add(1)
        if (not success or not verification_passed) and self._failure_counter:
            self._failure_counter.add(
                1,
                attributes={"failure_mode": failure_mode or "unknown"},
            )

        # Histograms
        if self._latency_histogram:
            self._latency_histogram.record(total_latency_ms)
        if self._chunks_histogram:
            self._chunks_histogram.record(retrieved_chunks)
        if self._tool_calls_histogram:
            self._tool_calls_histogram.record(tool_calls)

    # ── Status query ───────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled and OTEL_AVAILABLE and self._started
