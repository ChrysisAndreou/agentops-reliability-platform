"""
Tests for OpenTelemetry integration in AgentOps.

Covers:
- Disabled-by-default behavior (no OTEL overhead when not configured)
- trace_run context manager
- record_step and record_result
- Metric counter and histogram recording
- Start/shutdown lifecycle
- Integration with ReliabilityAgent
- Graceful degradation when OTEL SDK unavailable
"""

import os
import pytest
from unittest import mock


# ── Ensure OTEL is disabled for tests by default ──────────────────────

@pytest.fixture(autouse=True)
def disable_otel_env():
    """Ensure AGENTOPS_OTEL_ENABLED is not set during tests."""
    old = os.environ.pop("AGENTOPS_OTEL_ENABLED", None)
    yield
    if old is not None:
        os.environ["AGENTOPS_OTEL_ENABLED"] = old


# ── Imports ───────────────────────────────────────────────────────────

from agentops.tracing.opentelemetry import OTelObserver, _NoOpSpan


class TestOTelObserverDisabled:
    """OTEL observer should be a no-op when not enabled."""

    def test_not_enabled_by_default(self):
        observer = OTelObserver()
        assert observer.enabled is False

    def test_explicitly_disabled(self):
        observer = OTelObserver(enabled=False)
        observer.start()
        assert observer.enabled is False

    def test_enabled_env_var_off(self):
        os.environ["AGENTOPS_OTEL_ENABLED"] = "0"
        observer = OTelObserver()
        assert observer.enabled is False

    def test_trace_run_noop_when_disabled(self):
        observer = OTelObserver(enabled=False)
        observer.start()
        with observer.trace_run(task="test task") as span:
            assert isinstance(span, _NoOpSpan)
            span.set_attribute("key", "value")

    def test_record_step_noop_when_disabled(self):
        observer = OTelObserver(enabled=False)
        observer.start()
        observer.record_step("planner", "plan", latency_ms=100.0)

    def test_record_result_noop_when_disabled(self):
        observer = OTelObserver(enabled=False)
        observer.start()
        observer.record_result(
            success=True, verification_passed=True,
            grounded_ratio=0.8, tool_calls=2, total_latency_ms=500.0,
        )

    def test_shutdown_noop_when_disabled(self):
        observer = OTelObserver(enabled=False)
        observer.shutdown()

    def test_shutdown_before_start_safe(self):
        observer = OTelObserver(enabled=False)
        observer.shutdown()


class TestOTelObserverEnabled:
    """Tests with OTEL enabled and mocked providers."""

    @pytest.fixture
    def otel_mocks(self):
        """Mock the OTEL trace/metric globals to isolate tests from real SDK."""
        mock_tracer = mock.MagicMock()
        mock_span = mock.MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = mock.MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = mock.MagicMock(
            return_value=None
        )
        mock_tracer.start_span.return_value = mock_span

        mock_meter = mock.MagicMock()
        mock_counter = mock.MagicMock()
        mock_histogram = mock.MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram

        # Patch the OTEL globals in the opentelemetry module
        with mock.patch(
            "agentops.tracing.opentelemetry.otel_trace"
        ) as mock_otel_trace, mock.patch(
            "agentops.tracing.opentelemetry.otel_metrics"
        ) as mock_otel_metrics:
            mock_otel_trace.set_tracer_provider = mock.MagicMock()
            mock_otel_trace.get_tracer.return_value = mock_tracer
            mock_otel_metrics.set_meter_provider = mock.MagicMock()
            mock_otel_metrics.get_meter.return_value = mock_meter

            # Patch OTEL_AVAILABLE to True
            with mock.patch(
                "agentops.tracing.opentelemetry.OTEL_AVAILABLE", True
            ):
                yield {
                    "tracer": mock_tracer,
                    "meter": mock_meter,
                    "span": mock_span,
                    "counter": mock_counter,
                    "histogram": mock_histogram,
                    "otel_trace": mock_otel_trace,
                    "otel_metrics": mock_otel_metrics,
                }

    def test_start_creates_providers(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()
        assert observer._started is True
        assert observer._tracer is not None
        assert observer._meter is not None

    def test_start_idempotent(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()
        observer.start()
        assert observer._started is True

    def test_trace_run_creates_span(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()

        with observer.trace_run(task="Test task", task_id="t1", model="gpt-4o") as span:
            span.set_attribute("custom", "value")

        assert observer._tracer is not None

    def test_record_step_records_histogram(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()

        observer.record_step("verifier", "verify", latency_ms=230.5,
                             extra={"grounded_claims": 3})

        otel_mocks["histogram"].record.assert_called()
        otel_mocks["tracer"].start_span.assert_called()

    def test_record_result_increments_counters(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()

        observer.record_result(
            success=True, verification_passed=True,
            grounded_ratio=0.9, tool_calls=3,
            total_latency_ms=1200.0, retrieved_chunks=7,
        )

        otel_mocks["counter"].add.assert_called()
        otel_mocks["histogram"].record.assert_called()

    def test_record_result_failure_mode(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()

        observer.record_result(
            success=False, verification_passed=False,
            failure_mode="hallucination", error="2 ungrounded claims",
        )

        otel_mocks["counter"].add.assert_called()

    def test_shutdown_cleans_up(self, otel_mocks):
        observer = OTelObserver(enabled=True)
        observer.start()
        observer.shutdown()
        assert observer._started is False

    def test_graceful_failure_on_exporter_error(self):
        """When OTEL exporter fails, observer should still work (no crash)."""
        with mock.patch(
            "agentops.tracing.opentelemetry.OTLPSpanExporter",
            side_effect=Exception("Connection refused"),
        ), mock.patch(
            "agentops.tracing.opentelemetry.OTEL_AVAILABLE", True
        ):
            observer = OTelObserver(enabled=True)
            observer.start()
            assert observer._tracer is not None

    def test_config_from_env(self, otel_mocks):
        os.environ["AGENTOPS_OTEL_ENABLED"] = "1"
        os.environ["OTEL_SERVICE_NAME"] = "test-service"
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://collector:4317"

        observer = OTelObserver()
        observer.start()
        assert observer.enabled is True
        assert observer._service_name == "test-service"
        assert observer._otlp_endpoint == "http://collector:4317"


class TestOTelObserverNotAvailable:
    """When OTEL SDK is not installed, everything should be no-op."""

    def test_not_available_flag(self):
        with mock.patch(
            "agentops.tracing.opentelemetry.OTEL_AVAILABLE", False
        ):
            observer = OTelObserver(enabled=True)
            observer.start()
            assert observer.enabled is False

    def test_trace_run_returns_noop(self):
        with mock.patch(
            "agentops.tracing.opentelemetry.OTEL_AVAILABLE", False
        ):
            observer = OTelObserver(enabled=True)
            observer.start()
            with observer.trace_run(task="test") as span:
                assert isinstance(span, _NoOpSpan)


class TestNoOpSpan:
    """_NoOpSpan should implement the full span interface safely."""

    def test_context_manager(self):
        with _NoOpSpan() as span:
            span.set_attribute("key", "value")
            span.set_status(None, "ok")
            span.add_event("something")

    def test_can_be_nested(self):
        with _NoOpSpan() as outer:
            with _NoOpSpan() as inner:
                inner.set_attribute("nested", True)


class TestOTelIntegrationWithAgent:
    """Test that ReliabilityAgent passes OTEL observer through."""

    def test_agent_accepts_otel_observer(self):
        from agentops.agent.implementations import ReliabilityAgent
        from agentops.agent.tool_registry import ToolRegistry

        def mock_retrieval(query, k=5):
            return []

        observer = OTelObserver(enabled=False)

        # Mock the graph build to avoid real LLM calls
        with mock.patch(
            "agentops.agent.implementations.build_reliability_graph"
        ) as mock_build:
            mock_build.return_value = mock.MagicMock()
            agent = ReliabilityAgent(
                tool_registry=ToolRegistry(),
                retrieval_fn=mock_retrieval,
                model="gpt-4o",
                otel_observer=observer,
            )
            assert agent._otel is observer
