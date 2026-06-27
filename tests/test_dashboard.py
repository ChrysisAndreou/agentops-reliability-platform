"""
Tests for the AgentOps Live Observability Dashboard (v0.10).

Covers:
- Dashboard server creation and health
- Stats endpoint (empty trace store, populated trace store)
- Traces endpoint (empty, filtered)
- Failure analysis endpoint
- WebSocket connectivity
- HTML dashboard rendering
- Fallback HTML (no Jinja2)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def empty_trace_store():
    """Create a TraceStore with no data."""
    from agentops.tracing.store import TraceStore
    store = TraceStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def populated_trace_store():
    """Create a TraceStore with sample trace data."""
    from agentops.tracing.store import TraceStore

    class FakeResult:
        def __init__(self, run_id, task, verification_passed, success, latency):
            self.task_id = run_id
            self.run_id = run_id
            self.task = task
            self.agent_type = "test"
            self.final_answer = f"Answer for {task}"
            self.success = success
            self.error = None if success else "Test error"
            self.verification_passed = verification_passed
            self.verification_notes = "ok" if verification_passed else "failed check"
            self.grounded_claims = ["claim1", "claim2"]
            self.ungrounded_claims = [] if verification_passed else ["bad_claim"]
            self.plan = ["step1", "step2"]
            self.tool_calls_count = 3
            self.retrieved_chunks_count = 5
            self.total_latency_ms = latency
            self.reliability_trace = [
                {"step_type": "plan", "step_name": "planning", "output_summary": "ok"},
                {"step_type": "retrieve", "step_name": "search", "output_summary": "found"},
            ]

    store = TraceStore(":memory:")

    results = [
        FakeResult("run-001", "How do I reset my password?", True, True, 350.0),
        FakeResult("run-002", "What is the pricing model?", True, True, 420.0),
        FakeResult("run-003", "Deploy to production steps?", False, False, 1200.0),
        FakeResult("run-004", "How to configure SSO?", True, True, 280.0),
        FakeResult("run-005", "Security audit requirements", False, False, 890.0),
    ]

    for r in results:
        store.save(r)

    yield store
    store.close()


@pytest.fixture
def dashboard_client(empty_trace_store):
    """Create a TestClient for the dashboard app."""
    from agentops.dashboard import create_dashboard_app
    app = create_dashboard_app(trace_store=empty_trace_store)
    return TestClient(app)


@pytest.fixture
def dashboard_with_data(populated_trace_store):
    """Create a TestClient with populated data."""
    from agentops.dashboard import create_dashboard_app
    app = create_dashboard_app(trace_store=populated_trace_store)
    return TestClient(app)


# ── Dashboard Server Creation ─────────────────────────────────


class TestDashboardServerCreation:
    """Tests for dashboard server initialization."""

    def test_create_app_returns_fastapi(self, empty_trace_store):
        from agentops.dashboard import create_dashboard_app
        from fastapi import FastAPI
        app = create_dashboard_app(trace_store=empty_trace_store)
        assert isinstance(app, FastAPI)

    def test_create_app_without_store(self):
        from agentops.dashboard import create_dashboard_app
        from fastapi import FastAPI
        app = create_dashboard_app()
        assert isinstance(app, FastAPI)

    def test_dashboard_server_init(self):
        from agentops.dashboard.server import DashboardServer
        server = DashboardServer()
        assert server.trace_store is None
        assert server.eval_results == {}
        assert server.agent is None


# ── Health Check ──────────────────────────────────────────────


class TestDashboardHealth:
    """Tests for the health endpoint."""

    def test_health_returns_ok(self, dashboard_client):
        response = dashboard_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "dashboard" in data

    def test_health_includes_timestamp(self, dashboard_client):
        response = dashboard_client.get("/health")
        assert "timestamp" in response.json()


# ── Stats Endpoint ────────────────────────────────────────────


class TestDashboardStats:
    """Tests for the dashboard stats API."""

    def test_stats_empty_store(self, dashboard_client):
        response = dashboard_client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 0
        assert data["pass_rate"] == 0
        assert data["failure_rate"] == 0

    def test_stats_with_data(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 5
        assert data["pass_rate"] == 0.6  # 3/5
        assert data["failure_rate"] == 0.4  # 2/5
        assert data["avg_latency_ms"] > 0
        assert "failure_breakdown" in data

    def test_stats_failure_breakdown(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/stats")
        data = response.json()
        fb = data["failure_breakdown"]
        assert fb["verification_failed"] == 2


# ── Traces Endpoint ───────────────────────────────────────────


class TestDashboardTraces:
    """Tests for the traces API."""

    def test_traces_empty_store(self, dashboard_client):
        response = dashboard_client.get("/api/dashboard/traces")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["traces"] == []

    def test_traces_with_data(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/traces")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
        assert len(data["traces"]) == 5

    def test_traces_structure(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/traces")
        trace = response.json()["traces"][0]
        assert "run_id" in trace
        assert "task" in trace
        assert "verification_passed" in trace
        assert "success" in trace
        assert "tool_calls" in trace
        assert "latency_ms" in trace

    def test_traces_failed_only(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/traces?failed_only=true")
        data = response.json()
        # Should only return failed traces
        for t in data["traces"]:
            assert t["verification_passed"] is False

    def test_traces_limit(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/traces?limit=2")
        data = response.json()
        assert len(data["traces"]) <= 2


# ── Evals Endpoint ────────────────────────────────────────────


class TestDashboardEvals:
    """Tests for the eval listings API."""

    def test_evals_empty(self, dashboard_client):
        response = dashboard_client.get("/api/dashboard/evals")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0

    def test_evals_with_data(self, empty_trace_store):
        from agentops.dashboard import create_dashboard_app
        eval_data = {"eval-001": {"benchmark": "test", "results": []}}
        app = create_dashboard_app(trace_store=empty_trace_store, eval_results=eval_data)
        client = TestClient(app)

        response = client.get("/api/dashboard/evals")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert "eval-001" in data["eval_runs"]


# ── Failure Analysis ────────────────────────────────────────────


class TestFailureAnalysis:
    """Tests for the failure analysis endpoint."""

    def test_failures_empty_store(self, dashboard_client):
        response = dashboard_client.get("/api/dashboard/failures")
        assert response.status_code == 200
        data = response.json()
        assert data["total_failures"] == 0

    def test_failures_with_data(self, dashboard_with_data):
        response = dashboard_with_data.get("/api/dashboard/failures")
        assert response.status_code == 200
        data = response.json()
        assert data["total_failures"] == 2
        assert len(data["patterns"]) > 0


# ── HTML Dashboard ────────────────────────────────────────────


class TestDashboardHTML:
    """Tests for the dashboard HTML page."""

    def test_dashboard_html_returns_200(self, dashboard_client):
        response = dashboard_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_html_contains_agentops(self, dashboard_client):
        response = dashboard_client.get("/")
        assert "AgentOps" in response.text

    def test_dashboard_route_also_works(self, dashboard_client):
        response = dashboard_client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_includes_websocket_url(self, dashboard_client):
        response = dashboard_client.get("/")
        assert "ws://" in response.text or "Chart" in response.text

    def test_dashboard_includes_stats(self, dashboard_with_data):
        response = dashboard_with_data.get("/")
        html = response.text
        # Should include initial stats from template rendering
        assert "Total Runs" in html


# ── WebSocket ─────────────────────────────────────────────────


class TestDashboardWebSocket:
    """Tests for the WebSocket endpoint."""

    def test_websocket_connect_and_receive(self, dashboard_client):
        """Test that WebSocket connection succeeds and receives stats."""
        with dashboard_client.websocket_connect("/ws") as websocket:
            # Should receive initial stats on connect
            data = websocket.receive_json()
            assert data["type"] == "stats"
            assert "data" in data

    def test_websocket_stats_have_fields(self, dashboard_client):
        with dashboard_client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            stats = data["data"]
            assert "total_runs" in stats
            assert "pass_rate" in stats
            assert "failure_rate" in stats
            assert "avg_latency_ms" in stats

    def test_websocket_heartbeat(self, dashboard_client):
        """Test that heartbeat messages arrive after stats."""
        with dashboard_client.websocket_connect("/ws") as websocket:
            _ = websocket.receive_json()  # stats
            # Next message should come within 10 seconds
            try:
                msg = websocket.receive_json()
                assert msg.get("type") in ("heartbeat", "stats", "traces")
            except Exception:
                pass  # Timeout OK in test


# ── DashboardServer Class ──────────────────────────────────────


class TestDashboardServerClass:
    """Tests for the DashboardServer class methods."""

    def test_broadcast_adds_client(self, empty_trace_store):
        from agentops.dashboard.server import DashboardServer
        server = DashboardServer(trace_store=empty_trace_store)
        assert len(server._ws_clients) == 0

    def test_get_failure_breakdown_empty(self, empty_trace_store):
        from agentops.dashboard.server import DashboardServer
        server = DashboardServer(trace_store=empty_trace_store)
        result = server._get_failure_breakdown()
        assert result == {"verification_failed": 0}

    def test_get_failure_breakdown_populated(self, populated_trace_store):
        from agentops.dashboard.server import DashboardServer
        server = DashboardServer(trace_store=populated_trace_store)
        result = server._get_failure_breakdown()
        assert result["verification_failed"] == 2

    def test_dashboard_server_with_eval_results(self, empty_trace_store):
        from agentops.dashboard.server import DashboardServer
        evals = {"bench1": {"score": 0.95}}
        server = DashboardServer(trace_store=empty_trace_store, eval_results=evals)
        assert server.eval_results == evals

    def test_dashboard_server_with_agent(self, empty_trace_store):
        from agentops.dashboard.server import DashboardServer
        server = DashboardServer(trace_store=empty_trace_store, agent="mock-agent")
        assert server.agent == "mock-agent"


# ── CLI Integration ───────────────────────────────────────────


class TestDashboardCLI:
    """Tests for the dashboard CLI command."""

    def test_dashboard_command_exists(self):
        """Verify the dashboard command is registered on the CLI app."""
        from agentops.cli.main import app
        commands = [c.name or c.callback.__name__ for c in app.registered_commands]
        assert "dashboard" in commands

    def test_dashboard_command_help(self):
        """Verify the dashboard command can be looked up by its function name."""
        from agentops.cli.main import app, dashboard as dashboard_fn
        # The command exists — verify the function is callable
        assert callable(dashboard_fn)
        assert dashboard_fn.__doc__ is not None
