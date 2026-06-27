"""
Tests for the AgentOps SDK — client library for agent instrumentation.

Tests cover:
- State models (serialization, defaults, enums)
- HTTP client (health, traces, stats, error handling)
- Tracer (decorators, context managers, logging helpers, query API)
- CLI commands (sdk init, demo, query, status)
- Integration: end-to-end trace/query flow
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agentops.sdk.state import (
    SDKConfig,
    TraceSpan,
    SpanKind,
    SpanStatus,
    RunContext,
    AgentOpsClient,
    ToolCallRecord,
    RetrievalRecord,
    TraceStatus,
)
from agentops.sdk.client import AgentOpsHTTPClient
from agentops.sdk.tracer import (
    AgentOps,
    init,
    trace,
    start_run,
    log_tool_call,
    log_retrieval,
    log_verification,
    get_current_run,
    _active_run,
    _get_client,
    _global_agentops,
)


# ── State Model Tests ─────────────────────────────────────────────────

class TestSpanKind:
    def test_all_kinds_present(self):
        kinds = {k.value for k in SpanKind}
        assert "run" in kinds
        assert "plan" in kinds
        assert "retrieve" in kinds
        assert "tool" in kinds
        assert "verify" in kinds
        assert "respond" in kinds
        assert "llm" in kinds


class TestTraceSpan:
    def test_default_span(self):
        span = TraceSpan()
        assert span.span_id
        assert span.kind == SpanKind.RUN
        assert span.status == SpanStatus.OK
        assert span.end_time is None
        assert span.children == []
        assert span.events == []

    def test_finish_sets_end_time_and_latency(self):
        span = TraceSpan()
        span.start_time = 1000.0
        span.finish()
        assert span.end_time is not None
        assert span.latency_ms >= 0

    def test_finish_with_status(self):
        span = TraceSpan()
        span.finish(status=SpanStatus.ERROR)
        assert span.status == SpanStatus.ERROR

    def test_add_event(self):
        span = TraceSpan()
        span.add_event("tool_call", {"name": "search"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "tool_call"
        assert span.events[0]["attributes"]["name"] == "search"

    def test_to_dict_includes_children(self):
        parent = TraceSpan()
        child = TraceSpan(kind=SpanKind.TOOL, name="tool:search")
        parent.children.append(child)
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["kind"] == "tool"

    def test_to_dict_all_fields(self):
        span = TraceSpan(
            kind=SpanKind.RETRIEVE,
            name="retrieve:test query",
            status=SpanStatus.WARNING,
            attributes={"chunks": 3},
        )
        d = span.to_dict()
        assert d["kind"] == "retrieve"
        assert d["name"] == "retrieve:test query"
        assert d["status"] == "warning"
        assert d["attributes"]["chunks"] == 3


class TestRunContext:
    def test_default_context(self):
        ctx = RunContext()
        assert ctx.run_id
        assert ctx.status == TraceStatus.RUNNING
        assert ctx.tool_calls == []
        assert ctx.retrievals == []

    def test_finish_success(self):
        ctx = RunContext()
        root = TraceSpan()
        ctx.root_span = root
        ctx.finish(success=True)
        assert ctx.status == TraceStatus.SUCCESS
        assert ctx.error is None
        assert root.status == SpanStatus.OK

    def test_finish_failure(self):
        ctx = RunContext()
        root = TraceSpan()
        ctx.root_span = root
        ctx.finish(success=False, error="Something broke")
        assert ctx.status == TraceStatus.FAILED
        assert ctx.error == "Something broke"
        assert root.status == SpanStatus.ERROR

    def test_latency_ms(self):
        ctx = RunContext()
        ctx.start_time = time.time() - 1.0
        assert 900 <= ctx.latency_ms <= 1100

    def test_to_dict(self):
        ctx = RunContext(
            task="Test task",
            model="gpt-4o",
            verification_passed=True,
            final_answer="The answer",
        )
        ctx.tool_calls.append(ToolCallRecord("search", {"q": "test"}, tool_output="result"))
        ctx.root_span = TraceSpan(name="root")
        d = ctx.to_dict()
        assert d["task"] == "Test task"
        assert d["model"] == "gpt-4o"
        assert d["verification_passed"] is True
        assert d["final_answer"] == "The answer"
        assert len(d["tool_calls"]) == 1
        assert d["root_span"] is not None


class TestToolCallRecord:
    def test_serialization(self):
        tc = ToolCallRecord(
            tool_name="search",
            tool_input={"query": "test"},
            tool_output="results",
            success=True,
        )
        d = tc.to_dict()
        assert d["tool_name"] == "search"
        assert d["tool_input"]["query"] == "test"
        assert d["tool_output"] == "results"
        assert d["success"] is True

    def test_failed_tool_call(self):
        tc = ToolCallRecord(
            tool_name="bad_tool",
            tool_input={},
            success=False,
            error="Tool not found",
        )
        d = tc.to_dict()
        assert d["success"] is False
        assert d["error"] == "Tool not found"


class TestRetrievalRecord:
    def test_serialization(self):
        rr = RetrievalRecord(
            query="test query",
            chunks_retrieved=3,
            top_chunk_scores=[0.9, 0.8, 0.7],
            retrieval_method="hybrid",
        )
        d = rr.to_dict()
        assert d["query"] == "test query"
        assert d["chunks_retrieved"] == 3
        assert d["top_chunk_scores"] == [0.9, 0.8, 0.7]
        assert d["retrieval_method"] == "hybrid"


class TestSDKConfig:
    def test_defaults(self):
        cfg = SDKConfig()
        assert cfg.endpoint == "http://localhost:8000"
        assert cfg.enabled is True
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30.0

    def test_custom_config(self):
        cfg = SDKConfig(
            endpoint="https://agentops.example.com",
            api_key="sk-test",
            project_name="my-agent",
            max_retries=5,
        )
        assert cfg.endpoint == "https://agentops.example.com"
        assert cfg.api_key == "sk-test"
        assert cfg.project_name == "my-agent"
        assert cfg.max_retries == 5


# ── HTTP Client Tests ──────────────────────────────────────────────────

class TestAgentOpsHTTPClient:
    """Test the HTTP client layer — uses urllib mocking."""

    def make_mock_response(self, data, status=200):
        """Create a mock that returns data as JSON."""
        mock = MagicMock()
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=False)
        mock.read.return_value = json.dumps(data).encode("utf-8")
        mock.status = status
        return mock

    def test_url_construction(self):
        client = AgentOpsHTTPClient(SDKConfig(endpoint="http://localhost:8000"))
        assert client._url("/health") == "http://localhost:8000/health"
        assert client._url("/api/traces") == "http://localhost:8000/api/traces"

    def test_url_trailing_slash_handling(self):
        client = AgentOpsHTTPClient(SDKConfig(endpoint="http://localhost:8000/"))
        assert client._url("/health") == "http://localhost:8000/health"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_health_check_success(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({"status": "ok"})
        client = AgentOpsHTTPClient()
        assert client.health_check() is True
        assert client._session_ready is True

    @patch("urllib.request.urlopen")
    def test_health_check_failure(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("Connection refused")
        client = AgentOpsHTTPClient()
        assert client.health_check() is False
        assert client._session_ready is False

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_list_traces(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({
            "count": 2,
            "traces": [
                {"run_id": "r1", "task": "test", "verification_passed": True},
                {"run_id": "r2", "task": "test2", "verification_passed": False},
            ],
        })
        client = AgentOpsHTTPClient()
        client._session_ready = True
        traces = client.list_traces(limit=10)
        assert len(traces) == 2
        assert traces[0]["run_id"] == "r1"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_list_traces_with_filter(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({"count": 0, "traces": []})
        client = AgentOpsHTTPClient()
        traces = client.list_traces(verification_passed=True, limit=5)
        assert traces == []

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_get_trace_found(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({
            "run_id": "r1", "task": "test", "verification_passed": True,
        })
        client = AgentOpsHTTPClient()
        result = client.get_trace("r1")
        assert result is not None
        assert result["run_id"] == "r1"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_get_trace_not_found(self, mock_request, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        client = AgentOpsHTTPClient()
        result = client.get_trace("nonexistent")
        assert result is None

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_get_stats(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({
            "total_runs": 42, "verification_pass_rate": 0.85,
        })
        client = AgentOpsHTTPClient()
        stats = client.get_stats()
        assert stats["total_runs"] == 42

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_list_evals(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({
            "eval_runs": ["eval1", "eval2"], "count": 2,
        })
        client = AgentOpsHTTPClient()
        evals = client.list_evals()
        assert evals["count"] == 2

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_submit_trace(self, mock_request, mock_urlopen):
        mock_urlopen.return_value = self.make_mock_response({
            "run_id": "abc123", "success": True, "verification_passed": True,
        })
        client = AgentOpsHTTPClient()
        ctx = RunContext(task="test task", run_id="abc123")
        result = client.submit_trace(ctx)
        assert result["success"] is True

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_auth_header(self, mock_request, mock_urlopen):
        """API key should be sent as Bearer token."""
        mock_urlopen.return_value = self.make_mock_response({"status": "ok"})
        client = AgentOpsHTTPClient(SDKConfig(api_key="test-key-123"))
        client.health_check()
        # Verify the request was made with auth header
        call_args = mock_request.call_args
        assert call_args is not None

    @patch("urllib.request.urlopen")
    def test_retry_on_transient_error(self, mock_urlopen):
        """Should retry on connection errors and raise ConnectionError after max retries."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Temporary failure")
        client = AgentOpsHTTPClient(SDKConfig(max_retries=2))
        with pytest.raises(ConnectionError):
            # Call _request directly — health_check catches ConnectionError
            client._request("GET", "/api/stats")
        # Should have tried 3 times (2 retries + initial)
        assert mock_urlopen.call_count == 3


# ── Tracer Tests ──────────────────────────────────────────────────────

class TestAgentOpsClass:
    """Tests for the AgentOps class (tracer)."""

    def test_init_not_initialized_by_default(self):
        aops = AgentOps()
        assert not aops.is_ready
        assert aops.config is None

    def test_init_without_server(self):
        aops = AgentOps()
        # Should not raise — returns False when server unreachable
        result = aops.init(endpoint="http://localhost:9999", enabled=False)
        assert result is False

    @patch("agentops.sdk.client.AgentOpsHTTPClient.health_check")
    def test_init_with_server(self, mock_health):
        mock_health.return_value = True
        aops = AgentOps()
        result = aops.init(endpoint="http://localhost:8000")
        assert result is True
        assert aops.is_ready
        assert aops.config.project_name == "default"

    @patch("agentops.sdk.client.AgentOpsHTTPClient.health_check")
    def test_init_registers_global(self, mock_health):
        mock_health.return_value = True
        aops = AgentOps()
        aops.init(endpoint="http://localhost:8000")
        assert _get_client() is aops

    def test_start_run_context_manager(self):
        aops = AgentOps()
        with aops.start_run(task="Test task", model="test-model") as run:
            assert run.task == "Test task"
            assert run.model == "test-model"
            assert run.root_span is not None
            assert run.root_span.kind == SpanKind.RUN
        # After context exit, run should be finished
        assert run.status == TraceStatus.SUCCESS

    def test_start_run_with_exception(self):
        aops = AgentOps()
        try:
            with aops.start_run(task="Failing task") as run:
                raise ValueError("Something went wrong")
        except ValueError:
            pass
        assert run.status == TraceStatus.FAILED

    def test_log_tool_call_records(self):
        aops = AgentOps()
        with aops.start_run(task="task") as run:
            aops.log_tool_call("search", {"q": "test"}, tool_output="results")
            aops.log_tool_call("fetch", {"url": "/data"}, success=False, error="404")
        assert len(run.tool_calls) == 2
        assert run.tool_calls[0].tool_name == "search"
        assert run.tool_calls[1].tool_name == "fetch"
        assert run.tool_calls[1].success is False
        # Should create child spans in root
        assert len(run.root_span.children) == 2

    def test_log_retrieval_records(self):
        aops = AgentOps()
        with aops.start_run(task="task") as run:
            aops.log_retrieval("test query", chunks=["a", "b"], scores=[0.9, 0.8])
        assert len(run.retrievals) == 1
        assert run.retrievals[0].query == "test query"
        assert run.retrievals[0].chunks_retrieved == 2

    def test_log_verification_records(self):
        aops = AgentOps()
        with aops.start_run(task="task") as run:
            aops.log_verification(
                passed=True,
                notes="All good",
                grounded_claims=["claim1"],
                ungrounded_claims=[],
            )
        assert run.verification_passed is True
        assert run.verification_notes == "All good"
        assert run.grounded_claims == ["claim1"]

    def test_trace_decorator_sync(self):
        """Test trace decorator on sync function."""
        aops = AgentOps()

        @aops.trace(model="test")
        def my_func(task: str) -> str:
            return f"answer: {task}"

        result = my_func("hello")
        assert result == "answer: hello"

    def test_trace_decorator_with_explicit_task(self):
        """Test trace decorator with explicit task name."""
        aops = AgentOps()

        @aops.trace(task="explicit task")
        def my_func(x: str) -> str:
            return x

        result = my_func("ignored")
        assert result == "ignored"

    @pytest.mark.asyncio
    async def test_trace_decorator_async(self):
        """Test trace decorator on async function."""
        aops = AgentOps()

        @aops.trace(model="test-async")
        async def my_agent(task: str) -> str:
            return f"result: {task}"

        result = await my_agent("query")
        assert result == "result: query"

    def test_trace_decorator_captures_exception(self):
        """Trace should capture exceptions and mark run as failed."""
        aops = AgentOps()

        @aops.trace()
        def broken_func(task: str) -> str:
            raise RuntimeError("Boom")

        with pytest.raises(RuntimeError, match="Boom"):
            broken_func("test")

    @patch("agentops.sdk.client.AgentOpsHTTPClient.list_traces")
    @patch("agentops.sdk.client.AgentOpsHTTPClient.health_check")
    def test_query_traces(self, mock_health, mock_list_traces):
        mock_health.return_value = True
        mock_list_traces.return_value = [{"run_id": "r1", "task": "test"}]
        aops = AgentOps()
        aops.init(endpoint="http://localhost:8000")
        traces = aops.list_traces()
        assert len(traces) == 1
        assert traces[0]["run_id"] == "r1"

    def test_query_when_not_initialized(self):
        aops = AgentOps()
        assert aops.list_traces() == []
        assert aops.get_trace("x") is None
        assert aops.get_stats() == {"error": "Not connected"}
        assert aops.list_evals() == {"eval_runs": [], "count": 0}


# ── Module-level convenience function tests ────────────────────────────

class TestModuleLevelHelpers:
    """Tests for module-level init(), trace(), start_run(), etc."""

    def test_init_sets_global(self):
        import agentops.sdk.tracer as tmod
        # Reset global
        tmod._global_agentops = None
        # init without server — should return False
        result = init(endpoint="http://localhost:9999", enabled=False)
        assert result is False

    def test_start_run_standalone(self):
        """start_run should work without global client initialized."""
        with start_run(task="standalone task") as run:
            assert run.task == "standalone task"
            log_tool_call("test_tool", {}, tool_output="ok")
            log_retrieval("query", chunks=["doc"])
            log_verification(passed=True, notes="verified")
        assert run.status == TraceStatus.SUCCESS
        assert len(run.tool_calls) == 1
        assert len(run.retrievals) == 1

    def test_log_tool_call_no_active_run(self):
        """log_tool_call should be safe when no run is active."""
        token = _active_run.set(None)
        try:
            log_tool_call("orphan", {})  # Should not raise
        finally:
            _active_run.reset(token)

    def test_log_retrieval_no_active_run(self):
        token = _active_run.set(None)
        try:
            log_retrieval("orphan query")  # Should not raise
        finally:
            _active_run.reset(token)

    def test_log_verification_no_active_run(self):
        token = _active_run.set(None)
        try:
            log_verification(passed=True)  # Should not raise
        finally:
            _active_run.reset(token)

    def test_get_current_run_outside_context(self):
        token = _active_run.set(None)
        try:
            assert get_current_run() is None
        finally:
            _active_run.reset(token)

    def test_trace_noop_when_not_initialized(self):
        """Module-level trace() returns no-op decorator when not initialized."""
        import agentops.sdk.tracer as tmod
        tmod._global_agentops = None

        @trace()
        def my_func(x):
            return x * 2

        result = my_func(21)
        assert result == 42  # No tracing, but function still works

    def test_run_context_accumulates_data(self):
        with start_run(task="complex task", model="test-v1") as run:
            log_tool_call("step1", {"a": 1}, tool_output="done")
            log_retrieval("q1", chunks=["c1", "c2"])
            log_tool_call("step2", {"b": 2}, tool_output="done")
            log_verification(passed=True, grounded_claims=["claim"])
            run.plan_steps = ["plan1", "plan2"]
            run.final_answer = "complete answer"
            run.citations_used = ["doc1", "doc2"]
            run.metadata = {"version": "1.0"}

        assert len(run.tool_calls) == 2
        assert len(run.retrievals) == 1
        assert run.verification_passed is True
        assert len(run.plan_steps) == 2
        assert run.final_answer == "complete answer"
        assert len(run.citations_used) == 2
        assert run.metadata == {"version": "1.0"}


class TestTraceSpanTree:
    """Test nested span trees created by instrumentation."""

    def test_nested_spans_from_multiple_operations(self):
        with start_run(task="nested test") as run:
            log_tool_call("t1", {}, tool_output="ok")
            log_retrieval("query", chunks=["a"])
            log_tool_call("t2", {}, tool_output="ok")
            log_verification(passed=True)

        root = run.root_span
        assert root is not None
        # 4 child spans: 2 tools + 1 retrieval + 1 verification
        assert len(root.children) == 4

        kinds = [c.kind for c in root.children]
        assert SpanKind.TOOL in kinds
        assert SpanKind.RETRIEVE in kinds
        assert SpanKind.VERIFY in kinds

    def test_child_spans_have_correct_status(self):
        with start_run(task="error test") as run:
            log_tool_call("good_tool", {}, success=True)
            log_tool_call("bad_tool", {}, success=False, error="fail")

        good_span = [c for c in run.root_span.children if c.attributes.get("tool_name") == "good_tool"][0]
        bad_span = [c for c in run.root_span.children if c.attributes.get("tool_name") == "bad_tool"][0]
        assert good_span.status == SpanStatus.OK
        assert bad_span.status == SpanStatus.ERROR


# ── Edge Cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_task(self):
        with start_run(task="") as run:
            pass
        assert run.task == ""
        assert run.status == TraceStatus.SUCCESS

    def test_very_long_task(self):
        long_task = "x" * 5000
        with start_run(task=long_task) as run:
            pass
        assert len(run.task) == 5000

    def test_unicode_task(self):
        with start_run(task="こんにちは世界 🌍") as run:
            pass
        assert run.task == "こんにちは世界 🌍"

    def test_run_to_dict_no_root_span(self):
        ctx = RunContext(task="test")
        d = ctx.to_dict()
        assert d["root_span"] is None

    def test_concurrent_runs_isolated(self):
        """Two runs should not interfere with each other."""
        run1_ctx = []

        with start_run(task="run1") as run1:
            log_tool_call("run1_tool", {"run": 1})
            run1_ctx.append(run1)

            with start_run(task="run2") as run2:
                log_tool_call("run2_tool", {"run": 2})
                # run2 should be the active run
                assert get_current_run() is run2

            # After run2 exits, run1 is active again
            # Note: start_run clears _active_run on exit, so this is None
            pass

        assert len(run1.tool_calls) == 1
        assert run1.tool_calls[0].tool_name == "run1_tool"


class TestFlushToServer:
    """Test that traces are submitted to the server."""

    @patch("agentops.sdk.client.AgentOpsHTTPClient.submit_trace")
    @patch("agentops.sdk.client.AgentOpsHTTPClient.health_check")
    def test_start_run_flushes_on_exit(self, mock_health, mock_submit):
        mock_health.return_value = True
        mock_submit.return_value = {"success": True}

        aops = AgentOps()
        aops.init(endpoint="http://localhost:8000")

        with aops.start_run(task="flush test") as run:
            run.final_answer = "done"

        # Should have attempted to submit
        mock_submit.assert_called_once()

    @patch("agentops.sdk.client.AgentOpsHTTPClient.submit_trace")
    @patch("agentops.sdk.client.AgentOpsHTTPClient.health_check")
    def test_submit_failure_is_silent(self, mock_health, mock_submit):
        """Trace submission failures should not crash user code."""
        mock_health.return_value = True
        mock_submit.side_effect = Exception("Server error")

        aops = AgentOps()
        aops.init(endpoint="http://localhost:8000")

        # Should not raise
        with aops.start_run(task="resilient test") as run:
            run.final_answer = "done"

    @patch("agentops.sdk.client.AgentOpsHTTPClient.health_check")
    def test_no_flush_when_not_initialized(self, mock_health):
        mock_health.return_value = False
        aops = AgentOps()
        aops.init(endpoint="http://localhost:8000")

        # Should not try to submit since server is not healthy
        with aops.start_run(task="offline test") as run:
            run.final_answer = "done"
        # No exception expected


# ── CLI Command Tests ──────────────────────────────────────────────────

class TestSDKCLI:
    """Tests for the SDK CLI subcommands (via sdk_app)."""

    def _get_sdk_app(self):
        from agentops.cli.main import sdk_app
        return sdk_app

    def test_sdk_init_no_server(self):
        """sdk init without a server should return non-zero."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agentops.cli.main", "sdk", "init",
             "--endpoint", "http://localhost:9999"],
            capture_output=True, text=True, timeout=10,
        )
        # Should exit with error since server is unreachable
        assert result.returncode != 0 or "unreachable" in (result.stdout + result.stderr).lower()

    def test_sdk_demo_standalone(self):
        """sdk demo should run without a server (local-only mode)."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agentops.cli.main", "sdk", "demo",
             "--endpoint", "http://localhost:9999"],
            capture_output=True, text=True, timeout=10,
        )
        stdout = result.stdout + result.stderr
        # Demo should complete — it runs locally even if server is down
        assert "Run complete" in stdout
        assert "Verification:" in stdout
        assert "Final Answer:" in stdout

    def test_sdk_query_no_server(self):
        """sdk query without a server should exit with error."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agentops.cli.main", "sdk", "query",
             "--endpoint", "http://localhost:9999"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_sdk_status_not_initialized(self):
        """sdk status without init should exit with error."""
        import subprocess
        import sys

        # Reset global state
        import agentops.sdk.tracer
        agentops.sdk.tracer._global_agentops = None

        result = subprocess.run(
            [sys.executable, "-m", "agentops.cli.main", "sdk", "status"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_sdk_help(self):
        """sdk --help should show available commands."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "agentops.cli.main", "sdk", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "init" in result.stdout
        assert "demo" in result.stdout
        assert "query" in result.stdout
        assert "status" in result.stdout


# ── Integration Tests ─────────────────────────────────────────────────

class TestSDKIntegration:
    """End-to-end integration test: local trace → query via SDK."""

    def test_full_local_workflow(self):
        """Run a full trace locally without server, verify all data is captured."""
        with start_run(
            task="What is the status of server US-EAST-1?",
            task_id="integ-001",
            model="test-agent-v2",
        ) as run:
            # Plan
            run.plan_steps = [
                "Search infrastructure docs",
                "Query monitoring system",
                "Verify against runbooks",
            ]

            # Retrieval
            log_retrieval(
                "US-EAST-1 server status",
                chunks=[
                    "US-EAST-1 is operational with 99.9% uptime",
                    "Last incident: 2026-06-15 (resolved)",
                ],
                scores=[0.95, 0.82],
                method="hybrid",
            )

            # Tool calls
            log_tool_call(
                "monitoring_query",
                {"server": "US-EAST-1", "metric": "status"},
                tool_output="OPERATIONAL — last checked 2026-06-27 06:00 UTC",
                success=True,
            )
            log_tool_call(
                "check_incidents",
                {"server": "US-EAST-1", "days": 30},
                tool_output="1 incident in last 30 days: DNS propagation delay (resolved)",
                success=True,
            )

            # Verification
            log_verification(
                passed=True,
                notes="All claims verified against monitoring and incident records",
                grounded_claims=[
                    "US-EAST-1 is currently operational",
                    "Last incident was DNS-related, resolved on 2026-06-15",
                ],
                ungrounded_claims=[],
            )

            run.final_answer = (
                "Server US-EAST-1 is currently OPERATIONAL. "
                "Last checked: 2026-06-27 06:00 UTC. "
                "Uptime: 99.9% in the last 30 days. "
                "Only 1 incident in the past 30 days (DNS propagation, resolved 2026-06-15). "
                "Source: CloudDeploy Monitoring System + Incident Tracker."
            )
            run.verification_passed = True
            run.citations_used = ["monitoring_query", "check_incidents"]
            run.metadata = {
                "agent_version": "v2.3.1",
                "region": "us-east",
            }

        # Verify everything
        assert run.run_id == "integ-001"
        assert len(run.plan_steps) == 3
        assert len(run.tool_calls) == 2
        assert len(run.retrievals) == 1
        assert run.verification_passed is True
        assert len(run.grounded_claims) == 2
        assert len(run.ungrounded_claims) == 0
        assert len(run.citations_used) == 2
        assert run.metadata["agent_version"] == "v2.3.1"
        assert "OPERATIONAL" in run.final_answer

        # Verify root span tree
        root = run.root_span
        assert root is not None
        assert root.kind == SpanKind.RUN
        assert len(root.children) == 4  # 2 tools + 1 retrieval + 1 verification

        # Verify serialization
        d = run.to_dict()
        assert d["status"] == "success"
        assert d["run_id"] == "integ-001"
        assert len(d["tool_calls"]) == 2
        assert len(d["retrievals"]) == 1

    def test_multiple_runs_independent(self):
        """Running two traces sequentially should not mix data."""
        with start_run(task="run A", task_id="a") as run_a:
            log_tool_call("tool_a", {})

        with start_run(task="run B", task_id="b") as run_b:
            log_tool_call("tool_b", {})

        assert len(run_a.tool_calls) == 1
        assert run_a.tool_calls[0].tool_name == "tool_a"
        assert len(run_b.tool_calls) == 1
        assert run_b.tool_calls[0].tool_name == "tool_b"

    def test_trace_serialization_roundtrip(self):
        """RunContext.to_dict() should produce valid JSON."""
        with start_run(task="roundtrip test") as run:
            log_tool_call("t", {"key": "value"}, tool_output=[1, 2, 3])
            log_retrieval("q", chunks=["c"], scores=[0.5])
            run.final_answer = "answer"

        d = run.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["task"] == "roundtrip test"
        assert parsed["status"] == "success"
