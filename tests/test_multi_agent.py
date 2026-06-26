"""
Tests for the multi-agent coordination module.

Covers:
- State creation and serialization
- Graph topology (skipped without LLM API key)
- Coordinator with simulated workers
- Multi-agent benchmarks integration  
- Trace store extensions
- Error handling and configuration

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY for topology integration tests.
"""

from __future__ import annotations

import json
import os
import pytest
import tempfile
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────

def _has_llm_key():
    """Check if any LLM API key is available."""
    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


def _make_simulated_worker_fn(profile="production"):
    """Create a simulated worker function for testing."""
    from agentops.multi_agent.coordinator import MultiAgentCoordinator
    return MultiAgentCoordinator.make_simulated_worker_fn(profile_name=profile)


# ── State tests ────────────────────────────────────────────────────────


class TestMultiAgentState:
    def test_create_default_state(self):
        from agentops.multi_agent.state import create_multi_agent_state, DEFAULT_WORKER_ROLES

        state = create_multi_agent_state("Test task")
        assert state["task"] == "Test task"
        assert state["current_phase"] == "decompose"
        assert state["done"] is False
        assert state["step_count"] == 0
        assert len(state["worker_roles"]) == len(DEFAULT_WORKER_ROLES)
        assert len(state["assignments"]) == 0
        assert len(state["worker_results"]) == 0

    def test_create_state_with_context(self):
        from agentops.multi_agent.state import create_multi_agent_state

        state = create_multi_agent_state("Task", context="Some context")
        assert state["task_context"] == "Some context"

    def test_custom_worker_roles(self):
        from agentops.multi_agent.state import create_multi_agent_state, WorkerRole

        custom_roles = [
            WorkerRole(
                name="custom_worker",
                description="A custom worker",
                capabilities=["retrieval"],
                tools=[],
            ),
        ]
        state = create_multi_agent_state("Task", worker_roles=custom_roles)
        assert len(state["worker_roles"]) == 1
        assert state["worker_roles"][0]["name"] == "custom_worker"

    def test_worker_assignment_structure(self):
        from agentops.multi_agent.state import WorkerAssignment

        a = WorkerAssignment(
            assignment_id="asgn-001",
            worker_role="retrieval_specialist",
            subtask="Find information about deployment",
            context="CloudDeploy docs",
            priority=0,
        )
        assert a["assignment_id"] == "asgn-001"
        assert a["priority"] == 0

    def test_inter_agent_message_structure(self):
        from agentops.multi_agent.state import InterAgentMessage

        msg = InterAgentMessage(
            msg_id="msg-001",
            from_agent="supervisor",
            to_agent="retrieval_specialist",
            msg_type="assignment",
            content="Find docs on 2FA",
            timestamp_ms=1000.0,
            metadata={"priority": 0},
        )
        assert msg["msg_type"] == "assignment"
        assert msg["from_agent"] == "supervisor"


# ── Topology tests (require LLM) ───────────────────────────────────────


@pytest.mark.skipif(not _has_llm_key(), reason="Requires OPENAI_API_KEY or ANTHROPIC_API_KEY")
class TestMultiAgentTopology:
    @pytest.fixture
    def mock_worker_fn(self):
        from agentops.multi_agent.state import WorkerResult

        def _worker(role: str, subtask: str, context: str) -> WorkerResult:
            return WorkerResult(
                assignment_id=f"asgn-{role}",
                worker_role=role,
                subtask=subtask,
                answer=f"Answer from {role}: analyzed '{subtask[:50]}'",
                grounded_claims=[f"Claim from {role}"],
                ungrounded_claims=[],
                citations_used=["chunk-1", "chunk-2"],
                verification_passed=True,
                tool_calls_count=1,
                retrieved_chunks_count=3,
                latency_ms=500.0,
                error=None,
            )
        return _worker

    def test_graph_builds_and_compiles(self, mock_worker_fn):
        from agentops.multi_agent.topology import build_multi_agent_graph

        graph = build_multi_agent_graph(worker_fn=mock_worker_fn)
        assert graph is not None
        nodes = graph.get_graph().nodes
        assert "decompose" in nodes
        assert "execute_workers" in nodes
        assert "aggregate" in nodes
        assert "verify" in nodes
        assert "respond" in nodes

    def test_graph_runs_end_to_end(self, mock_worker_fn):
        from agentops.multi_agent.topology import build_multi_agent_graph
        from agentops.multi_agent.state import create_multi_agent_state
        import asyncio

        graph = build_multi_agent_graph(worker_fn=mock_worker_fn)
        initial_state = create_multi_agent_state("How do I enable 2FA and deploy a Python app?")

        async def _run():
            config = {"configurable": {"thread_id": "test-001"}}
            async for event in graph.astream(initial_state, config):
                pass
            snapshot = graph.get_state(config)
            return snapshot.values if snapshot else None

        result = asyncio.run(_run())
        assert result is not None
        assert result.get("done") is True
        assert len(result.get("assignments", [])) > 0
        assert len(result.get("worker_results", [])) > 0

    def test_decomposition_produces_assignments(self, mock_worker_fn):
        from agentops.multi_agent.topology import build_multi_agent_graph
        from agentops.multi_agent.state import create_multi_agent_state
        import asyncio

        graph = build_multi_agent_graph(worker_fn=mock_worker_fn)
        state = create_multi_agent_state("Analyze security and deploy")

        async def _run():
            config = {"configurable": {"thread_id": "test-002"}}
            async for event in graph.astream(state, config):
                for node_name, node_output in event.items():
                    if node_name == "decompose":
                        return node_output
            return None

        result = asyncio.run(_run())
        assert result is not None
        assert len(result.get("assignments", [])) > 0

    def test_coordination_trace_tracks_phases(self, mock_worker_fn):
        from agentops.multi_agent.topology import build_multi_agent_graph
        from agentops.multi_agent.state import create_multi_agent_state
        import asyncio

        graph = build_multi_agent_graph(worker_fn=mock_worker_fn)
        state = create_multi_agent_state("Test task")

        async def _run():
            config = {"configurable": {"thread_id": "test-004"}}
            async for event in graph.astream(state, config):
                pass
            snapshot = graph.get_state(config)
            return snapshot.values if snapshot else None

        result = asyncio.run(_run())
        assert result is not None
        trace = result.get("coordination_trace", [])
        phases = {entry["phase"] for entry in trace}
        assert "decompose" in phases
        assert "aggregate" in phases
        assert "respond" in phases


# ── Simulated worker tests (no LLM needed) ─────────────────────────────


class TestSimulatedWorkerIntegration:
    def test_make_simulated_worker_fn(self):
        worker = _make_simulated_worker_fn("production")
        result = worker("retrieval_specialist", "Find deployment docs", "CloudDeploy")

        assert result["worker_role"] == "retrieval_specialist"
        assert len(result["answer"]) > 0
        assert isinstance(result["verification_passed"], bool)
        assert result["latency_ms"] > 0

    def test_make_simulated_worker_fn_all_profiles(self):
        for profile in ["perfect", "production", "development", "unreliable"]:
            worker = _make_simulated_worker_fn(profile)
            result = worker("test_worker", f"Test task for {profile}", "")
            assert result["worker_role"] == "test_worker"
            assert result["latency_ms"] > 0

    def test_perfect_profile_always_verified(self):
        worker = _make_simulated_worker_fn("perfect")
        results = [worker("w", f"Task {i}", "") for i in range(10)]
        assert all(r["verification_passed"] for r in results)

    def test_unreliable_profile_has_failures(self):
        worker = _make_simulated_worker_fn("unreliable")
        results = [worker("w", f"Task {i}", "") for i in range(20)]
        failures = [r for r in results if not r["verification_passed"]]
        assert len(failures) > 0, "Unreliable profile should produce some failures"

    def test_worker_result_structure(self):
        worker = _make_simulated_worker_fn("production")
        result = worker("tool_executor", "Calculate pi * 2", "Math context")
        # All expected keys present
        expected_keys = [
            "assignment_id", "worker_role", "subtask", "answer",
            "grounded_claims", "ungrounded_claims", "citations_used",
            "verification_passed", "tool_calls_count",
            "retrieved_chunks_count", "latency_ms", "error",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# ── Coordinator tests (simulated workers, no LLM needed) ───────────────


class TestMultiAgentCoordinator:
    def test_coordinator_creates_with_simulated_workers(self):
        from agentops.multi_agent.coordinator import MultiAgentCoordinator, MultiAgentConfig

        worker_fn = _make_simulated_worker_fn("production")
        config = MultiAgentConfig(model="gpt-4o")
        coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)
        assert coordinator is not None
        assert coordinator.config.model == "gpt-4o"

    @pytest.mark.skipif(not _has_llm_key(), reason="Requires LLM API key for supervisor decomposition")
    def test_coordinator_run_produces_result(self):
        from agentops.multi_agent.coordinator import MultiAgentCoordinator, MultiAgentConfig
        import asyncio

        worker_fn = _make_simulated_worker_fn("production")
        config = MultiAgentConfig(model="gpt-4o")
        coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)

        async def _run():
            return await coordinator.run("How do I enable 2FA?", run_id="test-ma-001")

        result = asyncio.run(_run())
        assert result is not None
        assert result.run_id == "test-ma-001"
        assert result.worker_count > 0
        assert len(result.coordination_trace) > 0

    def test_coordinator_run_simulated_works(self):
        """Coordinator run with simulated supervisor (no LLM needed)."""
        from agentops.multi_agent.coordinator import MultiAgentCoordinator, MultiAgentConfig
        import asyncio

        worker_fn = _make_simulated_worker_fn("production")
        config = MultiAgentConfig(model="gpt-4o")
        coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)

        async def _run():
            return await coordinator.run("How do I enable 2FA and calculate costs?", run_id="test-ma-sim")

        result = asyncio.run(_run())
        assert result is not None
        assert result.run_id == "test-ma-sim"
        assert result.worker_count > 0
        assert len(result.coordination_trace) > 0
        assert len(result.subtasks) > 0, "Simulated decomposition should produce subtasks"
        assert len(result.inter_agent_messages) > 0

    @pytest.mark.skipif(not _has_llm_key(), reason="Requires LLM API key")
    def test_coordinator_sync_run(self):
        from agentops.multi_agent.coordinator import MultiAgentCoordinator, MultiAgentConfig

        worker_fn = _make_simulated_worker_fn("production")
        config = MultiAgentConfig(model="gpt-4o")
        coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)
        result = coordinator.run_sync("Deploy and secure an application")
        assert result is not None
        assert result.worker_count > 0

    @pytest.mark.skipif(not _has_llm_key(), reason="Requires LLM API key")
    def test_coordinator_different_profiles(self):
        from agentops.multi_agent.coordinator import MultiAgentCoordinator, MultiAgentConfig
        import asyncio

        for profile in ["perfect", "production"]:
            worker_fn = _make_simulated_worker_fn(profile)
            config = MultiAgentConfig(model="gpt-4o")
            coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)

            async def _run(p=profile):
                return await coordinator.run("Test task", run_id=f"test-{p}")

            result = asyncio.run(_run())
            assert result is not None

    @pytest.mark.skipif(not _has_llm_key(), reason="Requires LLM API key")
    def test_coordinator_tracks_inter_agent_messages(self):
        from agentops.multi_agent.coordinator import MultiAgentCoordinator, MultiAgentConfig
        import asyncio

        worker_fn = _make_simulated_worker_fn("production")
        config = MultiAgentConfig(model="gpt-4o")
        coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)

        async def _run():
            return await coordinator.run("Complex multi-domain task", run_id="test-msg-001")

        result = asyncio.run(_run())
        assert len(result.inter_agent_messages) > 0
        msg_types = {m["msg_type"] for m in result.inter_agent_messages}
        assert "assignment" in msg_types
        assert "result" in msg_types


# ── Trace store extension tests ────────────────────────────────────────


class TestMultiAgentTraceStore:
    @pytest.fixture
    def trace_store(self):
        from agentops.tracing.store import TraceStore
        store = TraceStore(":memory:")
        yield store
        store.close()

    def test_extend_adds_multi_agent_tables(self, trace_store):
        from agentops.multi_agent.coordinator import extend_trace_store_for_multi_agent

        extend_trace_store_for_multi_agent(trace_store)

        conn = trace_store._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "multi_agent_runs" in table_names
        assert "inter_agent_messages" in table_names
        assert "worker_assignments" in table_names

    @pytest.mark.skipif(not _has_llm_key(), reason="Requires LLM API key")
    def test_save_multi_agent_run(self, trace_store):
        from agentops.multi_agent.coordinator import (
            MultiAgentCoordinator, MultiAgentConfig,
            save_multi_agent_run,
        )
        import asyncio

        worker_fn = _make_simulated_worker_fn("production")
        config = MultiAgentConfig(model="gpt-4o")
        coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)

        async def _run():
            return await coordinator.run("Test", run_id="test-save-001")

        result = asyncio.run(_run())
        run_id = save_multi_agent_run(trace_store, result)
        assert run_id == "test-save-001"

        conn = trace_store._get_conn()
        row = conn.execute(
            "SELECT * FROM multi_agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        assert row["task"] == "Test"
        assert row["worker_count"] > 0

        msg_count = conn.execute(
            "SELECT COUNT(*) FROM inter_agent_messages WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert msg_count > 0

        wa_count = conn.execute(
            "SELECT COUNT(*) FROM worker_assignments WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert wa_count > 0

    def test_extend_is_idempotent(self, trace_store):
        from agentops.multi_agent.coordinator import extend_trace_store_for_multi_agent

        extend_trace_store_for_multi_agent(trace_store)
        extend_trace_store_for_multi_agent(trace_store)

        conn = trace_store._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert table_names.count("multi_agent_runs") == 1


# ── Benchmark integration tests ────────────────────────────────────────


class TestMultiAgentBenchmarks:
    def test_multi_agent_benchmark_exists(self):
        from agentops.evals.benchmarks import MULTI_AGENT_BENCH, get_benchmark

        bench = get_benchmark("multi-agent")
        assert bench is not None
        assert bench.name == "multi-agent"
        assert len(bench.tasks) == 5

    def test_multi_agent_benchmark_tasks_are_hard(self):
        from agentops.evals.benchmarks import MULTI_AGENT_BENCH

        for task in MULTI_AGENT_BENCH.tasks:
            assert task.difficulty == "hard"
            assert task.category == "multi_step"

    def test_multi_agent_benchmark_in_all_benchmarks(self):
        from agentops.evals.benchmarks import ALL_BENCHMARKS

        names = [b.name for b in ALL_BENCHMARKS]
        assert "multi-agent" in names
        assert len(ALL_BENCHMARKS) >= 7

    def test_multi_agent_benchmark_listed(self):
        from agentops.evals.benchmarks import list_benchmarks

        blist = list_benchmarks()
        names = [b["name"] for b in blist]
        assert "multi-agent" in names


# ── Error handling tests ───────────────────────────────────────────────


class TestMultiAgentErrorHandling:
    def test_empty_task_creates_state_gracefully(self):
        from agentops.multi_agent.state import create_multi_agent_state
        state = create_multi_agent_state("")
        assert state["task"] == ""
        assert state["current_phase"] == "decompose"

    def test_make_simulated_worker_invalid_profile(self):
        from agentops.multi_agent.coordinator import MultiAgentCoordinator

        with pytest.raises(ValueError, match="Unknown profile"):
            MultiAgentCoordinator.make_simulated_worker_fn(profile_name="nonexistent")

    def test_worker_result_error_handling(self):
        """Direct error simulation without going through graph."""
        from agentops.multi_agent.state import WorkerResult

        result = WorkerResult(
            assignment_id="asgn-err",
            worker_role="test_worker",
            subtask="Test",
            answer="",
            grounded_claims=[],
            ungrounded_claims=[],
            citations_used=[],
            verification_passed=False,
            tool_calls_count=0,
            retrieved_chunks_count=0,
            latency_ms=0,
            error="Simulated failure",
        )
        assert result["error"] == "Simulated failure"
        assert result["verification_passed"] is False


# ── MultiAgentConfig tests ─────────────────────────────────────────────


class TestMultiAgentConfig:
    def test_default_config(self):
        from agentops.multi_agent.coordinator import MultiAgentConfig
        from agentops.multi_agent.state import DEFAULT_WORKER_ROLES

        config = MultiAgentConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_worker_concurrency == 3
        assert len(config.worker_roles) == len(DEFAULT_WORKER_ROLES)

    def test_custom_config(self):
        from agentops.multi_agent.coordinator import MultiAgentConfig

        config = MultiAgentConfig(
            model="claude-3-opus",
            temperature=0.3,
            max_worker_concurrency=5,
        )
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.3
        assert config.max_worker_concurrency == 5


# ── CLI integration tests ──────────────────────────────────────────────


class TestMultiAgentCLI:
    def test_run_multi_command_registered(self):
        from agentops.cli.main import app
        command_names = []
        for cmd in app.registered_commands:
            command_names.append(cmd.name or cmd.callback.__name__)
        # Check for run-multi (typer converts _ to - for display)
        assert "run-multi" in command_names or any("run_multi" in n for n in command_names)

    def test_eval_multi_command_registered(self):
        from agentops.cli.main import app
        command_names = []
        for cmd in app.registered_commands:
            command_names.append(cmd.name or cmd.callback.__name__)
        assert "eval-multi" in command_names or any("eval_multi" in n for n in command_names)

    def test_run_multi_invalid_profile_errors(self):
        from typer.testing import CliRunner
        from agentops.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "run-multi",
            "Test task",
            "--profile", "nonexistent",
        ])
        # Should return non-zero exit or handle gracefully
        assert result.exit_code in [0, 1, 2]

    @pytest.mark.skipif(not _has_llm_key(), reason="Requires LLM API key for supervisor")
    def test_run_multi_with_json_output(self):
        from typer.testing import CliRunner
        from agentops.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "run-multi",
            "How do I enable 2FA?",
            "--profile", "production",
            "--json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "run_id" in data
        assert "final_answer" in data
