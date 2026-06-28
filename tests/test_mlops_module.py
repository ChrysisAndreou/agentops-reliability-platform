"""
Tests for the agentops.mlops module — W&B experiment tracking, artifact management,
hyperparameter sweeps, and local file-system fallback.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentops.mlops import SweepConfig, WandBSweep, WandBTracker
from agentops.mlops.sweeps import (
    SweepParameter,
    agent_profile_sweep,
    retrieval_sweep,
)
from agentops.mlops.tracker import RunMetadata, _generate_run_id


# ── WandBTracker: Initialization ─────────────────────────────────────

class TestWandBTrackerInit:
    def test_default_init(self):
        tracker = WandBTracker()
        assert tracker._project == "agentops-reliability-platform"
        assert tracker._mode == "local"  # wandb not installed → local fallback
        assert tracker._tags == []
        assert tracker._notes == ""

    def test_init_with_project(self):
        tracker = WandBTracker(project="my-agent-evals")
        assert tracker._project == "my-agent-evals"

    def test_init_with_tags_and_notes(self):
        tracker = WandBTracker(
            project="test-evals",
            tags=["v0.19", "benchmark"],
            notes="Testing W&B integration",
        )
        assert tracker._tags == ["v0.19", "benchmark"]
        assert tracker._notes == "Testing W&B integration"

    def test_init_with_config(self):
        tracker = WandBTracker(
            config={"model": "gpt-4o", "temperature": 0.0}
        )
        assert tracker._config == {"model": "gpt-4o", "temperature": 0.0}

    def test_init_with_entity(self):
        tracker = WandBTracker(entity="chrysis-team")
        assert tracker._entity == "chrysis-team"

    def test_init_local_dir_custom(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WandBTracker(local_dir=tmpdir)
            assert tracker._local_dir == Path(tmpdir)

    def test_is_available_false_without_wandb(self):
        tracker = WandBTracker()
        assert tracker.is_available is False

    def test_init_with_env_vars(self):
        with patch.dict(os.environ, {
            "WANDB_PROJECT": "env-project",
            "WANDB_ENTITY": "env-entity",
        }):
            tracker = WandBTracker()
            assert tracker._project == "env-project"
            assert tracker._entity == "env-entity"


# ── WandBTracker: Run Lifecycle (local fallback) ─────────────────────

class TestWandBTrackerLocalRun:
    def test_init_run_creates_metadata(self):
        tracker = WandBTracker(project="test-evals")
        meta = tracker.init_run(run_name="test-run-001")
        assert meta is not None
        assert meta.run_name == "test-run-001"
        assert meta.project == "test-evals"
        assert meta.status == "running"
        assert meta.run_id != ""
        assert meta.started_at != ""

    def test_init_run_auto_generates_name(self):
        tracker = WandBTracker()
        meta = tracker.init_run()
        assert meta is not None
        assert meta.run_name.startswith("agentops-")

    def test_init_run_with_explicit_id(self):
        tracker = WandBTracker()
        meta = tracker.init_run(run_name="explicit-id-run", run_id="my-custom-id")
        assert meta.run_id == "my-custom-id"

    def test_log_metrics_appends_to_history(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="metrics-test")
        tracker.log_metrics({"composite": 0.85, "groundedness": 0.92})
        tracker.log_metrics({"composite": 0.87}, step=10)

        meta = tracker.metadata
        assert meta is not None
        assert len(meta.metrics_history) == 2
        assert meta.metrics_history[0]["composite"] == 0.85
        assert meta.metrics_history[0]["_step"] == 0
        assert meta.metrics_history[1]["_step"] == 10

    def test_log_metrics_step_auto_increment(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="step-test")
        tracker.log_metrics({"a": 1.0})
        tracker.log_metrics({"b": 2.0})
        tracker.log_metrics({"c": 3.0})

        meta = tracker.metadata
        steps = [m["_step"] for m in meta.metrics_history]
        assert steps == [0, 1, 2]

    def test_log_config_updates_metadata(self):
        tracker = WandBTracker(config={"initial": True})
        tracker.init_run(run_name="config-test")
        tracker.log_config({"model": "gpt-4o", "temperature": 0.0})

        meta = tracker.metadata
        assert meta is not None
        assert meta.config.get("model") == "gpt-4o"
        assert meta.config.get("temperature") == 0.0

    def test_finish_sets_status(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="finish-test")
        meta = tracker.finish()
        assert meta is not None
        assert meta.status == "finished"

    def test_finish_failure_status(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="fail-test")
        meta = tracker.finish(exit_code=1)
        assert meta is not None
        assert meta.status == "failed"

    def test_finish_saves_local_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = WandBTracker(project="test-evals", local_dir=tmpdir)
            tracker.init_run(run_name="persist-test")
            tracker.log_metrics({"score": 0.95})
            tracker.finish()

            run_file = Path(tmpdir) / "persist-test.json"
            assert run_file.exists()
            data = json.loads(run_file.read_text())
            assert data["run_name"] == "persist-test"
            assert data["status"] == "finished"
            assert len(data["metrics_history"]) == 1

    def test_summary_returns_last_metric(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="summary-test")
        tracker.log_metrics({"a": 1.0}, step=0)
        tracker.log_metrics({"b": 2.0}, step=1)

        summary = tracker.summary()
        assert summary["b"] == 2.0


# ── WandBTracker: Benchmark Report Logging ───────────────────────────

class TestWandBTrackerLogBenchmark:
    def test_log_benchmark_report_from_dict(self):
        tracker = WandBTracker(project="test-bench")
        tracker.init_run(run_name="bench-test")

        report_dict = {
            "benchmark_name": "support-tickets",
            "model": "gpt-4o",
            "total_tasks": 5,
            "summary": {
                "composite_mean": 0.85,
                "groundedness_mean": 0.92,
                "citation_precision_mean": 0.88,
                "verification_pass_rate": 0.80,
                "avg_latency_ms": 2500.0,
                "tasks_evaluated": 5,
            },
            "results": [
                {
                    "task_id": "task-1",
                    "composite": 0.90,
                    "groundedness": 0.95,
                    "latency_ms": 2000,
                    "verification_passed": True,
                },
                {
                    "task_id": "task-2",
                    "composite": 0.80,
                    "groundedness": 0.89,
                    "latency_ms": 3000,
                    "verification_passed": True,
                },
            ],
            "failure_patterns": [
                {
                    "name": "hallucination",
                    "count": 2,
                    "severity": "high",
                    "description": "Agent produced ungrounded claims",
                },
            ],
        }

        tracker.log_benchmark_report(report_dict)
        meta = tracker.metadata
        assert meta is not None
        # Should have logged summary metrics
        assert len(meta.metrics_history) >= 1
        logged_keys = list(meta.metrics_history[0].keys())
        assert any("composite_mean" in k for k in logged_keys)

    def test_log_benchmark_report_with_config(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="bench-with-config")

        report_dict = {
            "benchmark_name": "code-review",
            "model": "gpt-4o",
            "summary": {"composite_mean": 0.75},
            "results": [],
            "failure_patterns": [],
        }

        tracker.log_benchmark_report(
            report_dict,
            benchmark_config={"chunk_size": 512, "top_k": 5},
        )

        meta = tracker.metadata
        assert meta is not None
        assert "benchmark/code-review/config" in meta.config

    def test_log_benchmark_report_rejects_invalid_type(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="invalid-test")

        with pytest.raises(TypeError, match="Expected EvalReport or dict"):
            tracker.log_benchmark_report("not a report")


# ── WandBTracker: Artifacts ──────────────────────────────────────────

class TestWandBTrackerArtifacts:
    def test_log_artifact_local_fallback(self):
        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as dstdir:
            # Create a dummy file in source dir
            test_file = Path(srcdir) / "test_report.md"
            test_file.write_text("# Test Report\n\nContent here.")

            tracker = WandBTracker(local_dir=dstdir)
            tracker.init_run(run_name="artifact-test")
            result = tracker.log_artifact(str(test_file), artifact_type="eval-output")

            assert result is not None
            assert Path(result).exists()
            assert Path(result).name == "test_report.md"

    def test_log_artifact_nonexistent_file(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="missing-artifact")
        result = tracker.log_artifact("/nonexistent/file.txt")
        assert result is None

    def test_log_artifact_custom_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "original.md"
            test_file.write_text("content")

            tracker = WandBTracker(local_dir=tmpdir)
            tracker.init_run(run_name="named-artifact")
            result = tracker.log_artifact(
                str(test_file), name="renamed-report.md"
            )

            assert result is not None
            assert Path(result).name == "renamed-report.md"

    def test_log_model_local_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "model.pkl"
            model_file.write_text("mock model data")

            tracker = WandBTracker(local_dir=tmpdir)
            tracker.init_run(run_name="model-test")
            result = tracker.log_model(
                str(model_file),
                model_name="agent-classifier-v1",
                metadata={"framework": "sklearn", "version": "1.0"},
                aliases=["latest", "production"],
            )

            assert result is not None
            model_dir = Path(result)
            assert model_dir.exists()
            assert (model_dir / "model.pkl").exists()

    def test_log_model_nonexistent_path(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="missing-model")
        result = tracker.log_model("/nonexistent/model.pkl", "bad-model")
        assert result is None


# ── WandBTracker: Properties ─────────────────────────────────────────

class TestWandBTrackerProperties:
    def test_run_property(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="prop-test")
        assert tracker.run is None  # No wandb installed
        assert tracker.is_available is False

    def test_metadata_property(self):
        tracker = WandBTracker()
        assert tracker.metadata is None
        tracker.init_run(run_name="prop-test")
        assert tracker.metadata is not None
        assert isinstance(tracker.metadata, RunMetadata)

    def test_summary_before_logging(self):
        tracker = WandBTracker()
        tracker.init_run(run_name="empty-summary")
        assert tracker.summary() == {}


# ── SweepParameter ───────────────────────────────────────────────────

class TestSweepParameter:
    def test_categorical_parameter(self):
        param = SweepParameter(
            name="model",
            values=["gpt-4o", "gpt-4o-mini", "claude-3-opus"],
        )
        d = param.to_wandb_dict()
        assert d["values"] == ["gpt-4o", "gpt-4o-mini", "claude-3-opus"]

    def test_uniform_parameter(self):
        param = SweepParameter(
            name="temperature",
            min=0.0,
            max=1.0,
            distribution="uniform",
            default=0.7,
        )
        d = param.to_wandb_dict()
        assert d["min"] == 0.0
        assert d["max"] == 1.0
        assert d["distribution"] == "uniform"

    def test_discrete_parameter(self):
        param = SweepParameter(
            name="top_k",
            values=[1, 3, 5, 10],
            distribution="discrete",
        )
        d = param.to_wandb_dict()
        assert d["values"] == [1, 3, 5, 10]
        assert d["distribution"] == "discrete"


# ── SweepConfig ──────────────────────────────────────────────────────

class TestSweepConfig:
    def test_basic_creation(self):
        config = SweepConfig(
            name="test-sweep",
            method="grid",
            metric="benchmark/support-tickets/composite_mean",
            goal="maximize",
        )
        assert config.name == "test-sweep"
        assert config.method == "grid"
        assert config.metric == "benchmark/support-tickets/composite_mean"
        assert config.goal == "maximize"

    def test_to_wandb_dict(self):
        config = SweepConfig(
            name="test-sweep",
            method="bayes",
            metric="loss",
            goal="minimize",
            parameters=[
                SweepParameter("lr", min=1e-5, max=1e-1, distribution="log_uniform"),
                SweepParameter("batch_size", values=[16, 32, 64]),
            ],
        )
        d = config.to_wandb_dict()
        assert d["name"] == "test-sweep"
        assert d["method"] == "bayes"
        assert d["metric"]["name"] == "loss"
        assert d["metric"]["goal"] == "minimize"
        assert "lr" in d["parameters"]
        assert "batch_size" in d["parameters"]
        assert d["parameters"]["lr"]["distribution"] == "log_uniform"

    def test_expand_grid_categorical(self):
        config = SweepConfig(
            name="grid-test",
            method="grid",
            metric="composite_mean",
            parameters=[
                SweepParameter("a", values=[1, 2, 3]),
                SweepParameter("b", values=["x", "y"]),
            ],
        )
        combos = config.expand_grid()
        assert len(combos) == 6  # 3 * 2
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 3, "b": "y"} in combos

    def test_expand_grid_single_param(self):
        config = SweepConfig(
            name="single-grid",
            method="grid",
            metric="score",
            parameters=[SweepParameter("lr", values=[0.001, 0.01, 0.1])],
        )
        combos = config.expand_grid()
        assert len(combos) == 3
        assert combos == [
            {"lr": 0.001},
            {"lr": 0.01},
            {"lr": 0.1},
        ]

    def test_expand_grid_numeric_range(self):
        config = SweepConfig(
            name="range-grid",
            method="grid",
            metric="score",
            parameters=[SweepParameter("threshold", min=0.0, max=1.0)],
        )
        combos = config.expand_grid()
        assert len(combos) == 5  # 5 discrete steps
        vals = sorted([c["threshold"] for c in combos])
        assert vals[0] == 0.0
        assert vals[-1] == 1.0

    def test_expand_grid_rejects_non_grid(self):
        config = SweepConfig(
            name="random-sweep",
            method="random",
            metric="score",
        )
        with pytest.raises(ValueError, match="expand_grid only works with method='grid'"):
            config.expand_grid()


# ── WandBSweep: Local Grid ───────────────────────────────────────────

class TestWandBSweepLocalGrid:
    def test_run_local_grid_basic(self):
        config = SweepConfig(
            name="basic-sweep",
            method="grid",
            metric="composite_mean",
            goal="maximize",
            parameters=[
                SweepParameter("groundedness", values=[0.7, 0.8, 0.9]),
                SweepParameter("tool_rate", values=[0.8, 0.9]),
            ],
        )

        sweeper = WandBSweep(config)

        def train_fn(params):
            score = params["groundedness"] * 0.6 + params["tool_rate"] * 0.4
            return {"composite_mean": score}

        results = sweeper.run_local_grid(train_fn)
        assert len(results) == 6  # 3 * 2
        assert sweeper.best_result is not None

        # Best should be at max groundedness + max tool_rate
        best = sweeper.best_result
        assert best["config"]["groundedness"] == 0.9
        assert best["config"]["tool_rate"] == 0.9

    def test_run_local_grid_saves_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SweepConfig(
                name="save-test",
                method="grid",
                metric="score",
                goal="maximize",
                parameters=[SweepParameter("x", values=[1, 2])],
            )
            sweeper = WandBSweep(config)

            def train_fn(params):
                return {"score": params["x"] * 10}

            sweeper.run_local_grid(train_fn, output_dir=tmpdir)

            results_path = Path(tmpdir) / "sweep_save-test_results.json"
            assert results_path.exists()
            data = json.loads(results_path.read_text())
            assert len(data) == 2

            best_path = Path(tmpdir) / "sweep_save-test_best.json"
            assert best_path.exists()
            best_data = json.loads(best_path.read_text())
            assert best_data["config"]["x"] == 2

    def test_run_local_grid_handles_errors(self):
        config = SweepConfig(
            name="error-sweep",
            method="grid",
            metric="score",
            goal="maximize",
            parameters=[SweepParameter("fail", values=[True, False])],
        )
        sweeper = WandBSweep(config)

        def train_fn(params):
            if params["fail"]:
                raise ValueError("simulated failure")
            return {"score": 100.0}

        results = sweeper.run_local_grid(train_fn)
        assert len(results) == 2
        # One succeeded, one failed
        errors = [r for r in results if "error" in r.get("metrics", {})]
        successes = [r for r in results if "error" not in r.get("metrics", {})]
        assert len(errors) == 1
        assert len(successes) == 1
        # Best should be the successful one
        assert sweeper.best_result is not None
        assert sweeper.best_result["metrics"]["score"] == 100.0

    def test_run_local_grid_minimize_goal(self):
        config = SweepConfig(
            name="minimize-sweep",
            method="grid",
            metric="loss",
            goal="minimize",
            parameters=[SweepParameter("x", values=[1, 10, 100])],
        )
        sweeper = WandBSweep(config)

        def train_fn(params):
            return {"loss": params["x"]}

        sweeper.run_local_grid(train_fn)
        assert sweeper.best_result is not None
        assert sweeper.best_result["config"]["x"] == 1

    def test_create_requires_wandb(self):
        config = SweepConfig(
            name="missing-wandb",
            method="grid",
            metric="score",
        )
        sweeper = WandBSweep(config)
        with pytest.raises(RuntimeError, match="W&B is not installed"):
            sweeper.create()

    def test_run_agent_requires_created_sweep(self):
        config = SweepConfig(
            name="no-sweep-yet",
            method="grid",
            metric="score",
        )
        sweeper = WandBSweep(config)
        with pytest.raises(RuntimeError, match="Sweep not created or W&B unavailable"):
            sweeper.run_agent(lambda c: {"score": 0})


# ── Pre-built Sweep Configurations ───────────────────────────────────

class TestPrebuiltSweeps:
    def test_agent_profile_sweep(self):
        config = agent_profile_sweep()
        assert config.name == "agent-profile-sweep"
        assert config.method == "grid"
        assert config.goal == "maximize"
        param_names = [p.name for p in config.parameters]
        assert "groundedness_target" in param_names
        assert "verification_pass_rate" in param_names
        assert "tool_success_rate" in param_names
        assert "hallucination_rate" in param_names
        # Expand grid
        combos = config.expand_grid()
        assert len(combos) == 81  # 3 * 3 * 3 * 3

    def test_retrieval_sweep(self):
        config = retrieval_sweep()
        assert config.name == "retrieval-sweep"
        assert "citation_precision_mean" in config.metric
        param_names = [p.name for p in config.parameters]
        assert "chunk_size" in param_names
        assert "retrieval_method" in param_names
        combos = config.expand_grid()
        assert len(combos) == 81  # 3 * 3 * 3 * 3


# ── Utilities ────────────────────────────────────────────────────────

class TestUtilities:
    def test_generate_run_id(self):
        rid1 = _generate_run_id()
        rid2 = _generate_run_id()
        assert len(rid1) == 12
        assert len(rid2) == 12
        assert rid1 != rid2  # Uniqueness

    def test_run_metadata_fields(self):
        meta = RunMetadata(
            run_name="test",
            project="test-project",
            entity="test-entity",
        )
        assert meta.run_name == "test"
        assert meta.project == "test-project"
        assert meta.entity == "test-entity"
        assert meta.status == "running"
        assert meta.metrics_history == []
        assert meta.artifacts == []
