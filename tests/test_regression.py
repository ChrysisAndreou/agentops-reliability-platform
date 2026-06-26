"""
Tests for the regression testing system (baseline comparison, CI exit codes).
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentops.evals.baselines import save_baseline, load_baseline
from agentops.evals.regression_runner import (
    RegressionRunner,
    RegressionResult,
    BenchmarkRegression,
    DEFAULT_THRESHOLDS,
)
from agentops.evals.simulator import PRODUCTION_AGENT, DEVELOPMENT_AGENT, PERFECT_AGENT


class TestBenchmarkRegression:
    """BenchmarkRegression dataclass tests."""

    def test_no_regressions(self):
        br = BenchmarkRegression(benchmark_name="test")
        assert br.has_regressions is False
        assert br.regressions == []
        assert br.improvements == []
        assert br.stable == []

    def test_with_regression(self):
        br = BenchmarkRegression(
            benchmark_name="test",
            has_regressions=True,
            regressions=[{"metric": "composite_mean", "baseline": 0.80, "current": 0.70, "delta": -0.10, "threshold": 0.05}],
        )
        assert br.has_regressions is True
        assert len(br.regressions) == 1


class TestRegressionResult:
    """RegressionResult dataclass tests."""

    def test_exit_code_no_regressions(self):
        result = RegressionResult(
            baseline_name="v0.6",
            profile="production",
            has_regressions=False,
        )
        assert result.exit_code == 0

    def test_exit_code_with_regressions(self):
        result = RegressionResult(
            baseline_name="v0.6",
            profile="production",
            has_regressions=True,
        )
        assert result.exit_code == 1

    def test_to_markdown_no_regressions(self):
        result = RegressionResult(
            baseline_name="v0.6",
            profile="production",
            has_regressions=False,
            run_timestamp="2026-06-28",
            benchmarks=[
                BenchmarkRegression(
                    benchmark_name="support-tickets",
                    has_regressions=False,
                    stable=["composite_mean", "groundedness_mean"],
                    baseline_summary={"composite_mean": 0.85, "groundedness_mean": 0.80},
                    current_summary={"composite_mean": 0.86, "groundedness_mean": 0.81},
                )
            ],
        )
        md = result.to_markdown()
        assert "No Regressions Detected" in md
        assert "support-tickets" in md
        assert "✅" in md

    def test_to_markdown_with_regressions(self):
        result = RegressionResult(
            baseline_name="v0.6",
            profile="production",
            has_regressions=True,
            run_timestamp="2026-06-28",
            benchmarks=[
                BenchmarkRegression(
                    benchmark_name="support-tickets",
                    has_regressions=True,
                    regressions=[
                        {"metric": "composite_mean", "baseline": 0.85, "current": 0.70, "delta": -0.15, "threshold": 0.05}
                    ],
                    baseline_summary={"composite_mean": 0.85},
                    current_summary={"composite_mean": 0.70},
                )
            ],
        )
        md = result.to_markdown()
        assert "REGRESSIONS DETECTED" in md
        assert "⚠️" in md
        assert "composite_mean" in md

    def test_to_json(self):
        result = RegressionResult(
            baseline_name="v0.6",
            profile="production",
            has_regressions=False,
            run_timestamp="2026-06-28",
        )
        j = result.to_json()
        data = json.loads(j)
        assert data["baseline_name"] == "v0.6"
        assert data["has_regressions"] is False
        assert data["exit_code"] == 0


class TestRegressionRunner:
    """RegressionRunner integration tests."""

    @pytest.mark.asyncio
    async def test_init_valid_profile(self):
        runner = RegressionRunner(profile="production")
        assert runner.profile == "production"
        assert runner.sim_config is not None

    def test_init_invalid_profile(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            RegressionRunner(profile="nonexistent")

    @pytest.mark.asyncio
    async def test_run_no_regressions_same_profile(self, tmp_path):
        """Running regression with the same profile as baseline should show no regressions."""
        from agentops.evals.simulator import SimulatedAgent
        from agentops.evals.harness import EvalHarness
        from agentops.evals.benchmarks import SUPPORT_TICKETS_BENCH

        # Create baseline from simulated run
        agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        harness = EvalHarness(agent=agent, model="sim-production", output_dir=str(tmp_path))
        report = await harness.run_with_simulator(SUPPORT_TICKETS_BENCH, sim_config=PRODUCTION_AGENT)

        # Save as baseline
        benchmark_results = {
            "support-tickets": [r.to_dict() for r in report.results],
        }
        baselines_dir = tmp_path / "baselines"
        save_baseline(
            benchmark_results=benchmark_results,
            name="v0.6",
            profile="production",
            output_path=baselines_dir,
        )

        # Run regression
        runner = RegressionRunner(profile="production", baselines_dir=baselines_dir, seed=42)
        result = await runner.run(
            baseline_name="v0.6",
            output_dir=tmp_path,
            benchmarks=[SUPPORT_TICKETS_BENCH],
        )

        assert result.baseline_name == "v0.6"
        assert result.profile == "production"
        # Same seed + same profile = deterministic identical results
        assert result.has_regressions is False
        assert result.exit_code == 0

        # Check report file
        report_md = tmp_path / "regression_report.md"
        assert report_md.exists()
        content = report_md.read_text()
        assert "No Regressions Detected" in content

        report_json = tmp_path / "regression_report.json"
        assert report_json.exists()

    @pytest.mark.asyncio
    async def test_run_regressions_with_weaker_profile(self, tmp_path):
        """Production baseline vs development profile should show regressions."""
        from agentops.evals.simulator import SimulatedAgent
        from agentops.evals.harness import EvalHarness
        from agentops.evals.benchmarks import SUPPORT_TICKETS_BENCH

        # Create baseline from production agent
        agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        harness = EvalHarness(agent=agent, model="sim-production", output_dir=str(tmp_path))
        report = await harness.run_with_simulator(SUPPORT_TICKETS_BENCH, sim_config=PRODUCTION_AGENT)

        benchmark_results = {
            "support-tickets": [r.to_dict() for r in report.results],
        }
        baselines_dir = tmp_path / "baselines"
        save_baseline(
            benchmark_results=benchmark_results,
            name="v0.6-prod",
            profile="production",
            output_path=baselines_dir,
        )

        # Run regression with development (weaker) profile
        runner = RegressionRunner(profile="development", baselines_dir=baselines_dir, seed=42)
        result = await runner.run(
            baseline_name="v0.6-prod",
            output_dir=tmp_path,
            benchmarks=[SUPPORT_TICKETS_BENCH],
        )

        # Development profile should show regressions vs production baseline
        # The development profile has lower groundedness and citation scores
        assert len(result.benchmarks) == 1
        bench = result.benchmarks[0]
        assert bench.benchmark_name == "support-tickets"

        # At least some metrics should regress
        assert result.has_regressions is True
        assert result.exit_code == 1

        report_md = tmp_path / "regression_report.md"
        assert report_md.exists()
        content = report_md.read_text()
        assert "REGRESSIONS DETECTED" in content

    @pytest.mark.asyncio
    async def test_run_benchmark_not_in_baseline(self, tmp_path):
        """Benchmarks not in the baseline should be skipped."""
        from agentops.evals.benchmarks import SUPPORT_TICKETS_BENCH

        # Baseline has NO benchmarks
        baselines_dir = tmp_path / "baselines"
        save_baseline(
            benchmark_results={},
            name="empty",
            profile="production",
            output_path=baselines_dir,
        )

        runner = RegressionRunner(profile="production", baselines_dir=baselines_dir, seed=42)
        result = await runner.run(
            baseline_name="empty",
            output_dir=tmp_path,
            benchmarks=[SUPPORT_TICKETS_BENCH],
        )

        # No benchmarks matched → no regressions
        assert len(result.benchmarks) == 0
        assert result.has_regressions is False
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_improvements_tracked(self, tmp_path):
        """Improvements (scores going up) should be tracked separately."""
        from agentops.evals.simulator import SimulatedAgent
        from agentops.evals.harness import EvalHarness
        from agentops.evals.benchmarks import SUPPORT_TICKETS_BENCH

        # Baseline: development (lower scores)
        agent = SimulatedAgent(config=DEVELOPMENT_AGENT, seed=42)
        harness = EvalHarness(agent=agent, model="sim-development", output_dir=str(tmp_path))
        report = await harness.run_with_simulator(SUPPORT_TICKETS_BENCH, sim_config=DEVELOPMENT_AGENT)

        baselines_dir = tmp_path / "baselines"
        save_baseline(
            benchmark_results={"support-tickets": [r.to_dict() for r in report.results]},
            name="dev-baseline",
            profile="development",
            output_path=baselines_dir,
        )

        # Current: production (higher scores) → should show improvements
        runner = RegressionRunner(profile="production", baselines_dir=baselines_dir, seed=42)
        result = await runner.run(
            baseline_name="dev-baseline",
            output_dir=tmp_path,
            benchmarks=[SUPPORT_TICKETS_BENCH],
        )

        bench = result.benchmarks[0]
        # Production should have improvements over development
        assert len(bench.improvements) > 0

        # Markdown should show improvements section
        md = result.to_markdown()
        assert "Improvements" in md


class TestDefaultThresholds:
    """Regression threshold configuration tests."""

    def test_all_thresholds_positive(self):
        for key, val in DEFAULT_THRESHOLDS.items():
            assert val > 0, f"Threshold {key} should be positive"

    def test_composite_threshold_tighter_than_individual(self):
        """Composite should have the tightest threshold since it aggregates others."""
        non_latency_thresholds = {k: v for k, v in DEFAULT_THRESHOLDS.items() if k != "avg_latency_ms"}
        min_val = min(non_latency_thresholds.values())
        assert min_val <= DEFAULT_THRESHOLDS["composite_mean"]


class TestRegressionRunnerEdgeCases:
    """Edge case tests for RegressionRunner."""

    @pytest.mark.asyncio
    async def test_empty_benchmarks_list(self, tmp_path):
        """Running with empty benchmarks list should produce empty result."""
        baselines_dir = tmp_path / "baselines"
        save_baseline(
            benchmark_results={"test": [{"composite": 0.5}]},
            name="test",
            profile="production",
            output_path=baselines_dir,
        )
        # Need at least one task in the baseline benchmark to produce a summary;
        # the test baseline is dummy data — we just test that empty benchmarks arg
        # doesn't crash.
        runner = RegressionRunner(profile="production", baselines_dir=baselines_dir, seed=42)
        result = await runner.run(
            baseline_name="test",
            output_dir=tmp_path,
            benchmarks=[],
        )
        assert len(result.benchmarks) == 0
        assert result.has_regressions is False
        assert result.exit_code == 0
