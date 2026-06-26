"""
Tests for model benchmarking (cross-model comparison framework).
"""

import pytest

from agentops.evals.model_benchmark import (
    ModelProfile,
    ModelComparisonResult,
    MultiModelReport,
    ModelBenchmark,
    MODEL_PROFILES,
)
from agentops.evals.judge.state import JudgeConfig


# ── ModelProfile Tests ───────────────────────────────────────────────

class TestModelProfile:
    def test_create_profile(self):
        p = ModelProfile(
            name="gpt-4o",
            provider="openai",
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
            avg_latency_ms=800,
        )
        assert p.name == "gpt-4o"
        assert p.provider == "openai"
        assert p.cost_per_1k_input == 0.0025

    def test_to_dict(self):
        p = ModelProfile(name="test", provider="test", cost_per_1k_input=0.001)
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["cost_per_1k_input"] == 0.001


# ── ModelComparisonResult Tests ──────────────────────────────────────

class TestModelComparisonResult:
    def test_create_result(self):
        p = ModelProfile(name="gpt-4o", provider="openai")
        r = ModelComparisonResult(
            model=p,
            benchmark_name="test-bench",
            mean_composite=0.85,
            pass_rate=0.9,
        )
        assert r.model.name == "gpt-4o"
        assert r.mean_composite == 0.85

    def test_to_dict(self):
        p = ModelProfile(name="gpt-4o", provider="openai")
        r = ModelComparisonResult(
            model=p,
            benchmark_name="test",
            mean_composite=0.85,
            pass_rate=0.9,
            dimension_scores={"accuracy": 0.9, "safety": 0.8},
        )
        d = r.to_dict()
        assert d["mean_composite"] == 0.85
        assert d["dimension_scores"]["accuracy"] == 0.9


# ── MultiModelReport Tests ───────────────────────────────────────────

class TestMultiModelReport:
    def _make_result(self, model_name, composite, pass_rate, cost=0.001):
        p = MODEL_PROFILES.get(model_name, ModelProfile(name=model_name, provider="test"))
        return ModelComparisonResult(
            model=p,
            benchmark_name="test-bench",
            mean_composite=composite,
            pass_rate=pass_rate,
            dimension_scores={"accuracy": composite, "safety": 0.85, "completeness": 0.8},
            estimated_cost_usd=cost,
            total_latency_ms=500,
            task_results=[
                {"task_id": "t1", "composite_score": composite},
                {"task_id": "t2", "composite_score": composite - 0.1},
            ],
        )

    def test_to_markdown(self):
        report = MultiModelReport(
            title="Test Comparison",
            benchmark_name="test-bench",
            results=[
                self._make_result("gpt-4o", 0.9, 0.95),
                self._make_result("claude-3-sonnet", 0.85, 0.9),
                self._make_result("deepseek-v4", 0.8, 0.85),
            ],
        )
        md = report.to_markdown()
        assert "# Test Comparison" in md
        assert "## Model Rankings" in md
        assert "gpt-4o" in md
        assert "claude-3-sonnet" in md
        assert "deepseek-v4" in md
        assert "## Dimension Comparison" in md
        assert "## Cost-Performance Analysis" in md

    def test_to_json(self):
        report = MultiModelReport(
            title="Test",
            benchmark_name="test-bench",
            results=[self._make_result("gpt-4o", 0.9, 0.95)],
        )
        js = report.to_json()
        assert '"title": "Test"' in js
        assert '"benchmark_name": "test-bench"' in js

    def test_empty_report(self):
        report = MultiModelReport(
            title="Empty",
            benchmark_name="empty",
        )
        md = report.to_markdown()
        assert "0" in md  # At least has some content


# ── ModelBenchmark Tests ─────────────────────────────────────────────

class TestModelBenchmark:
    def test_compare_models(self):
        bench = ModelBenchmark(use_simulated=True)
        outputs = {
            "gpt-4o": {
                "st-001": {
                    "output": "Check Docker daemon and pipeline settings in clouddeploy.yml.",
                    "key_terms": ["docker", "pipeline", "settings", "Dedicated"],
                },
                "st-002": {
                    "output": "Enable TOTP in Security settings. Recovery codes are provided.",
                    "key_terms": ["TOTP", "SMS", "Security", "recovery codes"],
                },
            },
            "claude-3-sonnet": {
                "st-001": {
                    "output": "Verify Docker runtime is enabled in pipeline configuration.",
                    "key_terms": ["docker", "pipeline", "settings", "Dedicated"],
                },
                "st-002": {
                    "output": "2FA options: TOTP authenticator app or SMS verification.",
                    "key_terms": ["TOTP", "SMS", "Security", "recovery codes"],
                },
            },
        }
        report = bench.compare(
            models=["gpt-4o", "claude-3-sonnet"],
            benchmark_name="support-tickets",
            agent_outputs=outputs,
        )
        assert isinstance(report, MultiModelReport)
        assert len(report.results) == 2
        # Should have rankings
        assert "gpt-4o" in report.rankings
        assert "claude-3-sonnet" in report.rankings
        # Markdown report should be generated
        md = report.to_markdown()
        assert "## Model Rankings" in md

    def test_compare_single_model(self):
        bench = ModelBenchmark(use_simulated=True)
        outputs = {
            "deepseek-v4": {
                "st-001": {
                    "output": "Check Docker daemon and pipeline configuration.",
                    "key_terms": ["docker", "pipeline"],
                },
            },
        }
        report = bench.compare(
            models=["deepseek-v4"],
            benchmark_name="test",
            agent_outputs=outputs,
        )
        assert len(report.results) == 1
        assert report.rankings["deepseek-v4"] == 1

    def test_pareto_frontier(self):
        bench = ModelBenchmark(use_simulated=True)
        r1 = ModelComparisonResult(
            model=ModelProfile(name="model-a", provider="test"),
            benchmark_name="test",
            mean_composite=0.9,
            pass_rate=0.95,
            estimated_cost_usd=0.01,
        )
        r2 = ModelComparisonResult(
            model=ModelProfile(name="model-b", provider="test"),
            benchmark_name="test",
            mean_composite=0.85,
            pass_rate=0.9,
            estimated_cost_usd=0.001,
        )
        # model-a has better score, model-b has better cost
        pareto = bench._compute_pareto([r1, r2])
        # Both should be on Pareto frontier (neither dominates the other)
        assert "model-a" in pareto
        assert "model-b" in pareto

        # Add a dominated model
        r3 = ModelComparisonResult(
            model=ModelProfile(name="model-c", provider="test"),
            benchmark_name="test",
            mean_composite=0.8,
            pass_rate=0.85,
            estimated_cost_usd=0.02,
        )
        pareto_all = bench._compute_pareto([r1, r2, r3])
        assert "model-c" not in pareto_all  # Dominated by model-a

    def test_compare_across_benchmarks(self):
        bench = ModelBenchmark(use_simulated=True)
        outputs = {
            "support-tickets": {
                "gpt-4o": {
                    "st-001": {"output": "Check Docker configuration.", "key_terms": ["docker"]},
                },
            },
            "systems-quality": {
                "gpt-4o": {
                    "sq-001": {"output": "Run health checks.", "key_terms": ["health"]},
                },
            },
        }
        reports = bench.compare_across_benchmarks(
            models=["gpt-4o"],
            agent_outputs=outputs,
        )
        assert len(reports) == 2
        assert "support-tickets" in reports
        assert "systems-quality" in reports

    def test_generate_aggregate_report(self):
        bench = ModelBenchmark(use_simulated=True)

        p1 = ModelProfile(name="gpt-4o", provider="openai")
        p2 = ModelProfile(name="claude-3-sonnet", provider="anthropic")

        r1 = ModelComparisonResult(
            model=p1, benchmark_name="bench1",
            mean_composite=0.9, pass_rate=0.95,
            dimension_scores={"accuracy": 0.9},
        )
        r2 = ModelComparisonResult(
            model=p2, benchmark_name="bench1",
            mean_composite=0.85, pass_rate=0.9,
            dimension_scores={"accuracy": 0.85},
        )

        report1 = MultiModelReport(
            title="Bench 1", benchmark_name="bench1",
            results=[r1, r2],
            rankings={"gpt-4o": 1, "claude-3-sonnet": 2},
        )
        report2 = MultiModelReport(
            title="Bench 2", benchmark_name="bench2",
            results=[r1, r2],
            rankings={"claude-3-sonnet": 1, "gpt-4o": 2},
        )

        agg = bench.generate_aggregate_report({"bench1": report1, "bench2": report2})
        assert "Aggregate Model Comparison" in agg
        assert "Overall Rankings" in agg
        assert "gpt-4o" in agg


# ── MODEL_PROFILES Tests ─────────────────────────────────────────────

class TestModelProfiles:
    def test_all_profiles_valid(self):
        assert len(MODEL_PROFILES) >= 7
        for name, profile in MODEL_PROFILES.items():
            assert isinstance(profile, ModelProfile)
            assert profile.name == name
            assert profile.provider != ""

    def test_simulated_models_have_zero_cost(self):
        assert MODEL_PROFILES["simulated-production"].cost_per_1k_input == 0.0
        assert MODEL_PROFILES["simulated-development"].cost_per_1k_output == 0.0

    def test_production_models_have_cost(self):
        assert MODEL_PROFILES["gpt-4o"].cost_per_1k_input > 0
        assert MODEL_PROFILES["claude-3-opus"].cost_per_1k_output > 0


# ── Edge Cases ───────────────────────────────────────────────────────

class TestModelBenchmarkEdgeCases:
    def test_empty_models_list(self):
        bench = ModelBenchmark(use_simulated=True)
        report = bench.compare(
            models=[],
            benchmark_name="test",
            agent_outputs={},
        )
        assert len(report.results) == 0

    def test_unknown_model_fallback(self):
        bench = ModelBenchmark(use_simulated=True)
        outputs = {
            "nonexistent-model": {
                "t1": {"output": "test", "key_terms": ["test"]},
            },
        }
        report = bench.compare(
            models=["nonexistent-model"],
            benchmark_name="test",
            agent_outputs=outputs,
        )
        assert len(report.results) == 1
        assert report.results[0].model.name == "nonexistent-model"
        assert report.results[0].model.provider == "unknown"

    def test_infinite_cost_efficiency(self):
        # Zero cost should produce inf (handled in to_markdown)
        r = ModelComparisonResult(
            model=ModelProfile(name="free-model", provider="test", cost_per_1k_input=0.0),
            benchmark_name="test",
            mean_composite=0.9,
            pass_rate=0.95,
            estimated_cost_usd=0.0,
        )
        assert r.estimated_cost_usd == 0.0
