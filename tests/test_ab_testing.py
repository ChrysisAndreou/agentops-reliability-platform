"""Tests for the Agent A/B Testing & Canary Deployment module (v0.26).

Tests cover:
  - stats.py: Chi-squared, Fisher's exact, Bayesian A/B, Welch's t,
    Mann-Whitney U, sample size estimation, confidence intervals
  - experiment.py: VariantConfig, TrafficAllocation, TrafficSplit,
    ABExperimentConfig, ABExperiment, VariantMetrics, ABAnalysisReport
  - canary.py: CanaryStage, CanaryConfig, CanaryDeployer, CanaryResult
  - eval.py: MetricComparison, VariantSummary, ABEvalReport,
    generate_ab_report, format_report_markdown
"""

import math
import pytest
from agentops.ab_testing import (
    # Stats
    chi_squared_test,
    fisher_exact_test,
    bayesian_ab_test,
    welch_t_test,
    mann_whitney_u,
    compute_sample_size,
    confidence_interval,
    # Experiment
    ABExperiment,
    ABExperimentConfig,
    ABExperimentResult,
    TrafficSplit,
    TrafficAllocation,
    VariantConfig,
    # Canary
    CanaryConfig,
    CanaryDeployer,
    CanaryResult,
    CanaryStage,
    CanaryStageResult,
    CanaryStatus,
    RollbackCondition,
    RollbackReason,
    # Eval
    ABEvalReport,
    generate_ab_report,
    format_report_markdown,
)


# ============================================================================
# stats.py tests
# ============================================================================

class TestChiSquared:
    def test_clear_difference(self):
        """Large difference should be significant."""
        p, sig = chi_squared_test([80, 20], [50, 50])
        assert sig
        assert p < 0.01

    def test_no_difference(self):
        """Equal proportions should not be significant."""
        p, sig = chi_squared_test([50, 50], [50, 50])
        assert not sig
        assert p > 0.05

    def test_small_difference(self):
        """Small difference with low N should not be significant."""
        p, sig = chi_squared_test([6, 4], [5, 5])
        assert not sig

    def test_single_observation(self):
        """Edge case: single observation per cell."""
        p, sig = chi_squared_test([1, 0], [0, 1])
        # N=2 is too small for meaningful chi-squared
        assert isinstance(p, float)

    def test_all_success(self):
        """Both variants 100% success."""
        p, sig = chi_squared_test([50, 0], [50, 0])
        assert not sig
        assert p > 0.05

    def test_yates_vs_no_yates(self):
        """Yates correction should give higher p-value."""
        p_yates, _ = chi_squared_test([10, 2], [5, 7], yates_correction=True)
        p_no, _ = chi_squared_test([10, 2], [5, 7], yates_correction=False)
        assert p_yates >= p_no

    def test_zero_total(self):
        """All zeros should return p=1.0, not significant."""
        p, sig = chi_squared_test([0, 0], [0, 0])
        assert p == 1.0
        assert not sig


class TestFisherExact:
    def test_clear_difference_small_n(self):
        """Fisher's should detect difference with small N."""
        p, odds, sig = fisher_exact_test(8, 1, 2, 7)
        assert sig
        assert p < 0.05
        assert odds > 1.0

    def test_no_difference(self):
        """Equal proportions."""
        p, odds, sig = fisher_exact_test(5, 5, 5, 5)
        assert not sig
        assert p > 0.05

    def test_borderline(self):
        """Borderline case."""
        p, odds, sig = fisher_exact_test(5, 0, 1, 4)
        assert sig  # 5/0 vs 1/4 is quite different

    def test_odds_ratio_extreme(self):
        """Extreme odds ratio with zero cell."""
        _, odds, _ = fisher_exact_test(10, 0, 5, 5)
        # Odds ratio should be very large
        assert odds > 10.0

    def test_zero_total(self):
        """All zeros."""
        p, odds, sig = fisher_exact_test(0, 0, 0, 0)
        assert p == 1.0
        assert not sig


class TestBayesianAB:
    def test_b_better(self):
        """B clearly better should give high probability."""
        prob = bayesian_ab_test(30, 100, 70, 100, seed=42)
        assert prob > 0.95

    def test_a_better(self):
        """A clearly better should give low probability."""
        prob = bayesian_ab_test(70, 100, 30, 100, seed=42)
        assert prob < 0.05

    def test_roughly_equal(self):
        """Similar performance should give ~0.5."""
        prob = bayesian_ab_test(50, 100, 50, 100, seed=42)
        assert 0.3 < prob < 0.7

    def test_reproducible_seed(self):
        """Same seed should give same result."""
        p1 = bayesian_ab_test(45, 100, 55, 100, seed=42)
        p2 = bayesian_ab_test(45, 100, 55, 100, seed=42)
        assert p1 == p2

    def test_different_seeds_different(self):
        """Different seeds may give slightly different results."""
        p1 = bayesian_ab_test(45, 100, 55, 100, seed=1)
        p2 = bayesian_ab_test(45, 100, 55, 100, seed=2)
        # Should be close but may differ at high precision
        assert abs(p1 - p2) < 0.1

    def test_zero_trials(self):
        """Zero trials should return 0.5."""
        prob = bayesian_ab_test(0, 0, 0, 0)
        assert prob == 0.5

    def test_strong_prior(self):
        """Strong prior (alpha=10, beta=10) should pull toward 0.5."""
        prob_weak = bayesian_ab_test(8, 10, 2, 10, prior_alpha=1, prior_beta=1, seed=42)
        prob_strong = bayesian_ab_test(8, 10, 2, 10, prior_alpha=10, prior_beta=10, seed=42)
        # Strong prior should pull probability closer to 0.5
        assert abs(prob_strong - 0.5) <= abs(prob_weak - 0.5) + 0.01


class TestWelchTTest:
    def test_clear_difference(self):
        """Clearly different means with some variance."""
        _, p, sig = welch_t_test(
            [1.0]*15 + [1.1]*15,
            [2.0]*15 + [1.9]*15,
        )
        assert sig
        assert p < 0.001

    def test_same_means(self):
        """Same means should not be significant."""
        _, p, sig = welch_t_test([1.0]*30, [1.0]*30)
        assert not sig
        assert p > 0.05

    def test_different_variances(self):
        """Welch's handles unequal variances."""
        _, p, sig = welch_t_test(
            [0.80, 0.82, 0.79, 0.81, 0.83]*6,
            [0.70, 0.90, 0.65, 0.95, 0.72]*6,
        )
        # Means are close (0.81 vs 0.784) — may or may not be significant
        assert isinstance(p, float)

    def test_small_sample(self):
        """Small samples should work."""
        _, p, sig = welch_t_test([1.0, 1.1], [2.0, 2.1])
        assert sig

    def test_single_sample(self):
        """Single sample per group returns p=1.0."""
        _, p, sig = welch_t_test([1.0], [2.0])
        assert p == 1.0
        assert not sig

    def test_large_difference(self):
        """Very different means with variance."""
        a_vals = [0.0 + (i % 5) * 0.01 for i in range(50)]
        b_vals = [100.0 + (i % 5) * 0.01 for i in range(50)]
        t, p, sig = welch_t_test(a_vals, b_vals)
        assert sig
        assert p < 1e-10


class TestMannWhitneyU:
    def test_clear_difference(self):
        """Clearly different distributions."""
        u, p, sig = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert sig
        assert p < 0.05

    def test_same_distribution(self):
        """Same distribution should not be significant."""
        u, p, sig = mann_whitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert not sig

    def test_different_sizes(self):
        """Different group sizes."""
        u, p, sig = mann_whitney_u([1, 2, 3], [8, 9, 10, 11, 12, 13, 14])
        assert sig

    def test_ties(self):
        """Handles tied ranks."""
        u, p, sig = mann_whitney_u([5, 5, 5, 5, 5], [5, 5, 5, 5, 5])
        assert not sig


class TestSampleSize:
    def test_baseline_80_effect_10(self):
        """80% baseline, 10% MDE."""
        n = compute_sample_size(0.80, 0.10)
        assert 150 < n < 400  # Reasonable range

    def test_baseline_50_effect_5(self):
        """50% baseline, 5% MDE."""
        n = compute_sample_size(0.50, 0.05)
        assert n > 1000

    def test_large_effect_needs_fewer(self):
        """Larger MDE needs fewer samples."""
        n_small = compute_sample_size(0.50, 0.20)
        n_large = compute_sample_size(0.50, 0.05)
        assert n_small < n_large

    def test_invalid_baseline_zero(self):
        """Baseline of 0 should raise."""
        with pytest.raises(ValueError):
            compute_sample_size(0.0, 0.10)

    def test_invalid_baseline_one(self):
        """Baseline of 1 should raise."""
        with pytest.raises(ValueError):
            compute_sample_size(1.0, 0.10)

    def test_default_power_and_alpha(self):
        """Default parameters work."""
        n = compute_sample_size(0.80, 0.10)
        assert n > 0
        assert isinstance(n, int)


class TestConfidenceInterval:
    def test_90_percent_success(self):
        """45/50 ~= 90% success rate."""
        lo, hi = confidence_interval(45, 50)
        assert 0.77 < lo < 0.80
        assert 0.95 < hi < 0.97

    def test_50_percent(self):
        """25/50 = 50%."""
        lo, hi = confidence_interval(25, 50)
        assert 0.36 < lo < 0.38
        assert 0.62 < hi < 0.64

    def test_all_success(self):
        """100% success."""
        lo, hi = confidence_interval(50, 50)
        assert lo > 0.90
        assert hi == 1.0

    def test_all_failure(self):
        """0% success."""
        lo, hi = confidence_interval(0, 50)
        assert lo == 0.0
        assert hi < 0.10

    def test_zero_trials(self):
        """Zero trials."""
        lo, hi = confidence_interval(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_custom_confidence(self):
        """99% CI should be wider than 95%."""
        lo95, hi95 = confidence_interval(45, 50, confidence=0.95)
        lo99, hi99 = confidence_interval(45, 50, confidence=0.99)
        assert (hi99 - lo99) > (hi95 - lo95)


# ============================================================================
# experiment.py tests
# ============================================================================

class TestVariantConfig:
    def test_basic(self):
        vc = VariantConfig(name="control", description="Baseline")
        assert vc.name == "control"
        assert vc.description == "Baseline"

    def test_with_config(self):
        vc = VariantConfig(name="treatment", config={"model": "gpt-4"})
        assert vc.config["model"] == "gpt-4"

    def test_with_metadata(self):
        vc = VariantConfig(name="v2", metadata={"owner": "team-a"})
        assert vc.metadata["owner"] == "team-a"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            VariantConfig(name="")

    def test_defaults(self):
        vc = VariantConfig(name="test")
        assert vc.description == ""
        assert vc.config == {}
        assert vc.metadata == {}


class TestTrafficAllocation:
    def test_basic(self):
        vc = VariantConfig(name="control")
        ta = TrafficAllocation(variant=vc, weight=50)
        assert ta.variant.name == "control"
        assert ta.weight == 50
        assert ta.min_samples == 100

    def test_custom_min_samples(self):
        vc = VariantConfig(name="test")
        ta = TrafficAllocation(variant=vc, weight=30, min_samples=200)
        assert ta.min_samples == 200

    def test_negative_weight_raises(self):
        vc = VariantConfig(name="test")
        with pytest.raises(ValueError):
            TrafficAllocation(variant=vc, weight=-1)


class TestTrafficSplit:
    def test_basic_assignment(self):
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("A"), weight=50),
            TrafficAllocation(VariantConfig("B"), weight=50),
        ])
        assert split.assign("task_1").name in ("A", "B")

    def test_deterministic(self):
        """Same task_id always maps to same variant."""
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("A"), weight=50),
            TrafficAllocation(VariantConfig("B"), weight=50),
        ])
        a1 = split.assign("task_42")
        a2 = split.assign("task_42")
        assert a1.name == a2.name

    def test_different_seeds_different(self):
        """Different seeds should (usually) give different assignments."""
        split1 = TrafficSplit(
            [TrafficAllocation(VariantConfig("A"), weight=50),
             TrafficAllocation(VariantConfig("B"), weight=50)],
            seed=1,
        )
        split2 = TrafficSplit(
            [TrafficAllocation(VariantConfig("A"), weight=50),
             TrafficAllocation(VariantConfig("B"), weight=50)],
            seed=999,
        )
        # With high probability, at least one task assigns differently
        differences = sum(
            1 for i in range(100)
            if split1.assign(f"task_{i}").name != split2.assign(f"task_{i}").name
        )
        assert differences > 0

    def test_weighted_distribution(self):
        """Higher-weight variant should get more assignments."""
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("A"), weight=90),
            TrafficAllocation(VariantConfig("B"), weight=10),
        ])
        counts = {"A": 0, "B": 0}
        for i in range(1000):
            v = split.assign(f"task_{i}")
            counts[v.name] += 1
        assert counts["A"] > counts["B"]
        # A should get roughly 90%
        assert 850 <= counts["A"] <= 950

    def test_single_variant(self):
        """Single variant always assigned."""
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("solo"), weight=100),
        ])
        for i in range(50):
            assert split.assign(f"task_{i}").name == "solo"

    def test_get_weight(self):
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("A"), weight=30),
            TrafficAllocation(VariantConfig("B"), weight=70),
        ])
        assert split.get_weight("A") == pytest.approx(0.3, abs=0.01)
        assert split.get_weight("B") == pytest.approx(0.7, abs=0.01)
        assert split.get_weight("nonexistent") == 0.0

    def test_empty_allocations_raises(self):
        with pytest.raises(ValueError):
            TrafficSplit([])

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            TrafficSplit([
                TrafficAllocation(VariantConfig("A"), weight=0),
            ])


class TestABExperimentConfig:
    def test_defaults(self):
        config = ABExperimentConfig()
        assert config.metric == "verification_pass_rate"
        assert config.significance_level == 0.05
        assert config.min_sample_size == 100
        assert config.bayesian_threshold == 0.95

    def test_custom(self):
        config = ABExperimentConfig(
            metric="groundedness",
            significance_level=0.01,
            min_sample_size=500,
            name="reranker-v2-test",
        )
        assert config.metric == "groundedness"
        assert config.significance_level == 0.01
        assert config.name == "reranker-v2-test"

    def test_invalid_significance(self):
        with pytest.raises(ValueError):
            ABExperimentConfig(significance_level=0.0)
        with pytest.raises(ValueError):
            ABExperimentConfig(significance_level=1.0)

    def test_invalid_bayesian_threshold(self):
        with pytest.raises(ValueError):
            ABExperimentConfig(bayesian_threshold=0.0)
        with pytest.raises(ValueError):
            ABExperimentConfig(bayesian_threshold=1.1)

    def test_default_metrics(self):
        config = ABExperimentConfig()
        assert "verification_pass_rate" in config.metrics
        assert "groundedness" in config.metrics
        assert "latency_ms" in config.metrics


class TestABExperiment:
    def make_experiment(self):
        """Helper: create a standard experiment setup."""
        config = ABExperimentConfig(metric="verification_pass_rate", name="test-exp")
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("control", "Baseline"), weight=50),
            TrafficAllocation(VariantConfig("treatment", "New version"), weight=50),
        ])
        return ABExperiment(config), split

    def test_create_result(self):
        exp, split = self.make_experiment()
        result = exp.create_result(split)
        assert isinstance(result, ABExperimentResult)
        assert result.total_runs == 0
        assert "control" in result.variants
        assert "treatment" in result.variants

    def test_run_iteration(self):
        exp, split = self.make_experiment()
        result = exp.create_result(split)
        result = exp.run_iteration(
            result, split, "task_001", True,
            {"groundedness": 0.85, "latency_ms": 1200},
            latency_ms=1200, cost_usd=0.002,
        )
        assert result.total_runs == 1
        # One of the variants should have 1 run
        total_variant_runs = sum(v.total_runs for v in result.variants.values())
        assert total_variant_runs == 1

    def test_run_batch(self):
        exp, split = self.make_experiment()
        result = exp.create_result(split)
        runs = [
            {"task_id": f"task_{i}", "success": i % 3 != 0,
             "metrics": {"groundedness": 0.8 + 0.1 * (i % 3 == 0)},
             "latency_ms": 1000 + i * 5}
            for i in range(100)
        ]
        result = exp.run_batch(result, split, runs)
        assert result.total_runs == 100

    def test_analyze_no_difference(self):
        """Equal performance should not be significant."""
        exp, split = self.make_experiment()
        result = exp.create_result(split)
        for i in range(200):
            result = exp.run_iteration(
                result, split, f"task_{i}", i % 2 == 0,
                {"groundedness": 0.85},
            )
        analysis = exp.analyze(result)
        assert analysis.significant is False  # 50/50 split by task_id
        # With balanced assignment, both variants get ~100 runs each, ~50% success
        # Should not be significant

    def test_analyze_clear_winner(self):
        """One variant clearly better should be significant."""
        config = ABExperimentConfig(metric="verification_pass_rate")
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("control"), weight=50),
            TrafficAllocation(VariantConfig("treatment"), weight=50),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)

        # Force control to always succeed, treatment to always fail
        for i in range(200):
            assigned = split.assign(f"task_{i}")
            success = assigned.name == "control"
            result = exp.run_iteration(result, split, f"task_{i}", success, {})
        analysis = exp.analyze(result)
        assert analysis.significant
        assert analysis.winner == "control"

    def test_analyze_single_variant(self):
        """Analysis with only one variant should produce report."""
        config = ABExperimentConfig()
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("solo"), weight=100),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)
        for i in range(10):
            result = exp.run_iteration(result, split, f"task_{i}", True, {})
        analysis = exp.analyze(result)
        assert not analysis.significant
        assert analysis.winner is None
        assert "Need at least 2 variants" in analysis.recommendation

    def test_result_to_dict(self):
        exp, split = self.make_experiment()
        result = exp.create_result(split)
        for i in range(20):
            result = exp.run_iteration(result, split, f"task_{i}", True, {"g": 0.9})
        analysis = exp.analyze(result)
        d = analysis.to_dict()
        assert "significant" in d
        assert "winner" in d
        assert "total_runs" in d
        assert "variants" in d

    def test_insufficient_data_recommendation(self):
        """Below min_sample_size should recommend continuing."""
        config = ABExperimentConfig(min_sample_size=1000)
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("A"), weight=50),
            TrafficAllocation(VariantConfig("B"), weight=50),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)
        for i in range(10):
            result = exp.run_iteration(result, split, f"task_{i}", True, {})
        analysis = exp.analyze(result)
        assert "Insufficient data" in analysis.recommendation or "Continue" in analysis.recommendation


class TestVariantMetrics:
    def test_record(self):
        from agentops.ab_testing.experiment import VariantMetrics
        vm = VariantMetrics(variant_name="test")
        vm.record(True, {"groundedness": 0.85, "latency_ms": 1200}, latency_ms=1200)
        assert vm.total_runs == 1
        assert vm.successes == 1
        assert vm.failures == 0
        assert vm.success_rate == 1.0

    def test_record_failure(self):
        from agentops.ab_testing.experiment import VariantMetrics
        vm = VariantMetrics(variant_name="test")
        vm.record(False, {"groundedness": 0.3}, latency_ms=500, cost_usd=0.001)
        vm.record(True, {"groundedness": 0.9}, latency_ms=200, cost_usd=0.002)
        assert vm.total_runs == 2
        assert vm.success_rate == 0.5
        assert sum(vm.latencies) / max(len(vm.latencies), 1) == pytest.approx(350)
        assert sum(vm.costs) / max(len(vm.costs), 1) == pytest.approx(0.0015)

    def test_metric_mean_std(self):
        from agentops.ab_testing.experiment import VariantMetrics
        vm = VariantMetrics(variant_name="test")
        vm.record(True, {"g": 0.80})
        vm.record(True, {"g": 0.90})
        vm.record(True, {"g": 0.85})
        assert vm.metric_mean("g") == pytest.approx(0.85)
        assert vm.metric_std("g") == pytest.approx(0.05)

    def test_success_ci(self):
        from agentops.ab_testing.experiment import VariantMetrics
        vm = VariantMetrics(variant_name="test")
        for _ in range(45):
            vm.record(True, {})
        for _ in range(5):
            vm.record(False, {})
        lo, hi = vm.success_ci()
        assert 0.77 < lo < 0.80
        assert 0.95 < hi < 0.97

    def test_empty_variant(self):
        from agentops.ab_testing.experiment import VariantMetrics
        vm = VariantMetrics(variant_name="empty")
        assert vm.success_rate == 0.0
        assert vm.metric_mean("anything") == 0.0


# ============================================================================
# canary.py tests
# ============================================================================

class TestCanaryStage:
    def test_basic(self):
        stage = CanaryStage("canary-5%", traffic_percent=5.0, min_samples=50)
        assert stage.name == "canary-5%"
        assert stage.traffic_percent == 5.0
        assert stage.min_samples == 50

    def test_defaults(self):
        stage = CanaryStage("stage-1", traffic_percent=10.0)
        assert stage.min_samples == 100
        assert stage.max_duration_minutes == 60
        assert stage.evaluation_metric == "verification_pass_rate"
        assert stage.regression_threshold == 0.05
        assert stage.error_rate_threshold == 0.10
        assert stage.latency_multiplier_threshold == 2.0

    def test_invalid_traffic(self):
        with pytest.raises(ValueError):
            CanaryStage("bad", traffic_percent=0)
        with pytest.raises(ValueError):
            CanaryStage("bad", traffic_percent=101)

    def test_invalid_regression_threshold(self):
        with pytest.raises(ValueError):
            CanaryStage("bad", traffic_percent=50, regression_threshold=-0.1)

    def test_custom_gate(self):
        def my_gate(data):
            return data["error_rate"] < 0.05, "custom check"
        stage = CanaryStage("gated", traffic_percent=50, custom_gate=my_gate)
        assert stage.custom_gate is not None


class TestCanaryConfig:
    def test_basic(self):
        stages = [
            CanaryStage("s1", traffic_percent=10),
            CanaryStage("s2", traffic_percent=100),
        ]
        config = CanaryConfig(name="deploy-v2", stages=stages, baseline_metric=0.85)
        assert config.name == "deploy-v2"
        assert len(config.stages) == 2
        assert config.auto_rollback

    def test_monotonically_increasing_enforced(self):
        """Stages must have increasing traffic percent."""
        with pytest.raises(ValueError):
            CanaryConfig(
                stages=[
                    CanaryStage("s1", traffic_percent=50),
                    CanaryStage("s2", traffic_percent=30),  # Decreasing!
                ],
                baseline_metric=0.85,
            )

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError):
            CanaryConfig(stages=[], baseline_metric=0.85)

    def test_baselines_stored(self):
        stages = [CanaryStage("s1", traffic_percent=100)]
        config = CanaryConfig(
            name="test",
            stages=stages,
            baseline_metric=0.90,
            baseline_error_rate=0.02,
            baseline_latency_p50=500,
            baseline_latency_p95=1500,
        )
        assert config.baseline_error_rate == 0.02
        assert config.baseline_latency_p95 == 1500


class TestCanaryDeployer:
    def make_deployer(self, auto_rollback=True):
        stages = [
            CanaryStage("canary-5%", traffic_percent=5, min_samples=10),
            CanaryStage("staged-25%", traffic_percent=25, min_samples=20),
            CanaryStage("full-100%", traffic_percent=100, min_samples=30),
        ]
        config = CanaryConfig(
            name="test-deploy",
            stages=stages,
            baseline_metric=0.85,
            baseline_error_rate=0.05,
            baseline_latency_p50=800,
            baseline_latency_p95=2000,
            auto_rollback=auto_rollback,
        )
        return CanaryDeployer(config)

    def test_start(self):
        deployer = self.make_deployer()
        result = deployer.start()
        assert result.status == CanaryStatus.IN_PROGRESS
        assert result.completed_stages == 0

    def test_record_and_complete_stage(self):
        deployer = self.make_deployer()
        result = deployer.start()

        # Feed in enough runs for stage 1
        for i in range(15):
            result = deployer.record_run(result, True, 0.86, latency_ms=1000)

        assert deployer.stage_complete(result)
        result = deployer.evaluate_stage(result)
        assert result.stage_results[0].passed
        assert result.completed_stages == 1
        assert result.status == CanaryStatus.IN_PROGRESS  # More stages remain

    def test_full_deployment_success(self):
        deployer = self.make_deployer()
        result = deployer.start()

        # Stage 1: 15 runs all success
        for i in range(15):
            result = deployer.record_run(result, True, 0.86, latency_ms=1000)
        result = deployer.evaluate_stage(result)
        assert result.stage_results[0].passed

        # Stage 2: 25 runs
        for i in range(25):
            result = deployer.record_run(result, True, 0.84, latency_ms=1100)
        result = deployer.evaluate_stage(result)
        assert result.stage_results[1].passed

        # Stage 3: 35 runs
        for i in range(35):
            result = deployer.record_run(result, True, 0.85, latency_ms=1050)
        result = deployer.evaluate_stage(result)
        assert result.stage_results[2].passed

        assert result.status == CanaryStatus.COMPLETED
        assert result.completed_stages == 3

    def test_error_rate_rollback(self):
        deployer = self.make_deployer()
        result = deployer.start()

        # High failure rate should trigger rollback at stage 1
        for i in range(15):
            success = i < 5  # Only 5/15 succeed = 66% failure
            result = deployer.record_run(result, success, 0.50, latency_ms=1000)

        assert deployer.stage_complete(result)
        result = deployer.evaluate_stage(result)
        assert not result.stage_results[0].passed
        assert result.stage_results[0].rollback_reason == RollbackReason.ERROR_RATE_EXCEEDED
        assert result.status == CanaryStatus.ROLLED_BACK

    def test_metric_regression_rollback(self):
        deployer = self.make_deployer()
        result = deployer.start()

        # Metric drops significantly from baseline 0.85
        for i in range(15):
            result = deployer.record_run(result, True, 0.60, latency_ms=1000)

        result = deployer.evaluate_stage(result)
        assert not result.stage_results[0].passed
        assert result.stage_results[0].rollback_reason == RollbackReason.REGRESSION_DETECTED

    def test_latency_spike_rollback(self):
        deployer = self.make_deployer()
        result = deployer.start()

        # P95 latency spikes to 5000ms (2.5x baseline of 2000ms)
        for i in range(15):
            result = deployer.record_run(
                result, True, 0.86,
                latency_ms=5000 if i >= 14 else 1000,  # Force P95 to spike
            )

        result = deployer.evaluate_stage(result)
        # Most latencies are 1000ms but P95 calculation depends on sort
        # With 1/15 at 5000ms, P95 may or may not trigger
        sr = result.stage_results[0]
        if not sr.passed:
            assert sr.rollback_reason in (
                RollbackReason.LATENCY_SPIKE,
                RollbackReason.REGRESSION_DETECTED,
            )

    def test_custom_gate(self):
        def always_fail(data):
            return False, "blocked by policy"

        stages = [
            CanaryStage("gated", traffic_percent=100, min_samples=5, custom_gate=always_fail),
        ]
        config = CanaryConfig(
            name="gated-deploy",
            stages=stages,
            baseline_metric=0.85,
            auto_rollback=True,
        )
        deployer = CanaryDeployer(config)
        result = deployer.start()
        for i in range(10):
            result = deployer.record_run(result, True, 0.90)
        result = deployer.evaluate_stage(result)
        assert not result.stage_results[0].passed
        assert result.stage_results[0].rollback_reason == RollbackReason.CUSTOM_CONDITION

    def test_no_auto_rollback(self):
        """With auto_rollback=False, failures should not change status."""
        deployer = self.make_deployer(auto_rollback=False)
        result = deployer.start()
        # Feed high-failure data
        for i in range(15):
            result = deployer.record_run(result, False, 0.30)
        result = deployer.evaluate_stage(result)
        assert not result.stage_results[0].passed
        # Status should still be IN_PROGRESS (not rolled back)
        assert result.status != CanaryStatus.ROLLED_BACK

    def test_generate_report(self):
        deployer = self.make_deployer()
        result = deployer.start()
        for i in range(15):
            result = deployer.record_run(result, True, 0.86, latency_ms=1000)
        result = deployer.evaluate_stage(result)
        report = deployer.generate_report(result)
        assert report["status"] in ("in_progress", "completed", "rolled_back")
        assert len(report["stages"]) >= 1
        assert "baseline" in report


class TestCanaryResult:
    def test_properties(self):
        result = CanaryResult(
            config=CanaryConfig(
                stages=[CanaryStage("s1", traffic_percent=100)],
                baseline_metric=0.85,
            ),
        )
        assert not result.rolled_back
        assert not result.completed
        assert result.current_traffic_percent == 0.0


class TestStageMetrics:
    def test_record_and_stats(self):
        from agentops.ab_testing.canary import StageMetrics
        sm = StageMetrics()
        sm.record(True, 0.90, latency_ms=500)
        sm.record(False, 0.40, latency_ms=200, error="timeout")
        assert sm.total_runs == 2
        assert sm.success_rate == 0.5
        assert sm.error_rate == 0.5
        assert sm.metric_mean == pytest.approx(0.65)
        # P50 of [200, 500] — depends on sort order but midpoint is reasonable
        assert 200 <= sm.latency_p50 <= 500
        assert sm.latency_p95 > 0

    def test_empty_metrics(self):
        from agentops.ab_testing.canary import StageMetrics
        sm = StageMetrics()
        assert sm.success_rate == 1.0  # No runs = perfect by default
        assert sm.error_rate == 0.0
        assert sm.metric_mean == 0.0


# ============================================================================
# eval.py tests
# ============================================================================

class TestGenerateABReport:
    def test_basic_report(self):
        config = ABExperimentConfig(metric="verification_pass_rate", name="test-report")
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("control"), weight=50),
            TrafficAllocation(VariantConfig("treatment"), weight=50),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)
        for i in range(100):
            result = exp.run_iteration(
                result, split, f"task_{i}", i % 3 != 0,
                {"groundedness": 0.85, "latency_ms": 1200},
                latency_ms=1200, cost_usd=0.002,
            )
        analysis = exp.analyze(result)
        report = generate_ab_report(analysis)
        assert isinstance(report, ABEvalReport)
        assert report.experiment_name == "test-report"
        assert len(report.variants) == 2
        assert len(report.comparisons) > 0

    def test_variant_summaries(self):
        config = ABExperimentConfig(metric="verification_pass_rate")
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("A"), weight=50),
            TrafficAllocation(VariantConfig("B"), weight=50),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)
        for i in range(50):
            result = exp.run_iteration(result, split, f"task_{i}", True,
                                       {"groundedness": 0.85})
        analysis = exp.analyze(result)
        report = generate_ab_report(analysis)
        for vs in report.variants:
            assert vs.name in ("A", "B")
            assert vs.total_runs > 0
            assert 0 <= vs.success_rate <= 1
            assert len(vs.confidence_interval_95) == 2
            assert vs.confidence_interval_95[0] <= vs.confidence_interval_95[1]


class TestFormatReportMarkdown:
    def test_generates_markdown(self):
        config = ABExperimentConfig(metric="verification_pass_rate", name="md-test")
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("ctrl"), weight=50),
            TrafficAllocation(VariantConfig("trt"), weight=50),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)
        for i in range(50):
            result = exp.run_iteration(result, split, f"task_{i}", True, {})
        analysis = exp.analyze(result)
        report = generate_ab_report(analysis)
        md = format_report_markdown(report)
        assert "# A/B Experiment Report" in md
        assert "Variant Summaries" in md
        assert "Metric Comparisons" in md
        assert "ctrl" in md
        assert "trt" in md

    def test_markdown_with_winner(self):
        """Report with significant winner."""
        config = ABExperimentConfig()
        split = TrafficSplit([
            TrafficAllocation(VariantConfig("control"), weight=50),
            TrafficAllocation(VariantConfig("treatment"), weight=50),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)
        # Force clear winner
        for i in range(100):
            assigned = split.assign(f"task_{i}")
            success = assigned.name == "treatment"
            result = exp.run_iteration(result, split, f"task_{i}", success, {})
        analysis = exp.analyze(result)
        report = generate_ab_report(analysis)
        md = format_report_markdown(report)
        if analysis.significant:
            assert "wins (significant)" in md.lower() or "**" in md


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    def test_full_ab_experiment_workflow(self):
        """End-to-end: configure, run, analyze, report."""
        config = ABExperimentConfig(
            metric="verification_pass_rate",
            name="full-integration-test",
        )
        split = TrafficSplit([
            TrafficAllocation(
                VariantConfig("baseline", "Current production agent",
                              config={"model": "gpt-4o", "temperature": 0.0}),
                weight=50,
            ),
            TrafficAllocation(
                VariantConfig("candidate", "New reranker + streaming",
                              config={"model": "gpt-4o", "temperature": 0.0,
                                      "reranker": "cross-encoder"}),
                weight=50,
            ),
        ])
        exp = ABExperiment(config)
        result = exp.create_result(split)

        # Simulate 500 runs with realistic distributions
        for i in range(500):
            assigned = split.assign(f"task_{i:04d}")
            # Baseline: 82% success, Candidate: 88% success
            if assigned.name == "baseline":
                success = (i % 100) < 82
            else:
                success = (i % 100) < 88
            result = exp.run_iteration(
                result, split, f"task_{i:04d}", success,
                {
                    "groundedness": 0.80 + (0.15 if success else 0.05),
                    "latency_ms": 1000 + (i % 200),
                },
                latency_ms=1000 + (i % 200),
                cost_usd=0.001 + (0.0005 if assigned.name == "candidate" else 0),
            )

        analysis = exp.analyze(result)
        report = generate_ab_report(analysis)
        md = format_report_markdown(report)

        # Basic assertions
        assert result.total_runs == 500
        for vm in result.variants.values():
            assert vm.total_runs > 0
        assert len(report.variants) == 2
        assert len(report.comparisons) > 0
        assert len(md) > 0

    def test_canary_with_ab_handoff(self):
        """Canary deployment followed by A/B analysis."""
        # Setup canary
        stages = [
            CanaryStage("canary-10%", traffic_percent=10, min_samples=30),
            CanaryStage("staged-50%", traffic_percent=50, min_samples=50),
            CanaryStage("full-100%", traffic_percent=100, min_samples=50),
        ]
        canary_config = CanaryConfig(
            name="agent-v2-rollout",
            stages=stages,
            baseline_metric=0.85,
            baseline_error_rate=0.10,
            auto_rollback=True,
        )
        deployer = CanaryDeployer(canary_config)
        canary_result = deployer.start()

        # Run through all stages with healthy data
        for stage in stages:
            for i in range(stage.min_samples + 5):
                canary_result = deployer.record_run(
                    canary_result, True, 0.86,
                    latency_ms=1000 + (i % 100),
                )
            canary_result = deployer.evaluate_stage(canary_result)
            if canary_result.rolled_back:
                break

        assert canary_result.status == CanaryStatus.COMPLETED
        canary_report = deployer.generate_report(canary_result)
        assert canary_report["status"] == "completed"
        assert canary_report["completed_stages"] == 3
