"""
Tests for expanded benchmarks, comparator, and budget gates.
"""

import asyncio

import pytest

from agentops.evals.benchmarks import (
    ALL_BENCHMARKS,
    BENCHMARK_MAP,
    get_benchmark,
    list_benchmarks,
    TOOL_USE_BENCH,
    MULTI_STEP_BENCH,
    EDGE_CASE_BENCH,
    HALLUCINATION_BENCH,
)
from agentops.evals.comparator import (
    EvalComparator,
    ComparisonResult,
    RegressionReport,
    run_multi_profile_eval,
)
from agentops.evals.simulator import (
    PRODUCTION_AGENT,
    DEVELOPMENT_AGENT,
    UNRELIABLE_AGENT,
    SimulatedAgent,
)
from agentops.evals.metrics import compute_metrics
from agentops.evals.budget import (
    CostBudget,
    LatencyBudget,
    BudgetState,
    BudgetGate,
    BudgetGateResult,
    BudgetExceededError,
    NORMAL_BUDGET,
    STRICT_BUDGET,
    FAST_LATENCY,
    NORMAL_LATENCY,
)


class TestExpandedBenchmarks:
    def test_six_benchmarks_registered(self):
        assert len(ALL_BENCHMARKS) >= 6

    def test_benchmark_map_has_all(self):
        assert len(BENCHMARK_MAP) >= 6
        for b in ALL_BENCHMARKS:
            assert b.name in BENCHMARK_MAP

    def test_get_benchmark_finds_all(self):
        for b in ALL_BENCHMARKS:
            found = get_benchmark(b.name)
            assert found is not None
            assert found.name == b.name

    def test_get_benchmark_returns_none_for_unknown(self):
        assert get_benchmark("nonexistent") is None

    def test_list_benchmarks_returns_metadata(self):
        blist = list_benchmarks()
        assert len(blist) >= 6
        for entry in blist:
            assert "name" in entry
            assert "task_count" in entry
            assert "categories" in entry
            assert entry["task_count"] > 0

    def test_tool_use_benchmark_has_tool_tasks(self):
        tool_tasks = [t for t in TOOL_USE_BENCH.tasks if t.requires_tool]
        assert len(tool_tasks) >= 3

    def test_hallucination_benchmark_has_empty_expected_sources(self):
        # At least some tasks should have no expected sources (hallucination test)
        empty_source_tasks = [t for t in HALLUCINATION_BENCH.tasks if not t.expected_sources]
        assert len(empty_source_tasks) >= 3

    def test_multi_step_benchmark_has_hard_tasks(self):
        hard_tasks = [t for t in MULTI_STEP_BENCH.tasks if t.difficulty == "hard"]
        assert len(hard_tasks) >= 3

    def test_all_tasks_have_unique_ids(self):
        all_ids = []
        for bench in ALL_BENCHMARKS:
            for task in bench.tasks:
                all_ids.append(task.id)
        assert len(all_ids) == len(set(all_ids)), f"Duplicate task IDs found"

    def test_total_task_count(self):
        total = sum(len(b.tasks) for b in ALL_BENCHMARKS)
        assert total >= 30  # 7 benchmarks × 5 tasks = 35


class TestComparator:
    @pytest.mark.asyncio
    async def test_compare_production_vs_development(self):
        comparator = EvalComparator()
        result = await comparator.compare(
            TOOL_USE_BENCH,
            config_a=PRODUCTION_AGENT,
            config_b=DEVELOPMENT_AGENT,
        )
        assert isinstance(result, ComparisonResult)
        assert result.benchmark_name == "tool-use"
        assert result.config_a == "production"
        assert result.config_b == "development"
        assert len(result.deltas) > 0
        assert len(result.metrics_a) == 5
        assert len(result.metrics_b) == 5

    @pytest.mark.asyncio
    async def test_production_beats_unreliable(self):
        comparator = EvalComparator()
        result = await comparator.compare(
            MULTI_STEP_BENCH,
            config_a=PRODUCTION_AGENT,
            config_b=UNRELIABLE_AGENT,
        )
        # Production should win against unreliable
        assert result.winner == "a"

    @pytest.mark.asyncio
    async def test_comparison_report_markdown(self):
        comparator = EvalComparator()
        result = await comparator.compare(
            TOOL_USE_BENCH,
            config_a=PRODUCTION_AGENT,
            config_b=DEVELOPMENT_AGENT,
        )
        md = result.to_markdown()
        assert "A/B Comparison" in md
        assert "production" in md
        assert "development" in md

    def test_detect_regression(self):
        metrics_baseline = []
        metrics_current = []

        sim = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        sim_bad = SimulatedAgent(config=UNRELIABLE_AGENT, seed=42)

        for task in TOOL_USE_BENCH.tasks:
            r = asyncio.run(sim.run(task.question, task_id=task.id))
            metrics_baseline.append(compute_metrics(r, key_terms=task.key_terms))
            r_bad = asyncio.run(sim_bad.run(task.question, task_id=task.id))
            metrics_current.append(compute_metrics(r_bad, key_terms=task.key_terms))

        comparator = EvalComparator()
        report = comparator.detect_regression(
            metrics_baseline, metrics_current,
            baseline_name="production", current_name="unreliable",
            benchmark_name="test-regression",
        )
        assert isinstance(report, RegressionReport)
        assert report.has_regressions  # Unreliable should regress vs production
        assert len(report.regressions) > 0

    def test_no_regression_when_identical(self):
        from agentops.evals.simulator import PERFECT_AGENT

        metrics = []
        sim = SimulatedAgent(config=PERFECT_AGENT, seed=42)
        for task in TOOL_USE_BENCH.tasks:
            r = asyncio.run(sim.run(task.question, task_id=task.id))
            metrics.append(compute_metrics(r, key_terms=task.key_terms))

        comparator = EvalComparator()
        report = comparator.detect_regression(metrics, metrics)
        assert not report.has_regressions

    @pytest.mark.asyncio
    async def test_multi_profile_eval(self):
        from agentops.evals.simulator import PERFECT_AGENT, PRODUCTION_AGENT

        results = await run_multi_profile_eval(
            TOOL_USE_BENCH,
            profiles=[PERFECT_AGENT, PRODUCTION_AGENT],
            seed=42,
        )
        assert "perfect" in results
        assert "production" in results
        assert len(results["perfect"]) == 5
        assert len(results["production"]) == 5


class TestBudgetGates:
    def test_cost_budget_estimation(self):
        budget = CostBudget()
        cost = budget.estimate_cost(1000, 500)
        expected = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.010
        assert abs(cost - expected) < 0.0001

    def test_budget_state_tracks_usage(self):
        state = BudgetState()
        state.start()
        state.record_step(input_tokens=500, output_tokens=200, latency_ms=5000)
        assert state.step_count == 1
        assert state.total_input_tokens == 500
        assert state.total_output_tokens == 200
        assert state.total_estimated_cost_usd > 0

    def test_budget_state_elapsed_time(self):
        state = BudgetState()
        state.start()
        elapsed = state.elapsed_ms
        assert elapsed >= 0

    def test_budget_gate_allows_normal_usage(self):
        state = BudgetState()
        state.start()
        state.record_step(input_tokens=100, output_tokens=50, latency_ms=1000)

        gate = BudgetGate()
        result = gate.check(state)
        assert result.allowed

    def test_budget_gate_blocks_cost_exceeded(self):
        budget = CostBudget(max_total_cost_usd=0.001)
        state = BudgetState(cost_budget=budget)
        state.start()
        # Record enough tokens to exceed the tiny budget
        state.record_step(input_tokens=10000, output_tokens=5000, latency_ms=1000)

        gate = BudgetGate(cost_budget=budget)
        result = gate.check(state)
        assert not result.allowed

    def test_strict_budget_is_stricter(self):
        assert STRICT_BUDGET.max_total_cost_usd < NORMAL_BUDGET.max_total_cost_usd

    def test_latency_budget_tracks_warnings(self):
        latency = LatencyBudget(max_per_step_latency_ms=1000, warn_at_latency_ms=1)
        state = BudgetState(latency_budget=latency)
        state.start()
        state.record_step(input_tokens=100, output_tokens=50, latency_ms=5000)

        gate = BudgetGate(latency_budget=latency)
        result = gate.check(state)
        # Should have latency warnings (5s step > 1s per-step budget)
        assert len(result.budget_state.warnings) >= 1

    def test_budget_exceeded_error(self):
        err = BudgetExceededError("Cost limit exceeded")
        assert "Cost limit exceeded" in str(err)

    def test_budget_gate_result_to_dict(self):
        state = BudgetState()
        state.start()
        gate = BudgetGate()
        result = gate.check(state)
        d = result.to_dict()
        assert d["allowed"] is True
        assert "budget" in d

    def test_should_abort_quick_check(self):
        budget = CostBudget(max_total_cost_usd=0.001)
        state = BudgetState(cost_budget=budget)
        state.start()
        state.record_step(input_tokens=10000, output_tokens=5000, latency_ms=100)

        gate = BudgetGate(cost_budget=budget)
        assert gate.should_abort(state)
