"""Tests for streaming performance evaluation module."""

import pytest
import time

from agentops.streaming.perf import (
    STREAMING_PERF_BENCHMARK,
    StreamingBenchmarkQuery,
    StreamingPerfCapture,
    StreamingPerfConfig,
    StreamingPerfEvaluator,
    StreamingPerfMetrics,
    StreamingPerfRegression,
    StreamingPerfRegressionResult,
    StreamingPerfResult,
    TokenTiming,
    simulate_stream,
)


# ── TokenTiming ─────────────────────────────────────────────────────────


class TestTokenTiming:
    def test_create(self):
        t = TokenTiming(token_index=0, token_text="Hello", elapsed_ms=100.0, delta_ms=0.0)
        assert t.token_index == 0
        assert t.token_text == "Hello"
        assert t.elapsed_ms == 100.0
        assert t.delta_ms == 0.0
        assert t.is_first is True

    def test_is_first_false(self):
        t = TokenTiming(token_index=5, token_text="world", elapsed_ms=250.0, delta_ms=30.0)
        assert t.is_first is False

    def test_token_text_truncation(self):
        long_text = "x" * 100
        t = TokenTiming(token_index=0, token_text=long_text, elapsed_ms=0, delta_ms=0)
        # TokenTiming stores text as-is; truncation is advisory in the docstring
        assert len(t.token_text) >= 50
        assert t.token_text == long_text


# ── StreamingPerfConfig ─────────────────────────────────────────────────


class TestStreamingPerfConfig:
    def test_defaults(self):
        cfg = StreamingPerfConfig()
        assert cfg.warmup_runs == 1
        assert cfg.min_output_tokens == 10
        assert cfg.percentile_thresholds == (50, 90, 95, 99)
        assert cfg.track_partial_quality is True
        assert cfg.partial_intervals == (0.25, 0.50, 0.75)
        assert cfg.abort_on_stall == 30.0

    def test_custom(self):
        cfg = StreamingPerfConfig(
            warmup_runs=3,
            min_output_tokens=20,
            abort_on_stall=10.0,
            track_partial_quality=False,
        )
        assert cfg.warmup_runs == 3
        assert cfg.min_output_tokens == 20
        assert cfg.abort_on_stall == 10.0


# ── StreamingPerfCapture ────────────────────────────────────────────────


class TestStreamingPerfCaptureBasic:
    def test_create(self):
        capture = StreamingPerfCapture(query="test query")
        assert capture.query == "test query"
        assert capture._start_time is None
        assert capture._token_index == 0
        assert len(capture._timings) == 0

    def test_wrap_records_timings(self):
        tokens = ["The", " quick", " brown", " fox"]
        delays = [100.0, 30.0, 30.0, 30.0]

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        output = list(capture.wrap(stream))

        assert output == tokens
        assert len(capture._timings) == 4
        assert capture._token_index == 4
        assert capture._finished is True
        assert capture._aborted is False

    def test_wrap_records_first_token(self):
        tokens = ["Hello", " world"]
        delays = [100.0, 30.0]

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        list(capture.wrap(stream))

        first = capture._timings[0]
        assert first.token_index == 0
        assert first.is_first is True

    def test_wrap_empty_stream(self):
        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream([], [])
        output = list(capture.wrap(stream))

        assert output == []
        assert capture._token_index == 0
        assert len(capture._timings) == 0


class TestStreamingPerfCaptureFinalize:
    def test_computes_ttft(self):
        tokens = ["A"] * 20
        delays = [200.0] + [30.0] * 19

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        # TTFT should be approximately 200ms
        assert 150 < metrics.ttft_ms < 250
        assert metrics.total_tokens == 20
        assert metrics.finished is True

    def test_computes_itl_percentiles(self):
        tokens = ["A"] * 10
        delays = [100.0] + [40.0] * 9

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        # All ITLs are 40ms, so percentiles should be ~40
        assert 35 < metrics.median_itl_ms < 45
        assert 35 < metrics.p90_itl_ms < 45
        assert metrics.total_tokens == 10

    def test_computes_tokens_per_second(self):
        tokens = ["A"] * 50
        delays = [100.0] + [20.0] * 49

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        # 50 tokens in ~100 + 49*20 = ~1080ms → ~46 tps
        assert metrics.tokens_per_second > 30
        assert metrics.tokens_per_second < 60

    def test_stall_detection(self):
        # Create a stream with a 600ms gap between tokens
        tokens = ["A", "B", "C"]
        delays = [100.0, 550.0, 20.0]  # 550ms > 500ms stall threshold

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        assert metrics.stall_count >= 1

    def test_partial_snapshots(self):
        tokens = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", " today"]
        delays = [50.0] * 10

        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(tokens, delays, jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        # Should have snapshots at 0.25, 0.50, 0.75
        assert len(metrics.partial_snapshots) > 0
        assert "0.25" in metrics.partial_snapshots or "0.50" in metrics.partial_snapshots

    def test_single_token(self):
        capture = StreamingPerfCapture(query="test")
        stream = simulate_stream(["hello"], [100.0], jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        assert metrics.total_tokens == 1
        assert metrics.finished is True
        # ITLs may be 0 since there's only one token

    def test_summary_property(self):
        capture = StreamingPerfCapture(query="test query")
        stream = simulate_stream(["A"] * 10, [100.0] + [30.0] * 9, jitter_pct=0.0)
        list(capture.wrap(stream))

        metrics = capture.finalize()
        summary = metrics.summary

        assert isinstance(summary, dict)
        assert "ttft_ms" in summary
        assert "total_tokens" in summary
        assert "tokens_per_second" in summary
        assert "stall_count" in summary
        assert "finished" in summary


# ── simulate_stream ─────────────────────────────────────────────────────


class TestSimulateStream:
    def test_basic(self):
        tokens = ["a", "b", "c"]
        result = list(simulate_stream(tokens, delays_ms=[10.0, 5.0, 5.0], jitter_pct=0.0))
        assert result == tokens

    def test_default_delays(self):
        tokens = ["a", "b", "c"]
        result = list(simulate_stream(tokens, jitter_pct=0.0))
        assert result == tokens

    def test_jitter(self):
        tokens = ["a"] * 5
        # With jitter, timing varies but output is same
        result = list(simulate_stream(tokens, jitter_pct=0.5))
        assert result == tokens

    def test_empty(self):
        result = list(simulate_stream([], jitter_pct=0.0))
        assert result == []


# ── StreamingPerfMetrics ────────────────────────────────────────────────


class TestStreamingPerfMetrics:
    def test_defaults(self):
        m = StreamingPerfMetrics()
        assert m.total_tokens == 0
        assert m.ttft_ms == 0.0
        assert m.finished is False

    def test_summary_rounds_values(self):
        m = StreamingPerfMetrics(
            query="test",
            total_tokens=10,
            ttft_ms=123.456,
            total_time_ms=500.0,
            mean_itl_ms=45.678,
            p95_itl_ms=80.0,
            tokens_per_second=20.123,
            finished=True,
        )
        s = m.summary
        assert s["ttft_ms"] == 123.5  # rounded to 1 decimal


# ── StreamingPerfResult ─────────────────────────────────────────────────


class TestStreamingPerfResult:
    def test_summary(self):
        result = StreamingPerfResult(
            benchmark_name="test_bench",
            num_queries=3,
            mean_ttft_ms=200.0,
            mean_total_time_ms=1500.0,
            completion_rate=1.0,
        )
        s = result.summary
        assert s["benchmark"] == "test_bench"
        assert s["num_queries"] == 3
        assert s["completion_rate"] == 1.0


# ── StreamingPerfEvaluator ──────────────────────────────────────────────


class TestStreamingPerfEvaluator:
    def test_run_simulated_benchmark(self):
        evaluator = StreamingPerfEvaluator(
            StreamingPerfConfig(warmup_runs=0)
        )
        result = evaluator.run_simulated_benchmark(
            queries=STREAMING_PERF_BENCHMARK[:3],
            ttft_ms=200.0,
            mean_itl_ms=35.0,
            jitter_pct=0.0,
        )

        assert result.benchmark_name == "streaming_perf_simulated"
        assert result.num_queries == 3
        assert result.completion_rate == 1.0
        assert len(result.per_query_metrics) == 3
        # TTFT should be ~200ms
        assert 150 < result.mean_ttft_ms < 250

    def test_run_simulated_with_warmup(self):
        evaluator = StreamingPerfEvaluator(
            StreamingPerfConfig(warmup_runs=2)
        )
        result = evaluator.run_simulated_benchmark(
            queries=STREAMING_PERF_BENCHMARK[:2],
            ttft_ms=100.0,
            mean_itl_ms=20.0,
            jitter_pct=0.0,
        )
        assert result.num_queries == 2
        assert result.completion_rate == 1.0

    def test_all_benchmark_queries(self):
        """Full benchmark — all 10 queries (reduced tokens for speed)."""
        evaluator = StreamingPerfEvaluator(
            StreamingPerfConfig(warmup_runs=0)
        )
        result = evaluator.run_simulated_benchmark(
            queries=STREAMING_PERF_BENCHMARK,
            ttft_ms=10.0,
            mean_itl_ms=2.0,
            jitter_pct=0.0,
        )
        assert result.num_queries == 10
        assert result.completion_rate == 1.0
        assert result.mean_tps > 0
        assert len(result.per_query_metrics) == 10

    def test_categories_in_benchmark(self):
        """All four categories present."""
        cats = {q.category for q in STREAMING_PERF_BENCHMARK}
        assert cats == {"short", "medium", "long", "technical"}

    def test_acceptable_limits_in_queries(self):
        """Each query has reasonable acceptable limits."""
        for q in STREAMING_PERF_BENCHMARK:
            assert q.expected_min_tokens > 0
            assert q.expected_max_tokens >= q.expected_min_tokens
            assert q.acceptable_ttft_ms > 0
            assert q.acceptable_mean_itl_ms > 0


# ── StreamingPerfRegression ─────────────────────────────────────────────


class TestStreamingPerfRegression:
    @pytest.fixture
    def baseline(self):
        return {"ttft_ms": 200.0, "mean_itl_ms": 40.0, "p95_itl_ms": 80.0, "tokens_per_second": 30.0}

    def test_no_regression(self, baseline):
        reg = StreamingPerfRegression(baseline, tolerance_pct=20.0)
        metrics = StreamingPerfMetrics(
            ttft_ms=210.0,  # +5%
            mean_itl_ms=42.0,  # +5%
            p95_itl_ms=85.0,  # +6.25%
            tokens_per_second=31.0,  # +3.3% (better)
        )
        result = reg.check(metrics)
        assert result.regressed is False
        assert result.num_violations == 0

    def test_regression_detected(self, baseline):
        reg = StreamingPerfRegression(baseline, tolerance_pct=10.0)
        metrics = StreamingPerfMetrics(
            ttft_ms=300.0,  # +50% — regression
            mean_itl_ms=40.0,
            p95_itl_ms=80.0,
            tokens_per_second=30.0,
        )
        result = reg.check(metrics)
        assert result.regressed is True
        assert result.num_violations >= 1

    def test_better_performance_not_regression(self, baseline):
        reg = StreamingPerfRegression(baseline, tolerance_pct=10.0)
        metrics = StreamingPerfMetrics(
            ttft_ms=100.0,  # -50% — better, not regression
            mean_itl_ms=20.0,  # -50% — better
            p95_itl_ms=40.0,  # -50% — better
            tokens_per_second=60.0,  # +100% — better, not regression
        )
        result = reg.check(metrics)
        # Higher TPS is not a regression; lower TTFT/ITL is better
        assert result.num_violations == 0

    def test_threshold_violation(self, baseline):
        thresholds = {"ttft_ms": (0, 400), "p95_itl_ms": (0, 100)}
        reg = StreamingPerfRegression(baseline, tolerance_pct=50.0, thresholds=thresholds)
        metrics = StreamingPerfMetrics(
            ttft_ms=500.0,  # Above 400ms threshold
            mean_itl_ms=40.0,
            p95_itl_ms=80.0,
            tokens_per_second=30.0,
        )
        result = reg.check(metrics)
        assert result.num_out_of_threshold >= 1

    def test_summary_property(self, baseline):
        reg = StreamingPerfRegression(baseline)
        metrics = StreamingPerfMetrics(
            ttft_ms=200.0, mean_itl_ms=40.0, p95_itl_ms=80.0, tokens_per_second=30.0,
        )
        result = reg.check(metrics)
        s = result.summary
        assert "regressed" in s
        assert "violations" in s


class TestStreamingPerfRegressionResult:
    def test_create(self):
        result = StreamingPerfRegressionResult(
            baseline_name="v1",
            violations=[],
            regressed=False,
            num_violations=0,
            num_out_of_threshold=0,
        )
        assert result.baseline_name == "v1"
        assert result.regressed is False

    def test_with_violations(self):
        result = StreamingPerfRegressionResult(
            baseline_name="v1",
            violations=[{"metric": "ttft_ms", "regressed": True}],
            regressed=True,
            num_violations=1,
            num_out_of_threshold=0,
        )
        assert result.regressed is True
        assert result.num_violations == 1


# ── StreamingPerfResult Aggregate Edge Cases ────────────────────────────


class TestPerfResultEdgeCases:
    def test_empty_queries(self):
        evaluator = StreamingPerfEvaluator(
            StreamingPerfConfig(warmup_runs=0)
        )
        result = evaluator.run_simulated_benchmark(queries=[], ttft_ms=200.0, mean_itl_ms=35.0)
        assert result.num_queries == 0

    def test_single_query(self):
        evaluator = StreamingPerfEvaluator(
            StreamingPerfConfig(warmup_runs=0)
        )
        result = evaluator.run_simulated_benchmark(
            queries=STREAMING_PERF_BENCHMARK[:1],
            ttft_ms=100.0,
            mean_itl_ms=20.0,
        )
        assert result.num_queries == 1
        assert len(result.per_query_metrics) == 1
