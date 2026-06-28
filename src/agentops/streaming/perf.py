"""
Streaming Performance Evaluation — latency, throughput, and partial-output quality.

While the existing streaming module (verifier, interceptor, claim_extractor)
focuses on **factuality** (hallucination detection during generation), this
module focuses on **performance**: time-to-first-token, inter-token latency,
tokens-per-second throughput, and partial-output quality assessment.

These metrics matter for production agent systems where user-facing latency
directly impacts experience. Standard evaluation frameworks (LM Eval Harness,
OpenAI Evals, Braintrust) evaluate final answers — not the streaming dynamics
that determine perceived responsiveness.

Modules:
- perf: Data models, timing capture, metric aggregation, benchmark corpus.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Literal


# ── Data Models ────────────────────────────────────────────────────────


@dataclass
class StreamingPerfConfig:
    """Configuration for streaming performance evaluation.

    Attributes:
        warmup_runs: Number of warmup runs before timing (default 1).
        min_output_tokens: Minimum output tokens for valid measurement.
        percentile_thresholds: Percentiles to compute for ITL distribution.
        track_partial_quality: Whether to snapshot partial output at intervals.
        partial_intervals: Output-completion fractions to snapshot (0.0-1.0).
        abort_on_stall: Abort measurement if no token for this many seconds.
    """

    warmup_runs: int = 1
    min_output_tokens: int = 10
    percentile_thresholds: tuple[int, ...] = (50, 90, 95, 99)
    track_partial_quality: bool = True
    partial_intervals: tuple[float, ...] = (0.25, 0.50, 0.75)
    abort_on_stall: float = 30.0


@dataclass
class TokenTiming:
    """Timing data for a single token in a streaming response.

    Attributes:
        token_index: Position in the output sequence (0-based).
        token_text: The token text (truncated to 50 chars for storage).
        elapsed_ms: Milliseconds since generation start.
        delta_ms: Milliseconds since the previous token (inter-token latency).
    """

    token_index: int
    token_text: str
    elapsed_ms: float
    delta_ms: float

    @property
    def is_first(self) -> bool:
        return self.token_index == 0


@dataclass
class StreamingPerfMetrics:
    """Aggregated streaming performance metrics for a single response.

    All timing values are in milliseconds. Throughput in tokens/second.

    Attributes:
        query: The input query text.
        total_tokens: Number of output tokens generated.
        ttft_ms: Time-to-first-token (milliseconds).
        total_time_ms: Total generation time (milliseconds).
        mean_itl_ms: Mean inter-token latency.
        median_itl_ms: Median inter-token latency (P50).
        p90_itl_ms: 90th percentile inter-token latency.
        p95_itl_ms: 95th percentile inter-token latency.
        p99_itl_ms: 99th percentile inter-token latency.
        std_itl_ms: Standard deviation of inter-token latency.
        tokens_per_second: Throughput (tokens/second).
        stall_count: Number of stalls (>500ms between tokens).
        partial_snapshots: Partial outputs at configured intervals.
    """

    query: str = ""
    total_tokens: int = 0
    ttft_ms: float = 0.0
    total_time_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    tokens_per_second: float = 0.0
    stall_count: int = 0
    partial_snapshots: dict[str, str] = field(default_factory=dict)
    token_timings: list[TokenTiming] = field(default_factory=list)
    finished: bool = False
    aborted: bool = False
    abort_reason: str = ""

    @property
    def summary(self) -> dict[str, Any]:
        """Machine-readable summary for dashboards and regression testing."""
        return {
            "query": self.query[:100],
            "total_tokens": self.total_tokens,
            "ttft_ms": round(self.ttft_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "mean_itl_ms": round(self.mean_itl_ms, 1),
            "median_itl_ms": round(self.median_itl_ms, 1),
            "p90_itl_ms": round(self.p90_itl_ms, 1),
            "p95_itl_ms": round(self.p95_itl_ms, 1),
            "p99_itl_ms": round(self.p99_itl_ms, 1),
            "std_itl_ms": round(self.std_itl_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "stall_count": self.stall_count,
            "finished": self.finished,
            "aborted": self.aborted,
        }


@dataclass
class StreamingPerfResult:
    """Complete streaming performance evaluation result.

    Contains per-query metrics and corpus-level aggregates.
    """

    benchmark_name: str
    num_queries: int
    config: StreamingPerfConfig = field(default_factory=StreamingPerfConfig)
    per_query_metrics: list[StreamingPerfMetrics] = field(default_factory=list)

    # Corpus-level aggregates
    mean_ttft_ms: float = 0.0
    mean_total_time_ms: float = 0.0
    mean_itl_ms: float = 0.0
    mean_tps: float = 0.0
    mean_stall_count: float = 0.0
    p90_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    completion_rate: float = 0.0

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "num_queries": self.num_queries,
            "completion_rate": round(self.completion_rate, 3),
            "mean_ttft_ms": round(self.mean_ttft_ms, 1),
            "p90_ttft_ms": round(self.p90_ttft_ms, 1),
            "p95_ttft_ms": round(self.p95_ttft_ms, 1),
            "p99_ttft_ms": round(self.p99_ttft_ms, 1),
            "mean_total_time_ms": round(self.mean_total_time_ms, 1),
            "mean_itl_ms": round(self.mean_itl_ms, 1),
            "mean_tps": round(self.mean_tps, 1),
            "mean_stall_count": round(self.mean_stall_count, 2),
        }


# ── Timing Capture ─────────────────────────────────────────────────────


class StreamingPerfCapture:
    """Captures token-level timing from any streaming text source.

    Wraps any sync generator yielding text chunks and records nanosecond-
    precision timestamps for every token boundary. Handles chunk-to-token
    mapping for sources that emit multi-token chunks.

    Usage:
        capture = StreamingPerfCapture(query="Explain Kubernetes")
        for chunk in capture.wrap(llm_stream):
            print(chunk, end="")
        metrics = capture.finalize()
        print(f"TTFT: {metrics.ttft_ms:.1f}ms")
    """

    def __init__(
        self,
        query: str = "",
        config: StreamingPerfConfig | None = None,
        *,
        tokenizer: Any = None,
    ) -> None:
        self.query = query
        self.config = config or StreamingPerfConfig()
        self._tokenizer = tokenizer

        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._last_token_time: float | None = None
        self._timings: list[TokenTiming] = []
        self._accumulated_text: list[str] = []
        self._token_index: int = 0
        self._stall_count: int = 0
        self._aborted: bool = False
        self._abort_reason: str = ""
        self._finished: bool = False

    def wrap(self, stream: Generator[str, None, None]) -> Generator[str, None, None]:
        """Wrap a sync generator, capturing timing for every yielded chunk.

        Each yielded text piece is treated as one "token" for timing purposes.
        For finer granularity, split chunks into individual tokens before yielding.
        """
        self._start_time = time.perf_counter()

        for chunk in stream:
            now = time.perf_counter()

            if self._first_token_time is None:
                self._first_token_time = now

            delta_ms = 0.0
            if self._last_token_time is not None:
                delta_ms = (now - self._last_token_time) * 1000
                if delta_ms > 500:
                    self._stall_count += 1

            elapsed_ms = (now - self._start_time) * 1000

            self._timings.append(
                TokenTiming(
                    token_index=self._token_index,
                    token_text=chunk[:50],
                    elapsed_ms=elapsed_ms,
                    delta_ms=delta_ms,
                )
            )
            self._accumulated_text.append(chunk)
            self._token_index += 1
            self._last_token_time = now

            # Check stall timeout
            if self._last_token_time and self._first_token_time:
                since_last = (time.perf_counter() - self._last_token_time)
                if since_last > self.config.abort_on_stall:
                    self._aborted = True
                    self._abort_reason = f"Stall timeout ({self.config.abort_on_stall}s)"
                    break

            yield chunk

        self._finished = not self._aborted

    def finalize(self) -> StreamingPerfMetrics:
        """Compute aggregated metrics from captured timings.

        Returns a StreamingPerfMetrics object even if no tokens were captured
        (all values default to 0).
        """
        if not self._timings or self._start_time is None:
            return StreamingPerfMetrics(
                query=self.query,
                finished=False,
                aborted=self._aborted,
                abort_reason=self._abort_reason or "No tokens captured",
            )

        total_time = (time.perf_counter() - self._start_time) * 1000
        ttft = (
            (self._first_token_time - self._start_time) * 1000
            if self._first_token_time
            else 0.0
        )

        itls = [t.delta_ms for t in self._timings if t.token_index > 0]
        if not itls:
            itls = [0.0]

        itls_sorted = sorted(itls)
        n = len(itls_sorted)
        total_tokens = self._token_index

        # Percentiles
        def _percentile(data: list[float], p: int) -> float:
            if not data:
                return 0.0
            if len(data) == 1:
                return data[0]
            k = (p / 100) * (len(data) - 1)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d0 = data[int(f)] * (c - k)
            d1 = data[int(c)] * (k - f)
            return d0 + d1

        tps = (total_tokens / (total_time / 1000)) if total_time > 0 else 0.0

        # Partial snapshots
        snapshots: dict[str, str] = {}
        if self.config.track_partial_quality and total_tokens > 0:
            full_text = "".join(self._accumulated_text)
            for frac in self.config.partial_intervals:
                cutoff = int(total_tokens * frac)
                if cutoff > 0 and cutoff <= total_tokens:
                    snapshots[f"{frac:.2f}"] = full_text[: min(500, cutoff * 10)]

        return StreamingPerfMetrics(
            query=self.query,
            total_tokens=total_tokens,
            ttft_ms=ttft,
            total_time_ms=total_time,
            mean_itl_ms=statistics.mean(itls),
            median_itl_ms=statistics.median(itls),
            p90_itl_ms=_percentile(itls_sorted, 90),
            p95_itl_ms=_percentile(itls_sorted, 95),
            p99_itl_ms=_percentile(itls_sorted, 99),
            std_itl_ms=statistics.stdev(itls) if len(itls) > 1 else 0.0,
            tokens_per_second=tps,
            stall_count=self._stall_count,
            partial_snapshots=snapshots,
            token_timings=self._timings,
            finished=self._finished,
            aborted=self._aborted,
            abort_reason=self._abort_reason,
        )


def simulate_stream(
    output_tokens: list[str],
    delays_ms: list[float] | None = None,
    *,
    jitter_pct: float = 0.0,
) -> Generator[str, None, None]:
    """Simulate a streaming LLM response for testing.

    Args:
        output_tokens: List of text tokens to emit.
        delays_ms: Per-token delay in ms. If None, uses realistic defaults.
        jitter_pct: Random jitter as a fraction of delay (0.0-1.0).

    Yields:
        Text tokens one at a time with realistic delays.
    """
    import random

    if delays_ms is None:
        # Realistic profile: TTFT ~200ms, then 20-50ms per token
        delays_ms = [200.0] + [random.uniform(20, 50) for _ in range(len(output_tokens) - 1)]

    for i, (token, delay) in enumerate(zip(output_tokens, delays_ms)):
        jitter = 1.0
        if jitter_pct > 0:
            jitter = 1.0 + random.uniform(-jitter_pct, jitter_pct)
        time.sleep((delay * jitter) / 1000)
        yield token


# ── Streaming Performance Benchmark Corpus ──────────────────────────────


@dataclass
class StreamingBenchmarkQuery:
    """A single query in the streaming performance benchmark.

    Attributes:
        query: The input prompt.
        category: Response category (short, medium, long, technical).
        expected_min_tokens: Minimum expected output tokens.
        expected_max_tokens: Maximum expected output tokens.
        acceptable_ttft_ms: Maximum acceptable TTFT for this query type.
        acceptable_mean_itl_ms: Maximum acceptable mean ITL.
    """

    query: str
    category: Literal["short", "medium", "long", "technical"]
    expected_min_tokens: int
    expected_max_tokens: int
    acceptable_ttft_ms: float = 500.0
    acceptable_mean_itl_ms: float = 100.0


STREAMING_PERF_BENCHMARK: list[StreamingBenchmarkQuery] = [
    StreamingBenchmarkQuery(
        query="What is Kubernetes? Answer in one sentence.",
        category="short",
        expected_min_tokens=5,
        expected_max_tokens=40,
        acceptable_ttft_ms=300.0,
        acceptable_mean_itl_ms=50.0,
    ),
    StreamingBenchmarkQuery(
        query="List three benefits of using Docker for deployment.",
        category="short",
        expected_min_tokens=10,
        expected_max_tokens=60,
        acceptable_ttft_ms=300.0,
        acceptable_mean_itl_ms=50.0,
    ),
    StreamingBenchmarkQuery(
        query="Explain how a retrieval-augmented generation (RAG) pipeline works.",
        category="medium",
        expected_min_tokens=40,
        expected_max_tokens=200,
        acceptable_ttft_ms=400.0,
        acceptable_mean_itl_ms=60.0,
    ),
    StreamingBenchmarkQuery(
        query="Compare and contrast supervised fine-tuning with RLHF for aligning language models.",
        category="medium",
        expected_min_tokens=50,
        expected_max_tokens=250,
        acceptable_ttft_ms=400.0,
        acceptable_mean_itl_ms=60.0,
    ),
    StreamingBenchmarkQuery(
        query="Describe the process of training a large language model from scratch, including data preparation, architecture decisions, and evaluation.",
        category="long",
        expected_min_tokens=80,
        expected_max_tokens=500,
        acceptable_ttft_ms=500.0,
        acceptable_mean_itl_ms=70.0,
    ),
    StreamingBenchmarkQuery(
        query="Write a comprehensive guide to deploying machine learning models in production, covering CI/CD, monitoring, A/B testing, and rollback strategies.",
        category="long",
        expected_min_tokens=100,
        expected_max_tokens=600,
        acceptable_ttft_ms=500.0,
        acceptable_mean_itl_ms=70.0,
    ),
    StreamingBenchmarkQuery(
        query="Write a Python function that implements a thread-safe LRU cache with TTL expiration.",
        category="technical",
        expected_min_tokens=60,
        expected_max_tokens=300,
        acceptable_ttft_ms=400.0,
        acceptable_mean_itl_ms=50.0,
    ),
    StreamingBenchmarkQuery(
        query="Design a distributed rate limiter using Redis that supports sliding window, token bucket, and fixed window algorithms.",
        category="technical",
        expected_min_tokens=70,
        expected_max_tokens=350,
        acceptable_ttft_ms=400.0,
        acceptable_mean_itl_ms=50.0,
    ),
    StreamingBenchmarkQuery(
        query="What are the key differences between gRPC and REST for microservice communication?",
        category="medium",
        expected_min_tokens=40,
        expected_max_tokens=200,
        acceptable_ttft_ms=400.0,
        acceptable_mean_itl_ms=60.0,
    ),
    StreamingBenchmarkQuery(
        query="Explain the CAP theorem and its implications for distributed database design.",
        category="medium",
        expected_min_tokens=50,
        expected_max_tokens=250,
        acceptable_ttft_ms=400.0,
        acceptable_mean_itl_ms=60.0,
    ),
]


class StreamingPerfEvaluator:
    """Evaluate streaming performance against a benchmark corpus.

    Runs a streaming generator function against each benchmark query,
    captures timing metrics, and produces a StreamingPerfResult with
    corpus-level aggregates.

    Works with both real LLM streams and simulated streams for CI testing.

    Usage:
        def my_stream(prompt: str) -> Generator[str, None, None]:
            for token in llm.generate_stream(prompt):
                yield token

        evaluator = StreamingPerfEvaluator()
        result = evaluator.run_benchmark(my_stream)
        print(result.summary)
    """

    def __init__(self, config: StreamingPerfConfig | None = None) -> None:
        self.config = config or StreamingPerfConfig()

    def run_benchmark(
        self,
        stream_fn,
        queries: list[StreamingBenchmarkQuery] | None = None,
        *,
        benchmark_name: str = "streaming_perf",
    ) -> StreamingPerfResult:
        """Run streaming performance benchmark.

        Args:
            stream_fn: Callable(prompt: str) -> Generator[str]. Must accept
                       a prompt string and yield text tokens.
            queries: Benchmark queries. Uses STREAMING_PERF_BENCHMARK if None.
            benchmark_name: Label for the result.

        Returns:
            StreamingPerfResult with per-query and aggregate metrics.
        """
        queries = queries if queries is not None else STREAMING_PERF_BENCHMARK
        per_query: list[StreamingPerfMetrics] = []

        # Warmup
        if self.config.warmup_runs > 0 and queries:
            warmup_query = queries[0].query
            for _ in range(self.config.warmup_runs):
                capture = StreamingPerfCapture(
                    query=f"warmup:{warmup_query[:50]}",
                    config=self.config,
                )
                for _ in capture.wrap(stream_fn(warmup_query)):
                    pass
                capture.finalize()

        # Benchmark
        for bq in queries:
            capture = StreamingPerfCapture(query=bq.query, config=self.config)
            for _ in capture.wrap(stream_fn(bq.query)):
                pass
            metrics = capture.finalize()
            per_query.append(metrics)

        return self._aggregate(queries, per_query, benchmark_name)

    def run_simulated_benchmark(
        self,
        queries: list[StreamingBenchmarkQuery] | None = None,
        *,
        benchmark_name: str = "streaming_perf_simulated",
        ttft_ms: float = 200.0,
        mean_itl_ms: float = 35.0,
        jitter_pct: float = 0.2,
    ) -> StreamingPerfResult:
        """Run benchmark with simulated streaming (for CI/testing).

        Generates plausible-looking responses at configured latency profiles.

        Args:
            queries: Benchmark queries.
            benchmark_name: Label for the result.
            ttft_ms: Simulated time-to-first-token.
            mean_itl_ms: Simulated mean inter-token latency.
            jitter_pct: Random jitter fraction.

        Returns:
            StreamingPerfResult with simulated metrics.
        """
        import random

        def _sim_stream(prompt: str) -> Generator[str, None, None]:
            # Generate plausible token count based on query length
            target_tokens = max(10, min(500, len(prompt.split()) * random.randint(4, 12)))
            tokens = [f"token_{i:04d}" for i in range(target_tokens)]

            # TTFT
            time.sleep((ttft_ms * random.uniform(0.8, 1.2)) / 1000)

            for i, token in enumerate(tokens):
                if i == 0:
                    pass  # TTFT already waited
                else:
                    delay = mean_itl_ms * random.uniform(0.5, 2.0)
                    if jitter_pct:
                        delay *= random.uniform(1 - jitter_pct, 1 + jitter_pct)
                    time.sleep(delay / 1000)
                yield token

        return self.run_benchmark(
            _sim_stream,
            queries=queries,
            benchmark_name=benchmark_name,
        )

    def _aggregate(
        self,
        queries: list[StreamingBenchmarkQuery],
        per_query: list[StreamingPerfMetrics],
        benchmark_name: str,
    ) -> StreamingPerfResult:
        """Compute corpus-level aggregates from per-query metrics."""
        n = len(per_query) or 1

        ttfts = [m.ttft_ms for m in per_query]
        total_times = [m.total_time_ms for m in per_query]
        itls = [m.mean_itl_ms for m in per_query]
        tps_vals = [m.tokens_per_second for m in per_query]
        stalls = [m.stall_count for m in per_query]
        completed = sum(1 for m in per_query if m.finished)

        def _percentile_safe(data: list[float], p: int) -> float:
            if not data:
                return 0.0
            s = sorted(data)
            if len(s) == 1:
                return s[0]
            k = (p / 100) * (len(s) - 1)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return s[int(k)]
            return s[int(f)] * (c - k) + s[int(c)] * (k - f)

        return StreamingPerfResult(
            benchmark_name=benchmark_name,
            num_queries=len(queries),
            config=self.config,
            per_query_metrics=per_query,
            mean_ttft_ms=statistics.mean(ttfts) if ttfts else 0.0,
            mean_total_time_ms=statistics.mean(total_times) if total_times else 0.0,
            mean_itl_ms=statistics.mean(itls) if itls else 0.0,
            mean_tps=statistics.mean(tps_vals) if tps_vals else 0.0,
            mean_stall_count=statistics.mean(stalls) if stalls else 0.0,
            p90_ttft_ms=_percentile_safe(ttfts, 90),
            p95_ttft_ms=_percentile_safe(ttfts, 95),
            p99_ttft_ms=_percentile_safe(ttfts, 99),
            completion_rate=completed / n,
        )


# ── Regression Testing Helpers ─────────────────────────────────────────


class StreamingPerfRegression:
    """Compare streaming performance metrics against baselines.

    Detects performance regressions by comparing current metrics against
    stored baseline values with configurable tolerance thresholds.

    Usage:
        baseline = {"ttft_ms": 200, "p95_itl_ms": 80, "tps": 25}
        reg = StreamingPerfRegression(baseline)
        result = reg.check(current_metrics)
        if result.regressed:
            print(f"REGRESSION: {result.violations}")
    """

    def __init__(
        self,
        baseline: dict[str, float],
        *,
        tolerance_pct: float = 20.0,
        thresholds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.baseline = baseline
        self.tolerance_pct = tolerance_pct
        self.thresholds = thresholds or {
            "ttft_ms": (0, 500),
            "p95_itl_ms": (0, 150),
            "tokens_per_second": (10, 1000),
        }

    def check(self, current: StreamingPerfMetrics) -> StreamingPerfRegressionResult:
        """Check current metrics against baseline for regressions."""
        violations: list[dict[str, Any]] = []

        checks = {
            "ttft_ms": current.ttft_ms,
            "mean_itl_ms": current.mean_itl_ms,
            "p95_itl_ms": current.p95_itl_ms,
            "tokens_per_second": current.tokens_per_second,
        }

        for key, value in checks.items():
            if key not in self.baseline:
                continue

            baseline_val = self.baseline[key]
            if baseline_val == 0:
                continue

            deviation_pct = abs(value - baseline_val) / baseline_val * 100

            # For latency metrics (ttft, itl), lower is better — only flag if SLOWER
            # For throughput (tps), higher is better — only flag if LOWER
            is_regression = False
            if key == "tokens_per_second":
                # Regression = throughput dropped more than tolerance
                if value < baseline_val:
                    is_regression = deviation_pct > self.tolerance_pct
            else:
                # Regression = latency increased more than tolerance
                if value > baseline_val:
                    is_regression = deviation_pct > self.tolerance_pct

            # Check absolute thresholds
            within_threshold = True
            if key in self.thresholds:
                lo, hi = self.thresholds[key]
                within_threshold = lo <= value <= hi

            violations.append({
                "metric": key,
                "baseline": baseline_val,
                "current": round(value, 1),
                "deviation_pct": round(deviation_pct, 1),
                "regressed": is_regression,
                "within_threshold": within_threshold,
            })

        regressed = any(v["regressed"] for v in violations)
        out_of_threshold = any(not v["within_threshold"] for v in violations)

        return StreamingPerfRegressionResult(
            baseline_name="streaming_perf_baseline",
            violations=violations,
            regressed=regressed or out_of_threshold,
            num_violations=sum(1 for v in violations if v["regressed"]),
            num_out_of_threshold=sum(1 for v in violations if not v["within_threshold"]),
        )


@dataclass
class StreamingPerfRegressionResult:
    """Result of a streaming performance regression check."""

    baseline_name: str
    violations: list[dict[str, Any]]
    regressed: bool
    num_violations: int
    num_out_of_threshold: int

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline_name,
            "regressed": self.regressed,
            "num_violations": self.num_violations,
            "num_out_of_threshold": self.num_out_of_threshold,
            "violations": self.violations,
        }
