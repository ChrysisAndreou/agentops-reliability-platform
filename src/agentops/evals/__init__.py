"""
Evaluation framework for the AgentOps Reliability Platform.

Provides reliability-focused metrics (groundedness, citation quality,
tool correctness, verification pass rate, latency), benchmark tasks,
and an evaluation harness that runs agents on tasks and produces
reports with failure analysis.
"""

from .metrics import ReliabilityMetrics, compute_metrics
from .benchmarks import ReliabilityBenchmark, ALL_BENCHMARKS, get_benchmark
from .harness import EvalHarness, EvalReport

__all__ = [
    "ReliabilityMetrics",
    "compute_metrics",
    "ReliabilityBenchmark",
    "ALL_BENCHMARKS",
    "get_benchmark",
    "EvalHarness",
    "EvalReport",
]
