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
from .judge.state import (
    JudgeConfig,
    JudgeDimension,
    JudgeResult,
    JudgeVerdict,
    JudgeRubric,
    JudgeBenchmarkResult,
    DEFAULT_RUBRICS,
)
from .judge.judge import LLMJudge, SimulatedJudge, JudgeRunner
from .model_benchmark import (
    ModelProfile,
    ModelComparisonResult,
    MultiModelReport,
    ModelBenchmark,
    MODEL_PROFILES,
)

__all__ = [
    # Metrics
    "ReliabilityMetrics",
    "compute_metrics",
    # Benchmarks
    "ReliabilityBenchmark",
    "ALL_BENCHMARKS",
    "get_benchmark",
    # Harness
    "EvalHarness",
    "EvalReport",
    # Judge
    "JudgeConfig",
    "JudgeDimension",
    "JudgeResult",
    "JudgeVerdict",
    "JudgeRubric",
    "JudgeBenchmarkResult",
    "DEFAULT_RUBRICS",
    "LLMJudge",
    "SimulatedJudge",
    "JudgeRunner",
    # Model Benchmark
    "ModelProfile",
    "ModelComparisonResult",
    "MultiModelReport",
    "ModelBenchmark",
    "MODEL_PROFILES",
]
