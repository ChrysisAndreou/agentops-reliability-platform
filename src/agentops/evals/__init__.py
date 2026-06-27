"""
Evaluation framework for the AgentOps Reliability Platform.

Provides reliability-focused metrics (groundedness, citation quality,
tool correctness, verification pass rate, latency), benchmark tasks,
and an evaluation harness that runs agents on tasks and produces
reports with failure analysis.
"""

from .benchmarks import ALL_BENCHMARKS, ReliabilityBenchmark, get_benchmark
from .harness import EvalHarness, EvalReport
from .judge.judge import JudgeRunner, LLMJudge, SimulatedJudge
from .judge.state import (
    DEFAULT_RUBRICS,
    JudgeBenchmarkResult,
    JudgeConfig,
    JudgeDimension,
    JudgeResult,
    JudgeRubric,
    JudgeVerdict,
)
from .metrics import ReliabilityMetrics, compute_metrics
from .model_benchmark import (
    MODEL_PROFILES,
    ModelBenchmark,
    ModelComparisonResult,
    ModelProfile,
    MultiModelReport,
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
