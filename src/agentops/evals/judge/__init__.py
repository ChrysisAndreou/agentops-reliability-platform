"""
LLM-as-Judge evaluation framework.

Provides deterministic (simulated) and real-LLM judge implementations
for evaluating agent output quality across configurable dimensions.
"""

from .state import (
    JudgeConfig,
    JudgeDimension,
    JudgeResult,
    JudgeVerdict,
    JudgeRubric,
    JudgeBenchmarkResult,
    DEFAULT_RUBRICS,
)
from .judge import (
    LLMJudge,
    SimulatedJudge,
    JudgeRunner,
)

__all__ = [
    # State models
    "JudgeConfig",
    "JudgeDimension",
    "JudgeResult",
    "JudgeVerdict",
    "JudgeRubric",
    "JudgeBenchmarkResult",
    "DEFAULT_RUBRICS",
    # Judge implementations
    "LLMJudge",
    "SimulatedJudge",
    "JudgeRunner",
]
