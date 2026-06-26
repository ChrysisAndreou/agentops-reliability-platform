"""
Prompt Management & Optimization module.

Provides prompt versioning, A/B comparison, and iterative optimization
integrated with the AgentOps evaluation framework.
"""

from agentops.prompts.state import (
    PromptTemplate,
    PromptVersion,
    PromptCategory,
    PromptDiff,
    ComparisonConfig,
    ComparisonResult,
    OptimizationRun,
    OptimizationResult,
    DEFAULT_PROMPTS,
)
from agentops.prompts.registry import PromptRegistry
from agentops.prompts.comparator import (
    PromptComparator,
    PromptOptimizer,
    create_comparator,
    create_optimizer,
)

__all__ = [
    "PromptTemplate",
    "PromptVersion",
    "PromptCategory",
    "PromptDiff",
    "ComparisonConfig",
    "ComparisonResult",
    "OptimizationRun",
    "OptimizationResult",
    "DEFAULT_PROMPTS",
    "PromptRegistry",
    "PromptComparator",
    "PromptOptimizer",
    "create_comparator",
    "create_optimizer",
]
