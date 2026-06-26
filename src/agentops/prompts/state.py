"""
State models for Prompt Management & Optimization.

Defines the data structures for prompt templates, versioning, registry,
A/B comparison, and iterative optimization results.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromptCategory(str, Enum):
    """Categories of prompts for organizational grouping."""
    SYSTEM = "system"
    TASK = "task"
    RETRIEVAL = "retrieval"
    VERIFICATION = "verification"
    TOOL_USE = "tool_use"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CUSTOM = "custom"


@dataclass
class PromptTemplate:
    """A prompt template with variable placeholders.

    Placeholders use {{variable}} syntax. Variables are extracted
    at registration time and validated at render time.
    """

    name: str
    content: str
    description: str = ""
    category: PromptCategory = PromptCategory.CUSTOM
    variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        """Extract {{variable}} placeholders from content."""
        import re
        return sorted(set(re.findall(r'\{\{(\w+)\}\}', self.content)))

    def render(self, **kwargs) -> str:
        """Render the template with provided variable values.

        Raises ValueError if required variables are missing.
        """
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing variables for prompt '{self.name}': {sorted(missing)}"
            )
        result = self.content
        for var, value in kwargs.items():
            result = result.replace(f"{{{{{var}}}}}", str(value))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "variables": self.variables,
            "metadata": self.metadata,
        }


@dataclass
class PromptVersion:
    """A specific version of a prompt template.

    Each version is immutable once created. The content_hash
    provides content-addressable identity.
    """

    prompt_name: str
    version: int
    content: str
    created_at: float = field(default_factory=time.time)
    author: str = "agentops"
    changelog: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "version": self.version,
            "content": self.content,
            "created_at": self.created_at,
            "author": self.author,
            "changelog": self.changelog,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


@dataclass
class PromptDiff:
    """Difference between two prompt versions."""

    prompt_name: str
    version_a: int
    version_b: int
    lines_added: list[str] = field(default_factory=list)
    lines_removed: list[str] = field(default_factory=list)
    lines_unchanged: int = 0

    @property
    def total_changes(self) -> int:
        return len(self.lines_added) + len(self.lines_removed)

    def to_summary(self) -> str:
        return (
            f"Prompt '{self.prompt_name}' v{self.version_a} → v{self.version_b}: "
            f"+{len(self.lines_added)} lines, -{len(self.lines_removed)} lines, "
            f"{self.lines_unchanged} unchanged"
        )


@dataclass
class ComparisonConfig:
    """Configuration for A/B prompt comparison."""

    prompt_name: str
    version_a: int
    version_b: int
    benchmark_names: list[str] = field(default_factory=lambda: ["support-tickets"])
    num_runs: int = 3
    significance_threshold: float = 0.05
    metric_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "version_a": self.version_a,
            "version_b": self.version_b,
            "benchmark_names": self.benchmark_names,
            "num_runs": self.num_runs,
            "significance_threshold": self.significance_threshold,
            "metric_weights": self.metric_weights,
        }


@dataclass
class ComparisonResult:
    """Result of A/B comparing two prompt versions."""

    config: ComparisonConfig
    version_a_scores: dict[str, float] = field(default_factory=dict)
    version_b_scores: dict[str, float] = field(default_factory=dict)
    winner: str = ""  # "a", "b", or "tie"
    confidence: float = 0.0
    per_benchmark: dict[str, dict[str, Any]] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.config.prompt_name,
            "version_a": self.config.version_a,
            "version_b": self.config.version_b,
            "winner": self.winner,
            "confidence": round(self.confidence, 3),
            "version_a_scores": {k: round(v, 3) for k, v in self.version_a_scores.items()},
            "version_b_scores": {k: round(v, 3) for k, v in self.version_b_scores.items()},
            "per_benchmark": self.per_benchmark,
            "recommendation": self.recommendation,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Prompt Comparison: {self.config.prompt_name}",
            f"",
            f"**Versions**: v{self.config.version_a} vs v{self.config.version_b}",
            f"**Winner**: {self.winner.upper()} (confidence: {self.confidence:.1%})",
            f"",
            f"## Aggregate Scores",
            f"",
            f"| Metric | v{self.config.version_a} | v{self.config.version_b} | Δ |",
            f"|--------|-------------------------|-------------------------|---|",
        ]
        all_metrics = set(self.version_a_scores.keys()) | set(self.version_b_scores.keys())
        for metric in sorted(all_metrics):
            a = self.version_a_scores.get(metric, 0)
            b = self.version_b_scores.get(metric, 0)
            delta = b - a
            sign = "+" if delta > 0 else ""
            lines.append(f"| {metric} | {a:.3f} | {b:.3f} | {sign}{delta:.3f} |")

        lines.extend([
            f"",
            f"## Recommendation",
            f"",
            self.recommendation,
        ])

        if self.per_benchmark:
            lines.extend([f"", f"## Per-Benchmark Results", f""])
            for bench_name, bench_data in self.per_benchmark.items():
                lines.append(f"### {bench_name}")
                lines.append(f"")
                if isinstance(bench_data, dict):
                    for k, v in bench_data.items():
                        lines.append(f"- **{k}**: {v}")
                lines.append("")

        return "\n".join(lines)


@dataclass
class OptimizationRun:
    """A single optimization iteration result."""

    iteration: int
    prompt_content: str
    scores: dict[str, float]  # metric → score
    changes_made: str = ""
    reasoning: str = ""


@dataclass
class OptimizationResult:
    """Result of iterative prompt optimization."""

    prompt_name: str
    initial_version: int
    final_content: str
    iterations: list[OptimizationRun] = field(default_factory=list)
    best_iteration: int = 0
    best_scores: dict[str, float] = field(default_factory=dict)
    improvement: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "initial_version": self.initial_version,
            "iterations": len(self.iterations),
            "best_iteration": self.best_iteration,
            "best_scores": {k: round(v, 3) for k, v in self.best_scores.items()},
            "improvement": {k: round(v, 3) for k, v in self.improvement.items()},
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Prompt Optimization: {self.prompt_name}",
            f"",
            f"**Initial version**: v{self.initial_version}",
            f"**Iterations**: {len(self.iterations)}",
            f"**Best iteration**: {self.best_iteration}",
            f"**Time**: {self.elapsed_seconds:.1f}s",
            f"",
            f"## Score Progression",
            f"",
            f"| Iteration | " + " | ".join(sorted(self.best_scores.keys())) + " |",
            f"|-----------|" + "|".join(["-" * 10 for _ in self.best_scores]) + "|",
        ]
        for run in self.iterations:
            scores = run.scores
            score_str = " | ".join(
                f"{scores.get(m, 0):.3f}" for m in sorted(self.best_scores.keys())
            )
            marker = " ← BEST" if run.iteration == self.best_iteration else ""
            lines.append(f"| {run.iteration}{marker} | {score_str} |")

        lines.extend([
            f"",
            f"## Improvement",
            f"",
        ])
        for metric, delta in sorted(self.improvement.items()):
            sign = "+" if delta > 0 else ""
            lines.append(f"- **{metric}**: {sign}{delta:.3f}")

        lines.extend([
            f"",
            f"## Optimized Prompt",
            f"",
            f"```",
            self.final_content,
            f"```",
        ])

        return "\n".join(lines)


# ── Built-in Prompt Templates ────────────────────────────────────────

DEFAULT_PROMPTS = {
    "reliability-agent-system": PromptTemplate(
        name="reliability-agent-system",
        description="System prompt for the reliability agent",
        category=PromptCategory.SYSTEM,
        content=(
            "You are a reliability-focused AI assistant for the {{product}} platform. "
            "Your task is to answer user questions accurately using ONLY the provided "
            "documentation. Follow these rules:\n\n"
            "1. **Retrieve first**: Search documentation before answering.\n"
            "2. **Ground every claim**: Cite specific document sections.\n"
            "3. **Verify before responding**: Check facts against retrieved content.\n"
            "4. **Be honest about gaps**: If information is missing, say so.\n"
            "5. **Use tools responsibly**: Only execute safe, documented operations.\n\n"
            "When you cannot verify a claim, prefix it with '[Unverified]'. "
            "Never invent features, APIs, or configurations."
        ),
    ),
    "support-ticket-triage": PromptTemplate(
        name="support-ticket-triage",
        description="Prompt for triaging support tickets",
        category=PromptCategory.TASK,
        content=(
            "You are triaging a support ticket for {{product}}.\n\n"
            "**Ticket**: {{ticket_content}}\n\n"
            "Classify this ticket by:\n"
            "1. **Severity**: critical/high/medium/low\n"
            "2. **Category**: authentication/deployment/configuration/billing/other\n"
            "3. **Plan**: the {{plan_type}} plan\n\n"
            "Provide your triage with reasoning."
        ),
    ),
    "verification-check": PromptTemplate(
        name="verification-check",
        description="Prompt for verifying answer correctness",
        category=PromptCategory.VERIFICATION,
        content=(
            "Verify the following answer against the source documentation.\n\n"
            "**Question**: {{question}}\n"
            "**Answer**: {{answer}}\n"
            "**Sources**: {{sources}}\n\n"
            "For each claim in the answer:\n"
            "1. Mark it as [VERIFIED], [PARTIALLY_VERIFIED], or [UNVERIFIED]\n"
            "2. Cite the specific source line that supports or contradicts it\n"
            "3. Flag any hallucinated content\n\n"
            "Return a verification report with confidence score (0.0-1.0)."
        ),
    ),
    "chain-of-thought-reasoning": PromptTemplate(
        name="chain-of-thought-reasoning",
        description="Chain-of-thought prompt for complex multi-step problems",
        category=PromptCategory.CHAIN_OF_THOUGHT,
        content=(
            "Solve the following problem step by step. Show your reasoning at each step.\n\n"
            "**Problem**: {{problem}}\n"
            "**Context**: {{context}}\n\n"
            "Follow this structure:\n"
            "1. **Understand**: Restate the problem in your own words.\n"
            "2. **Decompose**: Break into sub-problems.\n"
            "3. **Research**: What information do you need?\n"
            "4. **Solve**: Address each sub-problem.\n"
            "5. **Synthesize**: Combine into a final answer.\n"
            "6. **Verify**: Check each sub-solution against context.\n\n"
            "Show your work at each step."
        ),
    ),
    "tool-use-decision": PromptTemplate(
        name="tool-use-decision",
        description="Prompt for deciding when to use tools",
        category=PromptCategory.TOOL_USE,
        content=(
            "Determine which tools (if any) are needed to answer this question.\n\n"
            "**Question**: {{question}}\n"
            "**Available tools**: {{available_tools}}\n\n"
            "For each tool you consider:\n"
            "1. **Tool name**: Which tool?\n"
            "2. **Necessity**: Is this tool REQUIRED or OPTIONAL?\n"
            "3. **Parameters**: What parameters would you pass?\n"
            "4. **Risk**: Is this operation safe? (safe/caution/dangerous)\n\n"
            "Only recommend tools that are necessary. Prefer retrieval over execution.\n"
            "Flag any dangerous operations for human review."
        ),
    ),
}
