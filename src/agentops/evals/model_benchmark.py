"""
Cross-model benchmarking framework.

Compares agent outputs from different model configurations using
the LLM-judge evaluation system. Produces side-by-side comparison
reports with statistical analysis, cost tracking, and model rankings.

Use cases:
- Compare GPT-4o vs Claude vs DeepSeek on agent tasks
- Track model degradation over time (integration with regression framework)
- Evaluate which model performs best per task category
- Cost-performance Pareto analysis
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .judge.state import JudgeConfig, JudgeDimension, JudgeResult
from .judge.judge import SimulatedJudge, JudgeRunner
from .benchmarks import ReliabilityBenchmark, ALL_BENCHMARKS


@dataclass
class ModelProfile:
    """Profile for a model being benchmarked."""

    name: str  # e.g., "gpt-4o", "claude-3-opus", "deepseek-v4"
    provider: str  # e.g., "openai", "anthropic", "deepseek"
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    avg_latency_ms: float = 0.0  # Expected average latency
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "avg_latency_ms": self.avg_latency_ms,
            "notes": self.notes,
        }


@dataclass
class ModelComparisonResult:
    """Comparison of a single model against a benchmark."""

    model: ModelProfile
    benchmark_name: str
    mean_composite: float = 0.0
    pass_rate: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    task_results: list[dict[str, Any]] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "benchmark_name": self.benchmark_name,
            "mean_composite": round(self.mean_composite, 3),
            "pass_rate": round(self.pass_rate, 3),
            "dimension_scores": {k: round(v, 3) for k, v in self.dimension_scores.items()},
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "task_results": self.task_results,
        }


@dataclass
class MultiModelReport:
    """Side-by-side comparison of multiple models across benchmarks."""

    title: str
    models: list[ModelProfile] = field(default_factory=list)
    benchmark_name: str = ""
    results: list[ModelComparisonResult] = field(default_factory=list)
    rankings: dict[str, int] = field(default_factory=dict)
    pareto_frontier: list[str] = field(default_factory=list)
    generated_at: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            f"**Generated**: {self.generated_at}",
            f"**Benchmark**: {self.benchmark_name}",
            f"**Models Compared**: {len(self.results)}",
            "",
            "## Model Rankings",
            "",
            "| Rank | Model | Provider | Composite | Pass Rate | Est. Cost | Latency |",
            "|------|-------|----------|-----------|-----------|-----------|---------|",
        ]

        # Sort by composite score descending
        ranked = sorted(self.results, key=lambda r: r.mean_composite, reverse=True)
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {r.model.name} | {r.model.provider} "
                f"| {r.mean_composite:.3f} | {r.pass_rate:.1%} "
                f"| ${r.estimated_cost_usd:.4f} | {r.total_latency_ms:.0f}ms |"
            )
        lines.append("")

        # Dimension comparison
        all_dims = set()
        for r in self.results:
            all_dims.update(r.dimension_scores.keys())

        if all_dims:
            lines.append("## Dimension Comparison")
            lines.append("| Dimension | " + " | ".join(r.model.name for r in ranked) + " |")
            lines.append("|-----------|" + "|".join("---" for _ in ranked) + "|")
            for dim in sorted(all_dims):
                scores = []
                for r in ranked:
                    s = r.dimension_scores.get(dim, 0.0)
                    # Highlight best
                    best_in_dim = max(
                        (x.dimension_scores.get(dim, 0.0) for x in ranked), default=0.0
                    )
                    marker = " **🏆**" if s == best_in_dim and best_in_dim > 0 else ""
                    scores.append(f"{s:.3f}{marker}")
                lines.append(f"| {dim} | " + " | ".join(scores) + " |")
            lines.append("")

        # Cost-performance analysis
        lines.append("## Cost-Performance Analysis")
        lines.append("| Model | Composite | Est. Cost ($) | Cost Efficiency (score/$) |")
        lines.append("|-------|-----------|---------------|---------------------------|")
        for r in ranked:
            cost_eff = (
                r.mean_composite / r.estimated_cost_usd
                if r.estimated_cost_usd > 0
                else float("inf")
            )
            lines.append(
                f"| {r.model.name} | {r.mean_composite:.3f} "
                f"| ${r.estimated_cost_usd:.4f} "
                f"| {cost_eff:.1f} |"
            )
        lines.append("")

        # Per-task breakdown
        if self.results and self.results[0].task_results:
            lines.append("## Per-Task Comparison")
            task_ids = [t.get("task_id", "") for t in self.results[0].task_results]
            lines.append("| Task | " + " | ".join(r.model.name for r in ranked) + " |")
            lines.append("|------|" + "|".join("---" for _ in ranked) + "|")
            for i, tid in enumerate(task_ids):
                scores = []
                for r in ranked:
                    if i < len(r.task_results):
                        s = r.task_results[i].get("composite_score", 0.0)
                        scores.append(f"{s:.3f}")
                    else:
                        scores.append("—")
                lines.append(f"| {tid} | " + " | ".join(scores) + " |")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "title": self.title,
            "benchmark_name": self.benchmark_name,
            "generated_at": self.generated_at,
            "rankings": self.rankings,
            "pareto_frontier": self.pareto_frontier,
            "results": [r.to_dict() for r in self.results],
        }, indent=2)


# ── Pre-configured Model Profiles ────────────────────────────────────

MODEL_PROFILES: dict[str, ModelProfile] = {
    "gpt-4o": ModelProfile(
        name="gpt-4o",
        provider="openai",
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        avg_latency_ms=800,
        notes="OpenAI flagship; strong reasoning, broad knowledge",
    ),
    "gpt-4o-mini": ModelProfile(
        name="gpt-4o-mini",
        provider="openai",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        avg_latency_ms=400,
        notes="Cost-efficient OpenAI model; good for high-throughput eval",
    ),
    "claude-3-opus": ModelProfile(
        name="claude-3-opus",
        provider="anthropic",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        avg_latency_ms=1200,
        notes="Anthropic flagship; excellent at nuanced reasoning and safety",
    ),
    "claude-3-sonnet": ModelProfile(
        name="claude-3-sonnet",
        provider="anthropic",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=700,
        notes="Balanced Anthropic model; good price-performance",
    ),
    "deepseek-v4": ModelProfile(
        name="deepseek-v4",
        provider="deepseek",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.002,
        avg_latency_ms=600,
        notes="Cost-effective open-weight model; strong on technical tasks",
    ),
    "simulated-production": ModelProfile(
        name="simulated-production",
        provider="simulated",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=200,
        notes="Deterministic simulated agent; CI/CD baseline",
    ),
    "simulated-development": ModelProfile(
        name="simulated-development",
        provider="simulated",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=250,
        notes="Slightly degraded simulated agent; development baseline",
    ),
}


# ── Model Benchmark Runner ───────────────────────────────────────────

@dataclass
class ModelBenchmark:
    """Runs cross-model comparisons using the LLM-judge framework.

    Usage:
        bench = ModelBenchmark(judge_config=JudgeConfig())
        report = bench.compare(
            models=["gpt-4o", "claude-3-sonnet", "deepseek-v4"],
            benchmark_name="support-tickets",
            agent_outputs=outputs_dict,
        )
        print(report.to_markdown())
    """

    judge_config: JudgeConfig = field(default_factory=JudgeConfig)
    use_simulated: bool = True

    def compare(
        self,
        models: list[str],
        benchmark_name: str,
        agent_outputs: dict[str, dict[str, dict[str, Any]]],
    ) -> MultiModelReport:
        """Compare multiple models on the same benchmark.

        agent_outputs: {model_name: {task_id: {"output": str, "key_terms": [...], ...}}}
        """
        runner = JudgeRunner(config=self.judge_config, use_simulated=self.use_simulated)
        results: list[ModelComparisonResult] = []

        for model_name in models:
            profile = MODEL_PROFILES.get(model_name, ModelProfile(name=model_name, provider="unknown"))
            outputs = agent_outputs.get(model_name, {})

            judge_result = runner.evaluate_benchmark(
                benchmark_name=benchmark_name,
                agent_outputs=outputs,
                agent_model=model_name,
            )

            # Build model comparison result
            dim_scores = judge_result.summary.get("dimension_means", {})
            total_tokens = sum(
                len(o.get("output", "")) // 4 for o in outputs.values()
            )  # Rough token estimate
            estimated_cost = (
                total_tokens / 1000 * profile.cost_per_1k_output
                + total_tokens / 1000 * profile.cost_per_1k_input * 0.3
            )

            results.append(ModelComparisonResult(
                model=profile,
                benchmark_name=benchmark_name,
                mean_composite=judge_result.mean_composite,
                pass_rate=judge_result.pass_rate,
                dimension_scores=dim_scores,
                task_results=[r.to_dict() for r in judge_result.results],
                estimated_cost_usd=estimated_cost,
                total_latency_ms=profile.avg_latency_ms * len(outputs),
            ))

        # Compute rankings
        sorted_results = sorted(results, key=lambda r: r.mean_composite, reverse=True)
        rankings = {r.model.name: i + 1 for i, r in enumerate(sorted_results)}

        # Pareto frontier: best score/cost tradeoff
        pareto = self._compute_pareto(results)

        return MultiModelReport(
            title=f"Model Comparison: {benchmark_name}",
            models=[MODEL_PROFILES.get(m, ModelProfile(name=m, provider="unknown")) for m in models],
            benchmark_name=benchmark_name,
            results=results,
            rankings=rankings,
            pareto_frontier=pareto,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def compare_across_benchmarks(
        self,
        models: list[str],
        agent_outputs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    ) -> dict[str, MultiModelReport]:
        """Compare models across multiple benchmarks.

        agent_outputs: {benchmark_name: {model_name: {task_id: {...}}}}
        """
        reports: dict[str, MultiModelReport] = {}
        for bench_name, outputs in agent_outputs.items():
            reports[bench_name] = self.compare(
                models=models,
                benchmark_name=bench_name,
                agent_outputs=outputs,
            )
        return reports

    def _compute_pareto(self, results: list[ModelComparisonResult]) -> list[str]:
        """Identify models on the Pareto frontier (best score/cost tradeoff)."""
        pareto: list[str] = []
        for i, r1 in enumerate(results):
            dominated = False
            for j, r2 in enumerate(results):
                if i == j:
                    continue
                # r2 dominates r1 if it has better or equal score AND lower or equal cost
                if (r2.mean_composite >= r1.mean_composite
                        and r2.estimated_cost_usd <= r1.estimated_cost_usd
                        and (r2.mean_composite > r1.mean_composite
                             or r2.estimated_cost_usd < r1.estimated_cost_usd)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r1.model.name)
        return pareto

    def generate_aggregate_report(
        self,
        reports: dict[str, MultiModelReport],
    ) -> str:
        """Generate an aggregate markdown report across benchmarks."""
        lines = [
            "# Aggregate Model Comparison Report",
            f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Benchmarks**: {', '.join(reports.keys())}",
            "",
            "## Overall Rankings",
            "",
        ]

        # Aggregate scores across benchmarks
        model_scores: dict[str, list[float]] = {}
        model_benchmark_wins: dict[str, int] = {}
        for bench_name, report in reports.items():
            for r in report.results:
                name = r.model.name
                if name not in model_scores:
                    model_scores[name] = []
                    model_benchmark_wins[name] = 0
                model_scores[name].append(r.mean_composite)
                if report.rankings.get(name) == 1:
                    model_benchmark_wins[name] += 1

        avg_scores = {
            name: sum(scores) / len(scores)
            for name, scores in model_scores.items()
        }

        ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        lines.append("| Rank | Model | Avg Composite | Benchmark Wins |")
        lines.append("|------|-------|---------------|----------------|")
        for i, (name, score) in enumerate(ranked, 1):
            wins = model_benchmark_wins.get(name, 0)
            lines.append(f"| {i} | {name} | {score:.3f} | {wins}/{len(reports)} |")
        lines.append("")

        # Per-benchmark summaries
        for bench_name, report in reports.items():
            lines.append(f"## {bench_name}")
            ranked_b = sorted(report.results, key=lambda r: r.mean_composite, reverse=True)
            lines.append("| Rank | Model | Composite | Pass Rate | Cost |")
            lines.append("|------|-------|-----------|-----------|------|")
            for i, r in enumerate(ranked_b, 1):
                lines.append(
                    f"| {i} | {r.model.name} | {r.mean_composite:.3f} "
                    f"| {r.pass_rate:.1%} | ${r.estimated_cost_usd:.4f} |"
                )
            lines.append("")

        return "\n".join(lines)
