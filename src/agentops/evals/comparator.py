"""
Comparative evaluation framework for agent reliability.

Enables A/B testing between agent configurations, regression detection
across versions, and multi-profile comparison with statistical summaries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import ReliabilityMetrics, compute_metrics
from .benchmarks import ReliabilityBenchmark
from .simulator import SimConfig, SimulatedAgent


@dataclass
class ComparisonResult:
    """Head-to-head comparison of two agent configurations."""

    benchmark_name: str
    config_a: str
    config_b: str
    metrics_a: list[ReliabilityMetrics]
    metrics_b: list[ReliabilityMetrics]
    winner: str  # "a", "b", or "tie"
    deltas: dict[str, float] = field(default_factory=dict)
    significant_dims: list[str] = field(default_factory=list)
    run_timestamp: str = ""

    def to_markdown(self) -> str:
        # Compute per-config means for display
        def means(metrics_list):
            result = {}
            for name in ["groundedness", "citation_precision", "verification_pass_rate",
                         "tool_success_rate", "answer_completeness", "composite"]:
                vals = [getattr(m, name) for m in metrics_list]
                result[name] = sum(vals) / max(len(vals), 1)
            return result

        ma = means(self.metrics_a)
        mb = means(self.metrics_b)

        lines = [
            f"# A/B Comparison: {self.config_a} vs {self.config_b}",
            f"Benchmark: {self.benchmark_name} | {self.run_timestamp}",
            "",
            f"**Winner: {self.winner.upper()}**",
            "",
            "## Metric Deltas",
            "",
            f"| Metric | {self.config_a} | {self.config_b} | Delta | Winner |",
            "|--------|---------|---------|-------|--------|",
        ]

        for metric_name, delta in self.deltas.items():
            val_a = ma.get(metric_name, 0)
            val_b = mb.get(metric_name, 0)
            direction = "A" if delta > 0 else ("B" if delta < 0 else "tie")
            sign = "+" if delta > 0 else ""
            lines.append(
                f"| {metric_name} | {val_a:.3f} | {val_b:.3f} | {sign}{delta:.3f} | {direction} |"
            )

        if self.significant_dims:
            lines.append("")
            lines.append("## Significant Differences")
            for dim in self.significant_dims:
                lines.append(f"- **{dim}**: statistical difference detected")

        return "\n".join(lines)


@dataclass
class RegressionReport:
    """Regression detection report comparing current vs baseline."""

    baseline_name: str
    current_name: str
    benchmark_name: str
    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    stable: list[str] = field(default_factory=list)
    has_regressions: bool = False
    run_timestamp: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# Regression Report: {self.current_name} vs baseline {self.baseline_name}",
            f"Benchmark: {self.benchmark_name} | {self.run_timestamp}",
            "",
        ]

        if self.has_regressions:
            lines.append("## ⚠️ REGRESSIONS DETECTED")
            for r in self.regressions:
                lines.append(
                    f"- **{r['metric']}**: {r['baseline']:.3f} → {r['current']:.3f} "
                    f"(Δ {r['delta']:+.3f}, threshold={r['threshold']:.3f})"
                )
        else:
            lines.append("## ✅ No Regressions Detected")
            lines.append("All metrics are within acceptable thresholds of the baseline.")

        if self.improvements:
            lines.append("")
            lines.append("## Improvements")
            for imp in self.improvements:
                lines.append(
                    f"- **{imp['metric']}**: {imp['baseline']:.3f} → {imp['current']:.3f} "
                    f"(Δ {imp['delta']:+.3f})"
                )

        if self.stable:
            lines.append("")
            lines.append("## Stable Metrics")
            for m in self.stable:
                lines.append(f"- {m}")

        return "\n".join(lines)


class EvalComparator:
    """Compare agent configurations and detect regressions.

    Usage:
        comparator = EvalComparator()
        result = await comparator.compare(
            benchmark, config_a=PRODUCTION_AGENT, config_b=DEVELOPMENT_AGENT
        )
        print(result.to_markdown())

        regression = comparator.detect_regression(
            benchmark, baseline=PRODUCTION_AGENT, current=DEVELOPMENT_AGENT
        )
    """

    # Thresholds for "significant" regression (metric must drop by this much)
    REGRESSION_THRESHOLDS = {
        "groundedness": 0.05,
        "citation_precision": 0.10,
        "verification_pass_rate": 0.10,
        "tool_success_rate": 0.05,
        "answer_completeness": 0.10,
        "composite": 0.03,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    async def compare(
        self,
        benchmark: ReliabilityBenchmark,
        config_a: SimConfig,
        config_b: SimConfig,
    ) -> ComparisonResult:
        """Run A/B comparison of two agent configurations on a benchmark."""
        agent_a = SimulatedAgent(config=config_a, seed=self.seed)
        agent_b = SimulatedAgent(config=config_b, seed=self.seed + 1)

        metrics_a = []
        metrics_b = []

        for task in benchmark.tasks:
            result_a = await agent_a.run(task.question, task_id=task.id)
            result_b = await agent_b.run(task.question, task_id=task.id)
            metrics_a.append(compute_metrics(result_a, key_terms=task.key_terms))
            metrics_b.append(compute_metrics(result_b, key_terms=task.key_terms))

        # Compute mean deltas for each metric dimension
        metric_names = [
            "groundedness", "citation_precision", "verification_pass_rate",
            "tool_success_rate", "answer_completeness", "composite",
        ]

        deltas = {}
        significant_dims = []
        scores_a = 0
        scores_b = 0

        for name in metric_names:
            vals_a = [getattr(m, name) for m in metrics_a]
            vals_b = [getattr(m, name) for m in metrics_b]
            mean_a = sum(vals_a) / len(vals_a)
            mean_b = sum(vals_b) / len(vals_b)
            delta = mean_a - mean_b
            deltas[name] = round(delta, 4)

            if abs(delta) > self.REGRESSION_THRESHOLDS.get(name, 0.05):
                significant_dims.append(name)

            if delta > 0.001:
                scores_a += 1
            elif delta < -0.001:
                scores_b += 1

        # Determine winner
        if scores_a > scores_b:
            winner = "a"
        elif scores_b > scores_a:
            winner = "b"
        else:
            winner = "tie"

        return ComparisonResult(
            benchmark_name=benchmark.name,
            config_a=config_a.name,
            config_b=config_b.name,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            winner=winner,
            deltas=deltas,
            significant_dims=significant_dims,
            run_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def detect_regression(
        self,
        metrics_baseline: list[ReliabilityMetrics],
        metrics_current: list[ReliabilityMetrics],
        baseline_name: str = "baseline",
        current_name: str = "current",
        benchmark_name: str = "unknown",
    ) -> RegressionReport:
        """Detect regressions by comparing current metrics to baseline.

        A metric is considered a regression if it drops below the
        configured threshold relative to baseline.
        """
        metric_names = [
            "groundedness", "citation_precision", "verification_pass_rate",
            "tool_success_rate", "answer_completeness", "composite",
        ]

        regressions = []
        improvements = []
        stable = []

        for name in metric_names:
            vals_base = [getattr(m, name) for m in metrics_baseline]
            vals_curr = [getattr(m, name) for m in metrics_current]
            mean_base = sum(vals_base) / max(len(vals_base), 1)
            mean_curr = sum(vals_curr) / max(len(vals_curr), 1)
            delta = mean_curr - mean_base
            threshold = self.REGRESSION_THRESHOLDS.get(name, 0.05)

            entry = {
                "metric": name,
                "baseline": round(mean_base, 4),
                "current": round(mean_curr, 4),
                "delta": round(delta, 4),
                "threshold": threshold,
            }

            if delta < -threshold:
                regressions.append(entry)
            elif delta > threshold:
                improvements.append(entry)
            else:
                stable.append(name)

        return RegressionReport(
            baseline_name=baseline_name,
            current_name=current_name,
            benchmark_name=benchmark_name,
            regressions=regressions,
            improvements=improvements,
            stable=stable,
            has_regressions=len(regressions) > 0,
            run_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


async def run_multi_profile_eval(
    benchmark: ReliabilityBenchmark,
    profiles: list[SimConfig] | None = None,
    output_dir: str | None = None,
    seed: int = 42,
) -> dict[str, list[ReliabilityMetrics]]:
    """Run a benchmark against multiple agent profiles and collect results.

    Returns a dict mapping profile name to list of metrics.
    If output_dir is set, writes comparison reports.
    """
    from .simulator import ALL_PROFILES

    if profiles is None:
        profiles = ALL_PROFILES

    all_metrics: dict[str, list[ReliabilityMetrics]] = {}

    for profile in profiles:
        agent = SimulatedAgent(config=profile, seed=seed)
        metrics = []
        for task in benchmark.tasks:
            result = await agent.run(task.question, task_id=task.id)
            metrics.append(compute_metrics(result, key_terms=task.key_terms))
        all_metrics[profile.name] = metrics

    # Write summary report
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_comparison_summary(benchmark, all_metrics, out)

    return all_metrics


def _write_comparison_summary(
    benchmark: ReliabilityBenchmark,
    all_metrics: dict[str, list[ReliabilityMetrics]],
    output_dir: Path,
) -> None:
    """Write a multi-profile comparison report."""
    metric_names = [
        "groundedness", "citation_precision", "verification_pass_rate",
        "composite",
    ]

    lines = [
        f"# Multi-Profile Comparison: {benchmark.name}",
        f"Profiles: {', '.join(all_metrics.keys())}",
        f"Tasks: {len(benchmark.tasks)}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Profile | Groundedness | Citation Prec | Verify Rate | Composite |",
        "|---------|-------------|--------------|-------------|-----------|",
    ]

    for profile_name, metrics in all_metrics.items():
        means = {}
        for name in metric_names:
            vals = [getattr(m, name) for m in metrics]
            means[name] = sum(vals) / max(len(vals), 1)

        lines.append(
            f"| {profile_name} | {means['groundedness']:.3f} | "
            f"{means['citation_precision']:.3f} | "
            f"{means['verification_pass_rate']:.3f} | "
            f"{means['composite']:.3f} |"
        )

    # Per-task breakdown
    lines.append("")
    lines.append("## Per-Task Breakdown")
    lines.append("")

    for i, task in enumerate(benchmark.tasks):
        lines.append(f"### {task.id}: {task.question[:80]}...")
        lines.append("")
        lines.append("| Profile | Grounded | Verified | Composite |")
        lines.append("|---------|----------|----------|-----------|")

        for profile_name, metrics in all_metrics.items():
            m = metrics[i]
            verified = "✓" if m.verification_passed else "✗"
            lines.append(
                f"| {profile_name} | {m.groundedness:.3f} | "
                f"{verified} | {m.composite:.3f} |"
            )
        lines.append("")

    report_path = output_dir / f"{benchmark.name}_comparison.md"
    report_path.write_text("\n".join(lines))

    # Also write JSON for programmatic use
    json_path = output_dir / f"{benchmark.name}_comparison.json"
    json_data = {}
    for profile_name, metrics in all_metrics.items():
        json_data[profile_name] = {
            "summary": {
                name: round(
                    sum(getattr(m, name) for m in metrics) / max(len(metrics), 1), 4
                )
                for name in metric_names
            },
            "per_task": [m.to_dict() for m in metrics],
        }
    json_path.write_text(json.dumps(json_data, indent=2))
