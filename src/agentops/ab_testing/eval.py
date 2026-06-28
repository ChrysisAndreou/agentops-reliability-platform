"""
A/B experiment evaluation and reporting.

Provides structured evaluation reports, metric comparisons, and per-variant
summaries for A/B experiments. Generates human-readable markdown reports
suitable for sharing with stakeholders or embedding in CI/CD pipelines.

Classes:
    MetricComparison: Head-to-head comparison of a single metric.
    VariantSummary: Summary statistics for a single variant.
    ABEvalReport: Complete evaluation report.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from agentops.ab_testing.experiment import (
    ABAnalysisReport,
    ABExperimentResult,
    VariantMetrics,
)
from agentops.ab_testing.stats import confidence_interval


@dataclass
class MetricComparison:
    """Comparison of a single metric between two variants.

    Attributes:
        metric_name: Name of the metric (e.g., 'verification_pass_rate').
        control_value: Aggregate value for the control variant.
        treatment_value: Aggregate value for the treatment variant.
        absolute_difference: treatment - control.
        relative_change: (treatment - control) / control.
        significant: Whether the difference is statistically significant.
        p_value: P-value from the statistical test.
        test_used: Name of the statistical test used.
    """

    metric_name: str
    control_value: float
    treatment_value: float
    absolute_difference: float
    relative_change: float = 0.0
    significant: bool = False
    p_value: Optional[float] = None
    test_used: str = ""


@dataclass
class VariantSummary:
    """Summary statistics for a single variant in an experiment.

    Attributes:
        name: Variant name.
        total_runs: Total runs assigned.
        success_rate: Binary success rate (successes / total_runs).
        successes: Number of successful runs.
        failures: Number of failed runs.
        confidence_interval_95: 95% Wilson confidence interval for success_rate.
        metric_means: Per-metric mean values.
        mean_latency_ms: Mean latency in milliseconds.
        mean_cost_usd: Mean cost in USD.
    """

    name: str
    total_runs: int
    success_rate: float
    successes: int
    failures: int
    confidence_interval_95: tuple[float, float]
    metric_means: dict[str, float]
    mean_latency_ms: float
    mean_cost_usd: float


@dataclass
class ABEvalReport:
    """Complete A/B experiment evaluation report.

    Attributes:
        experiment_name: Name of the experiment.
        significant: Whether statistically significant difference found.
        winner: Name of winner variant (or None).
        recommendation: Human-readable recommendation.
        variants: Per-variant summaries.
        comparisons: Per-metric comparisons between control and treatment.
        raw_analysis: The underlying ABAnalysisReport.
    """

    experiment_name: str = ""
    significant: bool = False
    winner: Optional[str] = None
    recommendation: str = ""
    variants: list[VariantSummary] = field(default_factory=list)
    comparisons: list[MetricComparison] = field(default_factory=list)
    raw_analysis: Optional[ABAnalysisReport] = None


def generate_ab_report(analysis: ABAnalysisReport) -> ABEvalReport:
    """Generate a structured A/B evaluation report from an analysis.

    Produces a complete ABEvalReport with per-variant summaries and
    per-metric comparisons. Uses the first two variants by volume
    as control and treatment.

    Args:
        analysis: The ABAnalysisReport from ABExperiment.analyze().

    Returns:
        ABEvalReport with structured comparisons and summaries.
    """
    result = analysis.result
    config = result.config

    # Sort variants by volume
    sorted_variants = sorted(
        result.variants.values(),
        key=lambda v: v.total_runs,
        reverse=True,
    )

    # Build variant summaries
    variant_summaries = []
    for vm in sorted_variants:
        ci = confidence_interval(vm.successes, vm.total_runs)
        mean_latency = sum(vm.latencies) / max(len(vm.latencies), 1)
        mean_cost = sum(vm.costs) / max(len(vm.costs), 1)

        metric_means = {}
        for metric in config.metrics:
            metric_means[metric] = vm.metric_mean(metric)

        variant_summaries.append(
            VariantSummary(
                name=vm.variant_name,
                total_runs=vm.total_runs,
                success_rate=vm.success_rate,
                successes=vm.successes,
                failures=vm.failures,
                confidence_interval_95=ci,
                metric_means=metric_means,
                mean_latency_ms=mean_latency,
                mean_cost_usd=mean_cost,
            )
        )

    # Build metric comparisons (control vs treatment)
    comparisons = []
    if len(sorted_variants) >= 2:
        control = sorted_variants[0]
        treatment = sorted_variants[1]

        # Success rate comparison
        chi2_info = analysis.tests.get("chi_squared", {})
        comparisons.append(
            MetricComparison(
                metric_name="success_rate",
                control_value=control.success_rate,
                treatment_value=treatment.success_rate,
                absolute_difference=treatment.success_rate - control.success_rate,
                relative_change=(
                    (treatment.success_rate - control.success_rate) / max(control.success_rate, 1e-10)
                    if control.success_rate > 0
                    else 0.0
                ),
                significant=chi2_info.get("significant", False),
                p_value=chi2_info.get("p_value"),
                test_used="chi_squared",
            )
        )

        # Per-metric comparisons
        for metric in config.metrics:
            if metric == "success_rate" or metric == "verification_pass_rate":
                continue
            control_mean = control.metric_mean(metric)
            treatment_mean = treatment.metric_mean(metric)
            comparisons.append(
                MetricComparison(
                    metric_name=metric,
                    control_value=control_mean,
                    treatment_value=treatment_mean,
                    absolute_difference=treatment_mean - control_mean,
                    relative_change=(
                        (treatment_mean - control_mean) / max(abs(control_mean), 1e-10)
                        if abs(control_mean) > 1e-10
                        else 0.0
                    ),
                    significant=False,  # Only success_rate is tested
                    test_used="descriptive_only",
                )
            )

    return ABEvalReport(
        experiment_name=config.name or "A/B Experiment",
        significant=analysis.significant,
        winner=analysis.winner,
        recommendation=analysis.recommendation,
        variants=variant_summaries,
        comparisons=comparisons,
        raw_analysis=analysis,
    )


def format_report_markdown(report: ABEvalReport) -> str:
    """Format an ABEvalReport as a markdown string.

    Produces a human-readable report suitable for embedding in CI
    job summaries, PR comments, or dashboards.

    Args:
        report: The evaluation report to format.

    Returns:
        Markdown-formatted string.
    """
    lines = []
    lines.append(f"# A/B Experiment Report: {report.experiment_name}")
    lines.append("")

    # Verdict
    if report.significant and report.winner:
        lines.append(f"## Verdict: **{report.winner}** wins (significant)")
    else:
        lines.append("## Verdict: No statistically significant difference")
    lines.append("")

    lines.append(f"**Recommendation:** {report.recommendation}")
    lines.append("")

    # Variant summaries
    lines.append("## Variant Summaries")
    lines.append("")
    lines.append("| Variant | Runs | Success Rate | 95% CI | Mean Latency | Mean Cost |")
    lines.append("|---------|------|-------------|--------|-------------|----------|")
    for vs in report.variants:
        ci_str = f"[{vs.confidence_interval_95[0]:.1%}, {vs.confidence_interval_95[1]:.1%}]"
        lines.append(
            f"| {vs.name} | {vs.total_runs} | {vs.success_rate:.1%} | {ci_str} | "
            f"{vs.mean_latency_ms:.0f}ms | ${vs.mean_cost_usd:.4f} |"
        )
    lines.append("")

    # Metric comparisons
    if report.comparisons:
        lines.append("## Metric Comparisons (Control → Treatment)")
        lines.append("")
        lines.append("| Metric | Control | Treatment | Δ Abs | Δ Rel | Significant |")
        lines.append("|--------|---------|-----------|-------|-------|-------------|")
        for mc in report.comparisons:
            sig_marker = "✓" if mc.significant else "—"
            lines.append(
                f"| {mc.metric_name} | {mc.control_value:.4f} | {mc.treatment_value:.4f} | "
                f"{mc.absolute_difference:+.4f} | {mc.relative_change:+.1%} | {sig_marker} |"
            )
        lines.append("")

    return "\n".join(lines)
