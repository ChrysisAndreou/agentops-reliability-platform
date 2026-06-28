"""
Cost efficiency evaluation for AI agent operations.

Measures cost efficiency, budget compliance, and cost-vs-quality trade-offs.
Bridges the gap between raw cost data and actionable insights — answering not
just "how much did we spend?" but "was it worth it?"

Key evaluation dimensions:
- Cost-per-task and cost-per-success metrics
- Budget compliance tracking with violation analysis
- Cost vs. quality trade-off analysis (Pareto frontier)
- Cost regression detection across model/agent versions
- Structured reports suitable for CI/CD and team review
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from agentops.cost.tracker import BudgetStatus, CostBreakdown, CostRecord, CostTracker


@dataclass
class CostEfficiencyMetrics:
    """Core cost efficiency KPIs."""

    total_cost: float
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    cost_per_task: float
    cost_per_success: float
    avg_input_tokens_per_task: float
    avg_output_tokens_per_task: float
    cache_hit_rate: float
    avg_cost_per_1k_tokens: float

    @property
    def success_rate(self) -> float:
        if self.total_tasks <= 0:
            return 0.0
        return self.successful_tasks / self.total_tasks


@dataclass
class BudgetComplianceReport:
    """Budget compliance analysis across all defined budgets."""

    budget_name: str
    limit_usd: float
    spent_usd: float
    utilization: float
    status: str
    overspend_usd: float = 0.0
    days_remaining: Optional[int] = None
    projected_total: Optional[float] = None


@dataclass
class CostVsQuality:
    """Cost vs. quality trade-off data point.

    Used to build the Pareto frontier: for a given agent configuration,
    what quality level was achieved at what cost.
    """

    config_name: str
    total_cost: float
    success_rate: float
    cost_per_success: float

    def dominates(self, other: CostVsQuality) -> bool:
        """Returns True if this point dominates other (lower cost AND higher quality)."""
        return (
            self.cost_per_success <= other.cost_per_success
            and self.success_rate >= other.success_rate
        ) and (
            self.cost_per_success < other.cost_per_success
            or self.success_rate > other.success_rate
        )


@dataclass
class CostRegression:
    """Cost regression between two agent versions."""

    version_a: str
    version_b: str
    cost_a: float
    cost_b: float
    cost_change_pct: float
    is_regression: bool
    severity: str = "info"

    @property
    def description(self) -> str:
        direction = "increase" if self.cost_change_pct > 0 else "decrease"
        severity_marker = "⚠️" if self.is_regression else "✓"
        return (
            f"{severity_marker} {self.version_a} → {self.version_b}: "
            f"{abs(self.cost_change_pct):.1f}% cost {direction} "
            f"(${self.cost_a:.4f} → ${self.cost_b:.4f})"
        )


@dataclass
class CostEvalReport:
    """Complete cost evaluation report."""

    metrics: CostEfficiencyMetrics
    compliance: list[BudgetComplianceReport] = field(default_factory=list)
    pareto_frontier: list[CostVsQuality] = field(default_factory=list)
    regressions: list[CostRegression] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Overall health check — no major issues."""
        return (
            not any(r.is_regression and r.severity == "critical" for r in self.regressions)
            and not any(c.status in ("EXCEEDED", "EXHAUSTED") for c in self.compliance)
        )

    def summary(self) -> str:
        lines = [
            "Cost Efficiency Report",
            "=" * 60,
            f"Total cost:           ${self.metrics.total_cost:.4f}",
            f"Total tasks:          {self.metrics.total_tasks}",
            f"Success rate:         {self.metrics.success_rate:.1%}",
            f"Cost per task:        ${self.metrics.cost_per_task:.4f}",
            f"Cost per success:     ${self.metrics.cost_per_success:.4f}",
            f"Avg tokens/task:      {self.metrics.avg_input_tokens_per_task:.0f} in / {self.metrics.avg_output_tokens_per_task:.0f} out",
            f"Cache hit rate:       {self.metrics.cache_hit_rate:.1%}",
        ]

        if self.compliance:
            lines.append(f"\nBudget Compliance:")
            for c in self.compliance:
                lines.append(
                    f"  {c.budget_name:<20} ${c.spent_usd:.2f} / ${c.limit_usd:.2f}  "
                    f"({c.utilization:.0%}) [{c.status}]"
                )

        if self.regressions:
            lines.append(f"\nCost Regressions:")
            for r in self.regressions:
                lines.append(f"  {r.description}")

        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)


def evaluate_cost_efficiency(
    tracker: CostTracker,
    task_results: list[dict],
    session_id: str = "",
) -> CostEvalReport:
    """Evaluate cost efficiency from a tracker and task results.

    Args:
        tracker: CostTracker with recorded calls.
        task_results: List of {"task_id": str, "success": bool} for each task.
        session_id: Optional session filter.

    Returns:
        CostEvalReport with metrics, compliance, and recommendations.
    """
    breakdown = tracker.get_breakdown("project")
    total_cost = tracker.total_cost
    total_tasks = len(task_results)
    successful_tasks = sum(1 for t in task_results if t.get("success", False))
    failed_tasks = total_tasks - successful_tasks

    # Cost per task metrics
    cost_per_task = total_cost / total_tasks if total_tasks > 0 else 0.0
    cost_per_success = (
        total_cost / successful_tasks if successful_tasks > 0 else float("inf")
    )

    # Token statistics
    total_calls = breakdown.total.call_count
    avg_input = breakdown.total.input_tokens / total_calls if total_calls > 0 else 0
    avg_output = breakdown.total.output_tokens / total_calls if total_calls > 0 else 0
    cache_rate = (
        breakdown.total.cached_input_tokens
        / breakdown.total.input_tokens
        if breakdown.total.input_tokens > 0
        else 0.0
    )
    avg_cost_per_1k = (
        (total_cost / breakdown.total.total_tokens * 1000)
        if breakdown.total.total_tokens > 0
        else 0.0
    )

    metrics = CostEfficiencyMetrics(
        total_cost=total_cost,
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        cost_per_task=round(cost_per_task, 6),
        cost_per_success=round(cost_per_success, 6),
        avg_input_tokens_per_task=round(avg_input, 1),
        avg_output_tokens_per_task=round(avg_output, 1),
        cache_hit_rate=round(cache_rate, 4),
        avg_cost_per_1k_tokens=round(avg_cost_per_1k, 6),
    )

    # Compliance
    compliance: list[BudgetComplianceReport] = []
    for budget in tracker.budgets:
        spent = tracker._budget_spend_for(budget)
        status = budget.check(spent)
        overspend = max(0.0, spent - budget.limit_usd)
        compliance.append(
            BudgetComplianceReport(
                budget_name=budget.name,
                limit_usd=budget.limit_usd,
                spent_usd=round(spent, 4),
                utilization=round(budget.utilization(spent), 4),
                status=status.name,
                overspend_usd=round(overspend, 4),
            )
        )

    # Recommendations
    recommendations = _generate_recommendations(metrics, compliance)

    return CostEvalReport(
        metrics=metrics,
        compliance=compliance,
        recommendations=recommendations,
    )


def evaluate_cost_vs_quality(
    config_results: list[dict],
) -> tuple[list[CostVsQuality], list[CostVsQuality]]:
    """Compute Pareto frontier for cost-vs-quality trade-offs.

    Args:
        config_results: List of {"config_name": str, "total_cost": float,
                       "success_rate": float} for each agent configuration.

    Returns:
        (all_points, pareto_frontier_points) tuple.
    """
    points = [
        CostVsQuality(
            config_name=r["config_name"],
            total_cost=r["total_cost"],
            success_rate=r["success_rate"],
            cost_per_success=(
                r["total_cost"] / max(0.01, r["success_rate"])
                if r["success_rate"] > 0
                else float("inf")
            ),
        )
        for r in config_results
    ]

    # Find Pareto frontier — points not dominated by any other point
    frontier: list[CostVsQuality] = []
    for p in points:
        dominated = any(other.dominates(p) for other in points if other is not p)
        if not dominated:
            frontier.append(p)

    # Sort frontier by cost (ascending)
    frontier.sort(key=lambda p: p.cost_per_success)

    return points, frontier


def detect_cost_regressions(
    version_costs: list[dict],
    threshold_pct: float = 10.0,
) -> list[CostRegression]:
    """Detect cost regressions between consecutive agent versions.

    Args:
        version_costs: List of {"version": str, "cost": float} in order.
        threshold_pct: Percentage increase that triggers a regression alert.

    Returns:
        List of CostRegression entries.
    """
    regressions: list[CostRegression] = []

    for i in range(1, len(version_costs)):
        prev = version_costs[i - 1]
        curr = version_costs[i]

        cost_diff = curr["cost"] - prev["cost"]
        cost_change_pct = (cost_diff / prev["cost"]) * 100 if prev["cost"] > 0 else 0

        is_regression = cost_change_pct > threshold_pct
        severity = "info"
        if cost_change_pct > 50:
            severity = "critical"
        elif cost_change_pct > 25:
            severity = "warning"

        regressions.append(
            CostRegression(
                version_a=prev["version"],
                version_b=curr["version"],
                cost_a=prev["cost"],
                cost_b=curr["cost"],
                cost_change_pct=round(cost_change_pct, 2),
                is_regression=is_regression,
                severity=severity,
            )
        )

    return regressions


def format_cost_report(report: CostEvalReport, format: str = "text") -> str:
    """Format a cost evaluation report as text or markdown.

    Args:
        report: CostEvalReport to format.
        format: 'text' or 'markdown'.

    Returns:
        Formatted string.
    """
    if format == "markdown":
        return _format_markdown(report)
    return report.summary()


def _format_markdown(report: CostEvalReport) -> str:
    m = report.metrics
    lines = [
        "# Cost Efficiency Report",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cost | ${m.total_cost:.4f} |",
        f"| Total tasks | {m.total_tasks} |",
        f"| Success rate | {m.success_rate:.1%} |",
        f"| Cost per task | ${m.cost_per_task:.4f} |",
        f"| Cost per success | ${m.cost_per_success:.4f} |",
        f"| Avg tokens/task | {m.avg_input_tokens_per_task:.0f} in / {m.avg_output_tokens_per_task:.0f} out |",
        f"| Cache hit rate | {m.cache_hit_rate:.1%} |",
        f"| Avg cost/1K tokens | ${m.avg_cost_per_1k_tokens:.6f} |",
    ]

    if report.compliance:
        lines += [
            "",
            "## Budget Compliance",
            "",
            "| Budget | Spent | Limit | Util. | Status |",
            "|--------|-------|-------|-------|--------|",
        ]
        for c in report.compliance:
            lines.append(
                f"| {c.budget_name} | ${c.spent_usd:.2f} | ${c.limit_usd:.2f} | "
                f"{c.utilization:.0%} | {c.status} |"
            )

    if report.regressions:
        lines += [
            "",
            "## Cost Regressions",
            "",
        ]
        for r in report.regressions:
            icon = "⚠️" if r.is_regression else "✓"
            lines.append(f"- {icon} {r.description}")

    if report.recommendations:
        lines += [
            "",
            "## Recommendations",
            "",
        ]
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

    return "\n".join(lines)


def _generate_recommendations(
    metrics: CostEfficiencyMetrics,
    compliance: list[BudgetComplianceReport],
) -> list[str]:
    """Generate actionable recommendations from metrics and compliance data."""
    recommendations: list[str] = []

    if metrics.cost_per_success > 0.05:  # >$0.05 per successful task
        recommendations.append(
            f"Cost per success (${metrics.cost_per_success:.4f}) is high. "
            f"Consider model routing to cheaper alternatives for simpler tasks."
        )

    if metrics.cache_hit_rate < 0.10 and metrics.total_tasks > 5:
        recommendations.append(
            f"Cache hit rate is low ({metrics.cache_hit_rate:.1%}). "
            f"Enable prompt caching for repeated system prompts to reduce input costs."
        )

    if metrics.avg_input_tokens_per_task > 4000:
        recommendations.append(
            f"Average input tokens per task ({metrics.avg_input_tokens_per_task:.0f}) is high. "
            f"Review context pruning to trim unnecessary conversation history."
        )

    for c in compliance:
        if c.status in ("EXCEEDED", "EXHAUSTED"):
            recommendations.append(
                f"Budget '{c.budget_name}' exceeded (${c.spent_usd:.2f} / ${c.limit_usd:.2f}). "
                f"Consider increasing limit or implementing hard stops."
            )
        elif c.utilization > 0.85:
            recommendations.append(
                f"Budget '{c.budget_name}' approaching limit ({c.utilization:.0%}). "
                f"Review cost trends and consider adjustments."
            )

    if metrics.failed_tasks > 0 and metrics.success_rate < 0.85:
        recommendations.append(
            f"Success rate is {metrics.success_rate:.1%} with {metrics.failed_tasks} failures. "
            f"Failed tasks still incurred cost — review failure patterns to reduce wasted spend."
        )

    if not recommendations:
        recommendations.append("All cost efficiency metrics are within healthy ranges.")

    return recommendations
