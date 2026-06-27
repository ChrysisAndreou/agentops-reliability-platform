"""
Built-in alert rules for common agent reliability failure patterns.

Provides 8 production-ready rules covering verification quality,
groundedness, latency, failure rates, hallucination, tool reliability,
citation quality, and memory recall degradation. Rules are configurable,
composable, and testable without API keys.

The rule engine evaluates AlertConditions against metric snapshots.
Each rule fires when ALL its conditions are simultaneously true —
this AND-semantics prevents noisy single-metric false positives.
"""

from __future__ import annotations

from .state import AlertCondition, AlertRule, AlertSeverity


def evaluate_condition(condition: AlertCondition, current_value: float) -> bool:
    """Evaluate a single condition against a metric value."""
    return condition.evaluate(current_value)


def evaluate_rule(rule: AlertRule, metric_values: dict[str, float]) -> tuple[bool, list[AlertCondition]]:
    """Evaluate all conditions in a rule against current metric values.

    Returns (triggered, matched_conditions).
    """
    return rule.evaluate(metric_values)


# ── Built-in alert rules ──────────────────────────────────────────────


BUILT_IN_RULES: list[AlertRule] = [
    AlertRule(
        name="verification-drop-critical",
        description="Verification pass rate dropped below 60% — agent is rejecting most outputs",
        conditions=[
            AlertCondition("verification_pass_rate", "lt", 0.60, "last_20_runs"),
        ],
        severity=AlertSeverity.CRITICAL,
        channels=["console", "file", "webhook"],
        cooldown_seconds=120,
    ),
    AlertRule(
        name="verification-drop-warning",
        description="Verification pass rate dropped below 75% — investigate agent quality",
        conditions=[
            AlertCondition("verification_pass_rate", "lt", 0.75, "last_20_runs"),
        ],
        severity=AlertSeverity.WARNING,
        channels=["console", "file"],
        cooldown_seconds=300,
    ),
    AlertRule(
        name="hallucination-spike-critical",
        description="Hallucination rate exceeded 15% — agent is fabricating claims at dangerous levels",
        conditions=[
            AlertCondition("hallucination_rate", "gt", 0.15, "last_50_runs"),
        ],
        severity=AlertSeverity.CRITICAL,
        channels=["console", "file", "webhook"],
        cooldown_seconds=120,
    ),
    AlertRule(
        name="groundedness-drop-warning",
        description="Groundedness ratio dropped below 70% — agent claims are increasingly unsupported",
        conditions=[
            AlertCondition("groundedness", "lt", 0.70, "last_30_runs"),
        ],
        severity=AlertSeverity.WARNING,
        channels=["console", "file"],
        cooldown_seconds=300,
    ),
    AlertRule(
        name="latency-spike-warning",
        description="P95 latency exceeded 10 seconds — agent is becoming unresponsive",
        conditions=[
            AlertCondition("latency_p95_ms", "gt", 10_000, "last_20_runs"),
        ],
        severity=AlertSeverity.WARNING,
        channels=["console", "file"],
        cooldown_seconds=300,
    ),
    AlertRule(
        name="failure-rate-critical",
        description="Agent failure rate exceeded 20% — reliability emergency",
        conditions=[
            AlertCondition("failure_rate", "gt", 0.20, "last_50_runs"),
        ],
        severity=AlertSeverity.CRITICAL,
        channels=["console", "file", "webhook"],
        cooldown_seconds=120,
    ),
    AlertRule(
        name="tool-failure-warning",
        description="Tool call failure rate exceeded 10% — tools may be degraded",
        conditions=[
            AlertCondition("tool_failure_rate", "gt", 0.10, "last_50_runs"),
        ],
        severity=AlertSeverity.WARNING,
        channels=["console", "file"],
        cooldown_seconds=300,
    ),
    AlertRule(
        name="citation-quality-info",
        description="Citation quality dropped below 80% — not urgent but worth reviewing",
        conditions=[
            AlertCondition("citation_quality", "lt", 0.80, "last_100_runs"),
        ],
        severity=AlertSeverity.INFO,
        channels=["console", "file"],
        cooldown_seconds=600,
    ),
    AlertRule(
        name="composite-quality-warning",
        description="Composite quality score dropped below 0.50 — agent quality is degrading across multiple dimensions",
        conditions=[
            AlertCondition("composite_score", "lt", 0.50, "last_30_runs"),
        ],
        severity=AlertSeverity.WARNING,
        channels=["console", "file", "webhook"],
        cooldown_seconds=300,
    ),
    # Multi-condition rule: only fires when BOTH verification AND groundedness degrade
    AlertRule(
        name="multi-dimensional-degradation-critical",
        description="Both verification rate AND groundedness dropped — systemic agent degradation",
        conditions=[
            AlertCondition("verification_pass_rate", "lt", 0.70, "last_20_runs"),
            AlertCondition("groundedness", "lt", 0.65, "last_20_runs"),
        ],
        severity=AlertSeverity.CRITICAL,
        channels=["console", "file", "webhook"],
        cooldown_seconds=120,
    ),
    # Memory-specific alert
    AlertRule(
        name="memory-degradation-warning",
        description="Agent memory recall F1 dropped below 70% — agent is forgetting context",
        conditions=[
            AlertCondition("memory_f1", "lt", 0.70, "last_20_runs"),
        ],
        severity=AlertSeverity.WARNING,
        channels=["console", "file"],
        cooldown_seconds=300,
    ),
]


def get_built_in_rules() -> list[AlertRule]:
    """Return a copy of the built-in alert rules for safe consumer mutation."""
    return [AlertRule(
        name=r.name,
        description=r.description,
        conditions=[AlertCondition(c.metric, c.operator, c.threshold, c.window) for c in r.conditions],
        severity=r.severity,
        channels=list(r.channels),
        cooldown_seconds=r.cooldown_seconds,
        enabled=r.enabled,
        metadata=dict(r.metadata),
    ) for r in BUILT_IN_RULES]
