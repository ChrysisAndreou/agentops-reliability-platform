"""
Alerting state models — typed containers for rule definitions, alert
conditions, severity levels, channel routing, and evaluation reports.

Supports three severity levels (critical, warning, info), configurable
conditions with operators and evaluation windows, pluggable alert channels,
and pre-built alert profiles for different operational postures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertSeverity(str, Enum):
    """Alert severity levels — maps to incident response urgency."""
    CRITICAL = "critical"    # Agent is broken; immediate action required
    WARNING = "warning"      # Degradation detected; investigation needed
    INFO = "info"            # Informational threshold crossed


@dataclass
class AlertCondition:
    """A single numeric condition evaluated against agent metrics.

    Conditions are evaluated against a metrics snapshot produced by
    TraceStore.stats() or an EvalRun summary. The `window` field
    defines how many recent runs to include in the evaluation.

    Examples:
        AlertCondition("verification_pass_rate", "lt", 0.80, "last_20_runs")
        AlertCondition("hallucination_rate", "gt", 0.05, "last_50_runs")
        AlertCondition("latency_p95_ms", "gt", 5000, "last_hour")
    """
    metric: str
    operator: str           # lt, gt, lte, gte, eq
    threshold: float
    window: str = "last_50_runs"  # evaluation window identifier

    def evaluate(self, current_value: float) -> bool:
        """Check whether current_value triggers this condition."""
        if self.operator == "lt":
            return current_value < self.threshold
        elif self.operator == "gt":
            return current_value > self.threshold
        elif self.operator == "lte":
            return current_value <= self.threshold
        elif self.operator == "gte":
            return current_value >= self.threshold
        elif self.operator == "eq":
            return abs(current_value - self.threshold) < 1e-9
        return False

    def describe(self, value: float) -> str:
        """Human-readable description of the condition state."""
        return (
            f"{self.metric} ({value:.3f}) {self.operator} {self.threshold:.3f}"
        )


@dataclass
class AlertRule:
    """A named rule that triggers an alert when ALL conditions match.

    Conditions are AND-ed — every condition must evaluate to true for the
    rule to fire. This prevents noisy false positives from single-metric
    fluctuations.

    The `cooldown_seconds` field prevents alert storms by suppressing
    repeated triggers of the same rule within the cooldown window.
    """
    name: str
    description: str
    conditions: list[AlertCondition]
    severity: AlertSeverity = AlertSeverity.WARNING
    channels: list[str] = field(default_factory=lambda: ["console"])
    cooldown_seconds: int = 300  # 5 minutes between repeated alerts
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, metric_values: dict[str, float]) -> tuple[bool, list[AlertCondition]]:
        """Evaluate all conditions against current metric values.

        Returns (triggered, matched_conditions).
        """
        if not self.enabled or not self.conditions:
            return False, []

        matched: list[AlertCondition] = []
        for cond in self.conditions:
            if cond.metric not in metric_values:
                continue
            if cond.evaluate(metric_values[cond.metric]):
                matched.append(cond)

        triggered = len(matched) == len(self.conditions)
        return triggered, matched


@dataclass
class Alert:
    """A triggered alert — created when an AlertRule's conditions all match.

    Carries the full context needed for incident response: which rule fired,
    what current values look like vs thresholds, and when it happened.
    """
    rule_name: str
    severity: AlertSeverity
    message: str
    conditions_matched: list[AlertCondition]
    current_values: dict[str, float]
    triggered_at: str
    run_id: str = ""  # optional: specific run that triggered this

    def to_dict(self) -> dict[str, Any]:
        """Serialize alert to a JSON-serializable dict."""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "conditions": [
                c.describe(self.current_values.get(c.metric, 0))
                for c in self.conditions_matched
            ],
            "current_values": self.current_values,
            "triggered_at": self.triggered_at,
            "run_id": self.run_id,
        }

    def format_message(self) -> str:
        """Format a human-readable alert message."""
        lines = [
            f"[{self.severity.value.upper()}] {self.message}",
            f"Rule: {self.rule_name}",
            f"Time: {self.triggered_at}",
        ]
        for cond in self.conditions_matched:
            val = self.current_values.get(cond.metric, "N/A")
            lines.append(f"  - {cond.describe(val)}")
        return "\n".join(lines)


@dataclass
class AlertChannelConfig:
    """Configuration for a specific alert channel."""
    type: str                    # console, file, webhook
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    # file: {"path": "/var/log/agentops/alerts.jsonl"}
    # webhook: {"url": "https://hooks.slack.com/...", "headers": {...}}


@dataclass
class AlertManagerConfig:
    """Configuration for the AlertManager — rules, channels, profiles."""
    rules: list[AlertRule] = field(default_factory=list)
    channels: list[AlertChannelConfig] = field(default_factory=list)
    default_cooldown_seconds: int = 300
    evaluation_window: str = "last_50_runs"


@dataclass
class AlertReport:
    """Report produced after evaluating all rules against current metrics."""
    alerts_triggered: list[Alert] = field(default_factory=list)
    rules_evaluated: int = 0
    conditions_checked: int = 0
    conditions_matched: int = 0
    profile_name: str = ""

    @property
    def has_critical(self) -> bool:
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts_triggered)

    @property
    def has_warnings(self) -> bool:
        return any(a.severity == AlertSeverity.WARNING for a in self.alerts_triggered)

    @property
    def alert_count_by_severity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.alerts_triggered:
            counts[a.severity.value] = counts.get(a.severity.value, 0) + 1
        return counts

    def to_markdown(self) -> str:
        """Generate a markdown alert evaluation report."""
        lines = [
            "# Alert Evaluation Report",
            "",
            f"**Profile**: {self.profile_name or 'custom'}",
            f"**Rules evaluated**: {self.rules_evaluated}",
            f"**Conditions checked**: {self.conditions_checked}",
            f"**Conditions matched**: {self.conditions_matched}",
            f"**Alerts triggered**: {len(self.alerts_triggered)}",
            "",
        ]

        if self.alerts_triggered:
            lines.append("## Triggered Alerts")
            lines.append("")
            lines.append("| Severity | Rule | Message | Conditions |")
            lines.append("|----------|------|---------|------------|")
            for alert in self.alerts_triggered:
                conditions_str = ", ".join(
                    c.describe(alert.current_values.get(c.metric, 0))
                    for c in alert.conditions_matched
                )
                lines.append(
                    f"| {alert.severity.value} | {alert.rule_name} | "
                    f"{alert.message} | {conditions_str} |"
                )
            lines.append("")
        else:
            lines.append("✅ **No alerts triggered** — all metrics within thresholds.")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-serializable dict."""
        return {
            "profile": self.profile_name,
            "rules_evaluated": self.rules_evaluated,
            "conditions_checked": self.conditions_checked,
            "conditions_matched": self.conditions_matched,
            "alerts_triggered": len(self.alerts_triggered),
            "alerts": [a.to_dict() for a in self.alerts_triggered],
            "by_severity": self.alert_count_by_severity,
        }


# ── Pre-built alert profiles ──────────────────────────────────────────

ALERT_PROFILES: dict[str, AlertManagerConfig] = {}


def _build_profiles() -> None:
    """Build alert profiles — called on module import."""
    from agentops.alerting.rules import BUILT_IN_RULES

    # Aggressive profile — tight thresholds, all rules enabled
    ALERT_PROFILES["strict"] = AlertManagerConfig(
        rules=[r for r in BUILT_IN_RULES],
        channels=[
            AlertChannelConfig(type="console", enabled=True),
            AlertChannelConfig(
                type="file",
                enabled=True,
                config={"path": "alerts_strict.jsonl"},
            ),
        ],
        default_cooldown_seconds=60,
    )

    # Production profile — balanced thresholds, critical + warning rules
    ALERT_PROFILES["production"] = AlertManagerConfig(
        rules=[r for r in BUILT_IN_RULES if r.severity != AlertSeverity.INFO],
        channels=[
            AlertChannelConfig(type="console", enabled=True),
            AlertChannelConfig(
                type="file",
                enabled=True,
                config={"path": "alerts_production.jsonl"},
            ),
        ],
        default_cooldown_seconds=300,
    )

    # Permissive profile — critical-only, relaxed thresholds
    ALERT_PROFILES["permissive"] = AlertManagerConfig(
        rules=[r for r in BUILT_IN_RULES if r.severity == AlertSeverity.CRITICAL],
        channels=[
            AlertChannelConfig(type="console", enabled=True),
        ],
        default_cooldown_seconds=600,
    )

    # Silent profile — no alerts (for development/CI)
    ALERT_PROFILES["silent"] = AlertManagerConfig(
        rules=[],
        channels=[],
    )


_build_profiles()


def get_alert_profile(name: str) -> AlertManagerConfig | None:
    """Get a named alert profile, or None if not found."""
    return ALERT_PROFILES.get(name)
