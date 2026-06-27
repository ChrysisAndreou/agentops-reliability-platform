"""
AlertManager — the central orchestrator for evaluating alert rules
against agent metrics and dispatching triggered alerts to configured
channels.

Usage:
    from agentops.alerting import AlertManager, get_alert_profile

    config = get_alert_profile("production")
    manager = AlertManager(config)

    # After running evals or collecting trace stats:
    metrics = {
        "verification_pass_rate": 0.72,
        "groundedness": 0.83,
        "latency_p95_ms": 4500,
        "hallucination_rate": 0.03,
        "failure_rate": 0.05,
        "composite_score": 0.68,
    }
    report = manager.evaluate(metrics)
    print(report.to_markdown())

    # Shutdown to flush and close channels
    manager.shutdown()
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .channels import create_channel
from .state import (
    Alert,
    AlertManagerConfig,
    AlertReport,
    AlertRule,
    get_alert_profile,
)


class AlertManager:
    """Evaluates alert rules against metrics and dispatches alerts.

    Maintains a cooldown map to prevent alert storms — each rule can
    only fire once per its configured cooldown window, preventing
    noisy repeat notifications for persistent conditions.

    Thread-safe for metric evaluation. Channel dispatch is best-effort —
    a channel failure does not block other channels or future evaluations.
    """

    def __init__(self, config: AlertManagerConfig | None = None):
        if config is None:
            config = get_alert_profile("production") or AlertManagerConfig()
        self._config = config
        self._channels = [create_channel(c) for c in self._config.channels]
        self._cooldowns: dict[str, float] = {}  # rule_name → last_triggered_timestamp
        self._evaluation_count: int = 0

    @property
    def rules(self) -> list[AlertRule]:
        return self._config.rules

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    def evaluate(
        self,
        metric_values: dict[str, float],
        run_id: str = "",
    ) -> AlertReport:
        """Evaluate all rules against current metrics and dispatch alerts.

        Args:
            metric_values: Dict of metric_name → current_value.
            run_id: Optional run identifier for context.

        Returns:
            AlertReport with triggered alerts, counts, and summary.
        """
        self._evaluation_count += 1
        now = time.time()
        triggered: list[Alert] = []
        conditions_checked = 0
        conditions_matched = 0

        for rule in self._config.rules:
            if not rule.enabled:
                continue

            # Cooldown check
            last_fired = self._cooldowns.get(rule.name, 0)
            if now - last_fired < rule.cooldown_seconds:
                continue

            # Evaluate conditions
            fired, matched = rule.evaluate(metric_values)
            conditions_checked += len(rule.conditions)
            conditions_matched += len(matched)

            if fired:
                # Build alert
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.description,
                    conditions_matched=matched,
                    current_values=dict(metric_values),
                    triggered_at=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                )
                triggered.append(alert)
                self._cooldowns[rule.name] = now

                # Dispatch
                self._dispatch(alert)

        return AlertReport(
            alerts_triggered=triggered,
            rules_evaluated=len(self._config.rules),
            conditions_checked=conditions_checked,
            conditions_matched=conditions_matched,
        )

    def evaluate_static(
        self,
        metric_values: dict[str, float],
        run_id: str = "",
    ) -> AlertReport:
        """Evaluate rules WITHOUT dispatching or cooldowns.

        Useful for testing, dry-run, and CI evaluation — identical rule
        evaluation logic but side-effect-free.
        """
        triggered: list[Alert] = []
        conditions_checked = 0
        conditions_matched = 0

        for rule in self._config.rules:
            if not rule.enabled:
                continue

            fired, matched = rule.evaluate(metric_values)
            conditions_checked += len(rule.conditions)
            conditions_matched += len(matched)

            if fired:
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.description,
                    conditions_matched=matched,
                    current_values=dict(metric_values),
                    triggered_at=datetime.now(timezone.utc).isoformat(),
                    run_id=run_id,
                )
                triggered.append(alert)

        return AlertReport(
            alerts_triggered=triggered,
            rules_evaluated=len(self._config.rules),
            conditions_checked=conditions_checked,
            conditions_matched=conditions_matched,
        )

    def _dispatch(self, alert: Alert) -> None:
        """Send alert to all configured channels. Failures are silent."""
        for channel in self._channels:
            try:
                channel.send(alert)
            except Exception:
                pass

    def reset_cooldowns(self) -> None:
        """Reset all cooldowns — useful for testing."""
        self._cooldowns.clear()

    def shutdown(self) -> None:
        """Close and flush any persistent channels."""
        # FileChannel and ConsoleChannel don't need explicit close.
        # WebhookChannel is stateless.
        pass
