"""
Production Alerting — threshold-based monitoring, rule evaluation, and
multi-channel notifications for deployed AI agent systems.

Completes the observability loop: collect (traces) → evaluate (benchmarks)
→ visualize (dashboard) → alert (this module). Teams get notified when
agent quality degrades — verification pass rates drop, hallucinations
spike, latency balloons — before users notice.

Module for v0.13: provides alert state models, configurable rule engine,
built-in alert rules for common failure patterns, and pluggable channel
providers (console, file, JSON webhook). All testable without API keys
via deterministic trace/eval data.
"""

from agentops.alerting.state import (
    AlertSeverity,
    AlertCondition,
    AlertRule,
    Alert,
    AlertChannelConfig,
    AlertManagerConfig,
    AlertReport,
    ALERT_PROFILES,
    get_alert_profile,
)
from agentops.alerting.rules import (
    BUILT_IN_RULES,
    evaluate_condition,
    evaluate_rule,
    get_built_in_rules,
)
from agentops.alerting.channels import (
    ConsoleChannel,
    FileChannel,
    WebhookChannel,
    create_channel,
)
from agentops.alerting.manager import AlertManager

__all__ = [
    "AlertSeverity",
    "AlertCondition",
    "AlertRule",
    "Alert",
    "AlertChannelConfig",
    "AlertManagerConfig",
    "AlertReport",
    "ALERT_PROFILES",
    "get_alert_profile",
    "BUILT_IN_RULES",
    "evaluate_condition",
    "evaluate_rule",
    "get_built_in_rules",
    "ConsoleChannel",
    "FileChannel",
    "WebhookChannel",
    "create_channel",
    "AlertManager",
]
