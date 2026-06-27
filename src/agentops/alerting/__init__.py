"""
Production Alerting — threshold-based monitoring, rule evaluation, and
multi-channel notifications for deployed AI agent systems.

Completes the observability loop: collect (traces) → evaluate (benchmarks)
→ visualize (dashboard) → alert (this module). Teams get notified when
agent quality degrades — verification pass rates drop, hallucinations
spike, latency balloons — before users notice.

Provider for v0.16: also Slack Block Kit, Discord embeds, and SMTP email.
All testable without API keys via deterministic trace/eval data.
"""

from agentops.alerting.channels import (
    ConsoleChannel,
    DiscordEmbedChannel,
    EmailChannel,
    FileChannel,
    SlackBlockChannel,
    WebhookChannel,
    create_channel,
)
from agentops.alerting.manager import AlertManager
from agentops.alerting.rules import (
    BUILT_IN_RULES,
    evaluate_condition,
    evaluate_rule,
    get_built_in_rules,
)
from agentops.alerting.state import (
    ALERT_PROFILES,
    Alert,
    AlertChannelConfig,
    AlertCondition,
    AlertManagerConfig,
    AlertReport,
    AlertRule,
    AlertSeverity,
    get_alert_profile,
)

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
    "SlackBlockChannel",
    "DiscordEmbedChannel",
    "EmailChannel",
    "create_channel",
    "AlertManager",
]
