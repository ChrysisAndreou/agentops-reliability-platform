"""
Alert channel providers — pluggable notification backends for dispatching
alerts to console, files, webhook endpoints, Slack, Discord, and email.

Each channel implements a simple send() interface. Channels are safe to
instantiate even when external dependencies (network, filesystem, SMTP)
are unavailable — they degrade gracefully and log errors.

Extensible: add new channels by subclassing the implicit interface and
registering in create_channel().

v0.16: Added SlackBlockChannel (Block Kit formatting), DiscordEmbedChannel
(embed objects), and EmailChannel (SMTP with HTML/text multipart).
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from .state import Alert, AlertChannelConfig


class ConsoleChannel:
    """Writes alerts to stdout with ANSI color formatting.

    Critical alerts appear in red, warnings in yellow, info in blue.
    Safe for CI/CD — no filesystem or network dependencies.
    """

    ANSI_RED = "\033[91m"
    ANSI_YELLOW = "\033[93m"
    ANSI_BLUE = "\033[94m"
    ANSI_RESET = "\033[0m"
    ANSI_BOLD = "\033[1m"

    _COLORS: dict[str, str] = {
        "critical": ANSI_RED,
        "warning": ANSI_YELLOW,
        "info": ANSI_BLUE,
    }

    def __init__(self, config: AlertChannelConfig | None = None):
        self._enabled = config.enabled if config else True

    def send(self, alert: Alert) -> bool:
        """Print a formatted alert to stdout."""
        if not self._enabled:
            return False

        color = self._COLORS.get(alert.severity.value, "")
        reset = self.ANSI_RESET
        bold = self.ANSI_BOLD

        header = f"{color}{bold}[{alert.severity.value.upper()}]{reset} {bold}{alert.rule_name}{reset}"
        print(f"\n{header}")
        print(f"  Message: {alert.message}")
        print(f"  Time:    {alert.triggered_at}")

        for cond in alert.conditions_matched:
            val = alert.current_values.get(cond.metric, "N/A")
            print(f"  Condition: {cond.describe(val)}")

        if alert.run_id:
            print(f"  Run ID:  {alert.run_id}")

        print(f"{color}{'─' * 60}{reset}\n")
        return True


class FileChannel:
    """Appends alerts to a JSON Lines file for persistent audit logging.

    Each alert is written as a single JSON object on its own line.
    The file is created if it doesn't exist; parent directories are
    created automatically.

    Safe for concurrent writers — each write is an atomic append line.
    """

    def __init__(self, config: AlertChannelConfig):
        self._enabled = config.enabled if config else True
        self._path = Path(config.config.get("path", "alerts.jsonl")) if config else Path("alerts.jsonl")

    def send(self, alert: Alert) -> bool:
        """Append alert as a JSON line to the alerts file."""
        if not self._enabled:
            return False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
            return True
        except (OSError, IOError):
            return False

    @property
    def path(self) -> Path:
        return self._path


class WebhookChannel:
    """Sends alerts as HTTP POST to a webhook endpoint.

    Supports Slack incoming webhooks, Discord webhooks, and custom
    JSON endpoints. Alert payload is serialized as JSON with
    Content-Type: application/json.

    For Slack-specific Block Kit formatting, use SlackBlockChannel.
    For Discord embed formatting, use DiscordEmbedChannel.

    Timeout: 5 seconds per request. Failures are logged but not raised —
    alert delivery is best-effort, not guaranteed.
    """

    DEFAULT_TIMEOUT = 5  # seconds

    def __init__(self, config: AlertChannelConfig):
        self._enabled = config.enabled if config else True
        self._url = config.config.get("url", "") if config else ""
        self._headers = config.config.get("headers", {}) if config else {}
        self._headers.setdefault("Content-Type", "application/json")
        self._timeout = config.config.get("timeout", self.DEFAULT_TIMEOUT) if config else self.DEFAULT_TIMEOUT

    def send(self, alert: Alert) -> bool:
        """POST alert JSON to the webhook endpoint."""
        if not self._enabled or not self._url:
            return False

        payload = alert.to_dict()
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers=self._headers,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout)
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Slack Block Kit channel
# ═══════════════════════════════════════════════════════════════════════

_SLACK_SEVERITY_COLORS: dict[str, str] = {
    "critical": "#FF0000",
    "warning": "#FFA500",
    "info": "#0066FF",
}

_SLACK_SEVERITY_EMOJI: dict[str, str] = {
    "critical": ":red_circle:",
    "warning": ":warning:",
    "info": ":information_source:",
}


class SlackBlockChannel:
    """Sends alerts to Slack using Block Kit formatting.

    Formats alerts as Slack Block Kit messages with a colored header,
    structured sections for message and conditions, and a context footer
    with timestamp. Designed for Slack incoming webhooks.

    Block Kit reference: https://api.slack.com/block-kit

    Example config:
        AlertChannelConfig(
            type="slack",
            enabled=True,
            config={"webhook_url": "https://hooks.slack.com/services/T.../B.../..."}
        )
    """

    DEFAULT_TIMEOUT = 5

    def __init__(self, config: AlertChannelConfig):
        self._enabled = config.enabled if config else True
        self._url = config.config.get("webhook_url", "") if config else ""
        self._timeout = config.config.get("timeout", self.DEFAULT_TIMEOUT) if config else self.DEFAULT_TIMEOUT

    def send(self, alert: Alert) -> bool:
        """Format alert as Block Kit and POST to Slack webhook."""
        if not self._enabled or not self._url:
            return False

        blocks = self._build_blocks(alert)
        payload = {"blocks": blocks}

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout)
            return True
        except Exception:
            return False

    def _build_blocks(self, alert: Alert) -> list[dict[str, Any]]:
        """Build Slack Block Kit blocks for an alert."""
        sev = alert.severity.value
        emoji = _SLACK_SEVERITY_EMOJI.get(sev, ":bell:")
        color = _SLACK_SEVERITY_COLORS.get(sev, "#808080")

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} [{sev.upper()}] {alert.rule_name}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert.message}*",
                },
            },
        ]

        # Conditions section
        if alert.conditions_matched:
            conditions_text = "\n".join(
                f"• {c.describe(alert.current_values.get(c.metric, 0))}"
                for c in alert.conditions_matched
            )
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Conditions triggered:*\n{conditions_text}",
                },
            })

        # Metrics summary
        if alert.current_values:
            metrics_lines = []
            for key, val in sorted(alert.current_values.items()):
                metrics_lines.append(f"• `{key}`: {val:.4f}" if isinstance(val, float) else f"• `{key}`: {val}")
            first_chunk = "\n".join(metrics_lines[:6])
            if first_chunk:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Current metrics:*\n{first_chunk}",
                    },
                })

        # Context footer
        footer_elements = [{"type": "mrkdwn", "text": f":clock3: {alert.triggered_at}"}]
        if alert.run_id:
            footer_elements.insert(0, {"type": "mrkdwn", "text": f"Run: `{alert.run_id}`"})
        blocks.append({"type": "context", "elements": footer_elements})

        # Divider
        blocks.append({"type": "divider"})

        return blocks


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Discord embed channel
# ═══════════════════════════════════════════════════════════════════════

_DISCORD_SEVERITY_COLORS: dict[str, int] = {
    "critical": 0xFF0000,  # red
    "warning": 0xFFA500,   # orange
    "info": 0x0066FF,      # blue
}


class DiscordEmbedChannel:
    """Sends alerts to Discord using rich embed formatting.

    Formats alerts as Discord embed objects with colored sidebars,
    structured fields for conditions and metrics, and a timestamp
    footer. Designed for Discord webhooks.

    Discord embed reference: https://discord.com/developers/docs/resources/channel#embed-object

    Example config:
        AlertChannelConfig(
            type="discord",
            enabled=True,
            config={"webhook_url": "https://discord.com/api/webhooks/..."}
        )
    """

    DEFAULT_TIMEOUT = 5

    def __init__(self, config: AlertChannelConfig):
        self._enabled = config.enabled if config else True
        self._url = config.config.get("webhook_url", "") if config else ""
        self._timeout = config.config.get("timeout", self.DEFAULT_TIMEOUT) if config else self.DEFAULT_TIMEOUT

    def send(self, alert: Alert) -> bool:
        """Format alert as Discord embed and POST to webhook."""
        if not self._enabled or not self._url:
            return False

        embed = self._build_embed(alert)
        payload = {"embeds": [embed]}

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout)
            return True
        except Exception:
            return False

    def _build_embed(self, alert: Alert) -> dict[str, Any]:
        """Build a Discord embed object for an alert."""
        sev = alert.severity.value
        color = _DISCORD_SEVERITY_COLORS.get(sev, 0x808080)

        embed: dict[str, Any] = {
            "title": f"[{sev.upper()}] {alert.rule_name}",
            "description": alert.message,
            "color": color,
            "timestamp": alert.triggered_at,
        }

        # Fields for conditions
        if alert.conditions_matched:
            conditions_str = "\n".join(
                f"• {c.describe(alert.current_values.get(c.metric, 0))}"
                for c in alert.conditions_matched
            )
            embed.setdefault("fields", []).append({
                "name": "Conditions Triggered",
                "value": conditions_str[:1024],
                "inline": False,
            })

        # Fields for metrics
        if alert.current_values:
            metric_items = []
            for key, val in sorted(alert.current_values.items()):
                formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                metric_items.append(f"`{key}`: {formatted}")
            # Discord limits: max 25 fields, 1024 chars per field value
            chunks = [metric_items[i:i + 5] for i in range(0, len(metric_items), 5)]
            for idx, chunk in enumerate(chunks):
                embed.setdefault("fields", []).append({
                    "name": "Current Metrics" if idx == 0 else "Metrics (cont.)",
                    "value": "\n".join(chunk)[:1024],
                    "inline": True,
                })

        # Footer
        footer_text = f"AgentOps Reliability Platform"
        if alert.run_id:
            footer_text += f" • Run: {alert.run_id}"
        embed["footer"] = {"text": footer_text}

        return embed

    def build_embed(self, alert: Alert) -> dict[str, Any]:
        """Public method: build embed dict without sending. For testing."""
        return self._build_embed(alert)


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Email channel (SMTP)
# ═══════════════════════════════════════════════════════════════════════

_EMAIL_SEVERITY_SUBJECT_PREFIX: dict[str, str] = {
    "critical": "CRITICAL",
    "warning": "WARNING",
    "info": "INFO",
}


def _fmt_val(v: Any) -> str:
    """Format a metric value for display — floats to 4 decimal places."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


class EmailChannel:
    """Sends alerts via SMTP email with HTML and plaintext body.

    Uses Python's stdlib smtplib — no external dependencies. Supports
    TLS (STARTTLS) for secure delivery. Falls back gracefully if SMTP
    server is unreachable.

    SMTP configuration is passed through AlertChannelConfig.config dict:
        smtp_host: str        — SMTP server hostname (default: localhost)
        smtp_port: int        — SMTP server port (default: 587)
        use_tls: bool         — Use STARTTLS (default: True)
        username: str | None  — SMTP auth username (optional)
        password: str | None  — SMTP auth password (optional)
        from_addr: str        — From address (default: agentops@localhost)
        to_addrs: list[str]   — List of recipient addresses (required)
        subject_prefix: str   — Custom prefix for email subject

    Example config:
        AlertChannelConfig(
            type="email",
            enabled=True,
            config={
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "username": "alerts@example.com",
                "password": "app-password-here",
                "from_addr": "alerts@example.com",
                "to_addrs": ["oncall@example.com", "lead@example.com"],
            }
        )
    """

    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 587
    DEFAULT_TIMEOUT = 10  # seconds for SMTP connection

    def __init__(self, config: AlertChannelConfig):
        self._enabled = config.enabled if config else True
        cfg = config.config if config else {}

        self._smtp_host = cfg.get("smtp_host", self.DEFAULT_HOST)
        self._smtp_port = cfg.get("smtp_port", self.DEFAULT_PORT)
        self._use_tls = cfg.get("use_tls", True)
        self._username = cfg.get("username")
        self._password = cfg.get("password")
        self._from_addr = cfg.get("from_addr", "agentops@localhost")
        self._to_addrs: list[str] = cfg.get("to_addrs", [])
        self._subject_prefix = cfg.get("subject_prefix", "")

    def send(self, alert: Alert) -> bool:
        """Send alert as email via SMTP."""
        if not self._enabled or not self._to_addrs:
            return False

        msg = self._build_message(alert)

        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=self.DEFAULT_TIMEOUT) as smtp:
                if self._use_tls:
                    smtp.starttls()
                if self._username and self._password:
                    smtp.login(self._username, self._password)
                smtp.send_message(msg)
            return True
        except Exception:
            return False

    def _build_message(self, alert: Alert) -> MIMEMultipart:
        """Build MIME multipart email with HTML and plaintext parts."""
        sev = alert.severity.value
        prefix = _EMAIL_SEVERITY_SUBJECT_PREFIX.get(sev, "ALERT")
        subject = f"[AgentOps {prefix}] {alert.rule_name}"
        if self._subject_prefix:
            subject = f"{self._subject_prefix} {subject}"
        if alert.run_id:
            subject += f" (Run: {alert.run_id[:12]})"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_addr
        msg["To"] = ", ".join(self._to_addrs)

        # Plaintext part
        text_body = self._build_text_body(alert)
        msg.attach(MIMEText(text_body, "plain", "utf-8"))

        # HTML part
        html_body = self._build_html_body(alert)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        return msg

    def _build_text_body(self, alert: Alert) -> str:
        """Build plaintext email body."""
        lines = [
            f"AgentOps Reliability Platform — {alert.severity.value.upper()} Alert",
            "=" * 60,
            "",
            f"Rule:     {alert.rule_name}",
            f"Message:  {alert.message}",
            f"Time:     {alert.triggered_at}",
        ]
        if alert.run_id:
            lines.append(f"Run ID:   {alert.run_id}")

        if alert.conditions_matched:
            lines.append("")
            lines.append("Conditions Triggered:")
            for cond in alert.conditions_matched:
                val = alert.current_values.get(cond.metric, "N/A")
                lines.append(f"  - {cond.describe(val)}")

        if alert.current_values:
            lines.append("")
            lines.append("Current Metrics:")
            for key, val in sorted(alert.current_values.items()):
                formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                lines.append(f"  {key}: {formatted}")

        lines.append("")
        lines.append("--")
        lines.append("Sent by AgentOps Reliability Platform")
        return "\n".join(lines)

    def _build_html_body(self, alert: Alert) -> str:
        """Build HTML email body."""
        sev = alert.severity.value
        color = _SLACK_SEVERITY_COLORS.get(sev, "#808080")

        conditions_html = ""
        if alert.conditions_matched:
            items = "".join(
                f"<li>{c.describe(alert.current_values.get(c.metric, 0))}</li>"
                for c in alert.conditions_matched
            )
            conditions_html = f"""
                <h3 style="color:#555;">Conditions Triggered</h3>
                <ul>{items}</ul>
            """

        metrics_html = ""
        if alert.current_values:
            rows = "".join(
                f"<tr><td style='padding:4px 12px;'><code>{k}</code></td>"
                f"<td style='padding:4px 12px;'>{_fmt_val(v)}</td></tr>"
                for k, v in sorted(alert.current_values.items())
            )
            metrics_html = f"""
                <h3 style="color:#555;">Current Metrics</h3>
                <table style="border-collapse:collapse;border:1px solid #ddd;">
                    <tr style="background:#f5f5f5;">
                        <th style="padding:6px 12px;text-align:left;">Metric</th>
                        <th style="padding:6px 12px;text-align:left;">Value</th>
                    </tr>
                    {rows}
                </table>
            """

        run_id_html = f"<p><strong>Run ID:</strong> <code>{alert.run_id}</code></p>" if alert.run_id else ""

        return f"""<!DOCTYPE html>
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:640px;margin:0 auto;padding:20px;color:#333;">
    <div style="border-left:4px solid {color};padding:0 16px;margin-bottom:24px;">
        <h2 style="color:{color};margin:0 0 8px;">[{sev.upper()}] {alert.rule_name}</h2>
        <p style="font-size:15px;margin:0 0 12px;">{alert.message}</p>
    </div>
    <p><strong>Time:</strong> {alert.triggered_at}</p>
    {run_id_html}
    {conditions_html}
    {metrics_html}
    <hr style="border:none;border-top:1px solid #eee;margin:24px 0;">
    <p style="color:#999;font-size:12px;">Sent by AgentOps Reliability Platform</p>
</body>
</html>"""

    def build_message(self, alert: Alert) -> MIMEMultipart:
        """Public method: build MIME message without sending. For testing."""
        return self._build_message(alert)


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════


class _NoOpChannel:
    """Silent channel for disabled or unknown channel types."""
    def send(self, alert: Alert) -> bool:
        return False


def create_channel(config: AlertChannelConfig) -> Any:
    """Factory: create a channel instance from its config.

    Supported types: console, file, webhook, slack, discord, email.

    Args:
        config: AlertChannelConfig with type and per-channel settings.

    Returns:
        Channel instance appropriate for the type.

    Raises:
        ValueError: if the channel type is unknown.
    """
    if not config.enabled:
        return _NoOpChannel()

    channel_type = config.type

    if channel_type == "console":
        return ConsoleChannel(config)
    elif channel_type == "file":
        return FileChannel(config)
    elif channel_type == "webhook":
        return WebhookChannel(config)
    elif channel_type == "slack":
        return SlackBlockChannel(config)
    elif channel_type == "discord":
        return DiscordEmbedChannel(config)
    elif channel_type == "email":
        return EmailChannel(config)
    else:
        raise ValueError(f"Unknown alert channel type: {channel_type}")
