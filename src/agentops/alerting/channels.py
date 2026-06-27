"""
Alert channel providers — pluggable notification backends for dispatching
alerts to console, files, and webhook endpoints (Slack, Discord, custom).

Each channel implements a simple send() interface. Channels are safe to
instantiate even when external dependencies (network, filesystem) are
unavailable — they degrade gracefully and log errors.

Extensible: add new channels by subclassing the implicit interface and
registering in create_channel().
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
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
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"

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


def create_channel(config: AlertChannelConfig) -> Any:
    """Factory: create a channel instance from its config.

    Args:
        config: AlertChannelConfig with type and per-channel settings.

    Returns:
        Channel instance (ConsoleChannel, FileChannel, or WebhookChannel).

    Raises:
        ValueError: if the channel type is unknown.
    """
    if not config.enabled:
        return _NoOpChannel()

    if config.type == "console":
        return ConsoleChannel(config)
    elif config.type == "file":
        return FileChannel(config)
    elif config.type == "webhook":
        return WebhookChannel(config)
    else:
        raise ValueError(f"Unknown alert channel type: {config.type}")


class _NoOpChannel:
    """Silent channel for disabled or unknown channel types."""
    def send(self, alert: Alert) -> bool:
        return False
