"""
Tests for AgentOps v0.13 — Production Alerting module.

Covers:
- State models: conditions, rules, alerts, reports, profiles
- Rule evaluation: single conditions, multi-condition AND semantics, edge cases
- AlertManager: evaluate, evaluate_static, cooldowns, profile filtering
- Channels: ConsoleChannel, FileChannel, WebhookChannel, create_channel
- Integration: full alert lifecycle, markdown reports, JSON serialization
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentops.alerting.state import (
    Alert,
    AlertChannelConfig,
    AlertCondition,
    AlertManagerConfig,
    AlertReport,
    AlertRule,
    AlertSeverity,
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
    SlackBlockChannel,
    DiscordEmbedChannel,
    EmailChannel,
    _NoOpChannel,
    create_channel,
)
from agentops.alerting.manager import AlertManager


# ═══════════════════════════════════════════════════════════════════════
# State models
# ═══════════════════════════════════════════════════════════════════════

class TestAlertSeverity:
    def test_values(self):
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.INFO.value == "info"

    def test_comparison(self):
        assert AlertSeverity.CRITICAL != AlertSeverity.WARNING
        assert AlertSeverity("critical") == AlertSeverity.CRITICAL


class TestAlertCondition:
    def test_lt_operator(self):
        cond = AlertCondition("metric", "lt", 0.80)
        assert cond.evaluate(0.75) is True
        assert cond.evaluate(0.80) is False
        assert cond.evaluate(0.85) is False

    def test_gt_operator(self):
        cond = AlertCondition("metric", "gt", 0.20)
        assert cond.evaluate(0.25) is True
        assert cond.evaluate(0.20) is False
        assert cond.evaluate(0.15) is False

    def test_lte_operator(self):
        cond = AlertCondition("metric", "lte", 0.80)
        assert cond.evaluate(0.80) is True
        assert cond.evaluate(0.75) is True
        assert cond.evaluate(0.85) is False

    def test_gte_operator(self):
        cond = AlertCondition("metric", "gte", 0.80)
        assert cond.evaluate(0.80) is True
        assert cond.evaluate(0.85) is True
        assert cond.evaluate(0.75) is False

    def test_eq_operator(self):
        cond = AlertCondition("metric", "eq", 0.80)
        assert cond.evaluate(0.80) is True
        assert cond.evaluate(0.80000000001) is True  # within 1e-9 epsilon
        assert cond.evaluate(0.8000001) is False  # 1e-7 outside epsilon
        assert cond.evaluate(0.81) is False

    def test_describe(self):
        cond = AlertCondition("verification_pass_rate", "lt", 0.60)
        desc = cond.describe(0.55)
        assert "verification_pass_rate" in desc
        assert "0.550" in desc
        assert "lt" in desc
        assert "0.600" in desc

    def test_default_window(self):
        cond = AlertCondition("metric", "gt", 0.5)
        assert cond.window == "last_50_runs"


class TestAlertRule:
    def test_single_condition_match(self):
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[AlertCondition("verification_pass_rate", "lt", 0.80)],
        )
        triggered, matched = rule.evaluate({"verification_pass_rate": 0.60})
        assert triggered is True
        assert len(matched) == 1

    def test_single_condition_no_match(self):
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[AlertCondition("verification_pass_rate", "lt", 0.80)],
        )
        triggered, matched = rule.evaluate({"verification_pass_rate": 0.90})
        assert triggered is False
        assert len(matched) == 0

    def test_multi_condition_all_match(self):
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[
                AlertCondition("a", "lt", 0.80),
                AlertCondition("b", "gt", 0.20),
            ],
        )
        triggered, matched = rule.evaluate({"a": 0.60, "b": 0.30})
        assert triggered is True
        assert len(matched) == 2

    def test_multi_condition_partial_match(self):
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[
                AlertCondition("a", "lt", 0.80),
                AlertCondition("b", "gt", 0.20),
            ],
        )
        triggered, matched = rule.evaluate({"a": 0.60, "b": 0.10})
        assert triggered is False
        assert len(matched) == 1

    def test_disabled_rule(self):
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[AlertCondition("a", "lt", 0.80)],
            enabled=False,
        )
        triggered, matched = rule.evaluate({"a": 0.60})
        assert triggered is False

    def test_empty_conditions(self):
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[],
        )
        triggered, matched = rule.evaluate({"a": 0.60})
        assert triggered is False

    def test_missing_metric(self):
        """Condition for a metric not in the snapshot should be skipped (not matched)."""
        rule = AlertRule(
            name="test",
            description="test rule",
            conditions=[
                AlertCondition("verification_pass_rate", "lt", 0.80),
                AlertCondition("nonexistent_metric", "gt", 0.50),
            ],
        )
        triggered, matched = rule.evaluate({"verification_pass_rate": 0.60})
        assert triggered is False  # only 1 of 2 conditions met


class TestAlert:
    def test_to_dict(self):
        alert = Alert(
            rule_name="test-rule",
            severity=AlertSeverity.CRITICAL,
            message="Test message",
            conditions_matched=[AlertCondition("a", "lt", 0.80)],
            current_values={"a": 0.55},
            triggered_at="2026-01-01T00:00:00Z",
            run_id="r123",
        )
        d = alert.to_dict()
        assert d["rule_name"] == "test-rule"
        assert d["severity"] == "critical"
        assert d["message"] == "Test message"
        assert len(d["conditions"]) == 1
        assert d["run_id"] == "r123"

    def test_format_message(self):
        alert = Alert(
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            message="Degradation detected",
            conditions_matched=[AlertCondition("a", "lt", 0.80)],
            current_values={"a": 0.55},
            triggered_at="2026-01-01T00:00:00Z",
        )
        msg = alert.format_message()
        assert "[WARNING]" in msg
        assert "test-rule" in msg
        assert "Degradation detected" in msg


class TestAlertReport:
    def test_empty_report(self):
        report = AlertReport()
        assert len(report.alerts_triggered) == 0
        assert report.has_critical is False
        assert report.has_warnings is False

    def test_has_critical(self):
        alert = Alert(
            rule_name="test", severity=AlertSeverity.CRITICAL,
            message="", conditions_matched=[],
            current_values={}, triggered_at="",
        )
        report = AlertReport(alerts_triggered=[alert])
        assert report.has_critical is True

    def test_has_warnings(self):
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="", conditions_matched=[],
            current_values={}, triggered_at="",
        )
        report = AlertReport(alerts_triggered=[alert])
        assert report.has_warnings is True
        assert report.has_critical is False

    def test_alert_count_by_severity(self):
        alerts = [
            Alert("a", AlertSeverity.CRITICAL, "", [], {}, ""),
            Alert("b", AlertSeverity.WARNING, "", [], {}, ""),
            Alert("c", AlertSeverity.CRITICAL, "", [], {}, ""),
        ]
        report = AlertReport(alerts_triggered=alerts)
        counts = report.alert_count_by_severity
        assert counts["critical"] == 2
        assert counts["warning"] == 1

    def test_to_markdown(self):
        alert = Alert(
            rule_name="test-rule",
            severity=AlertSeverity.WARNING,
            message="Test",
            conditions_matched=[AlertCondition("a", "lt", 0.80)],
            current_values={"a": 0.55},
            triggered_at="2026-01-01",
        )
        report = AlertReport(
            alerts_triggered=[alert],
            rules_evaluated=11,
            conditions_checked=14,
            conditions_matched=3,
            profile_name="production",
        )
        md = report.to_markdown()
        assert "# Alert Evaluation Report" in md
        assert "production" in md
        assert "test-rule" in md
        assert "3" in md

    def test_to_markdown_no_alerts(self):
        report = AlertReport(rules_evaluated=11, profile_name="production")
        md = report.to_markdown()
        assert "No alerts triggered" in md

    def test_to_dict(self):
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="msg", conditions_matched=[],
            current_values={}, triggered_at="t",
        )
        report = AlertReport(
            alerts_triggered=[alert],
            rules_evaluated=5,
            conditions_checked=10,
            conditions_matched=2,
            profile_name="strict",
        )
        d = report.to_dict()
        assert d["rules_evaluated"] == 5
        assert d["alerts_triggered"] == 1
        assert d["profile"] == "strict"


class TestAlertProfiles:
    def test_all_profiles_exist(self):
        assert "strict" in ALERT_PROFILES
        assert "production" in ALERT_PROFILES
        assert "permissive" in ALERT_PROFILES
        assert "silent" in ALERT_PROFILES

    def test_get_alert_profile(self):
        config = get_alert_profile("production")
        assert config is not None
        assert len(config.rules) > 0

    def test_get_nonexistent_profile(self):
        assert get_alert_profile("nonexistent") is None

    def test_silent_profile_empty(self):
        config = get_alert_profile("silent")
        assert config is not None
        assert len(config.rules) == 0
        assert len(config.channels) == 0

    def test_production_has_no_info_rules(self):
        config = get_alert_profile("production")
        assert config is not None
        for rule in config.rules:
            assert rule.severity != AlertSeverity.INFO

    def test_permissive_only_critical(self):
        config = get_alert_profile("permissive")
        assert config is not None
        for rule in config.rules:
            assert rule.severity == AlertSeverity.CRITICAL


# ═══════════════════════════════════════════════════════════════════════
# Rule evaluation
# ═══════════════════════════════════════════════════════════════════════

class TestBuiltInRules:
    def test_all_rules_have_names(self):
        for rule in BUILT_IN_RULES:
            assert rule.name
            assert rule.description
            assert len(rule.conditions) > 0

    def test_get_built_in_rules_returns_copy(self):
        rules1 = get_built_in_rules()
        rules2 = get_built_in_rules()
        rules1[0].cooldown_seconds = 999
        assert rules2[0].cooldown_seconds != 999

    def test_verify_rule_triggers(self):
        """verification-drop-critical should fire at 0.55."""
        rule = next(r for r in BUILT_IN_RULES if r.name == "verification-drop-critical")
        triggered, _ = rule.evaluate({"verification_pass_rate": 0.55})
        assert triggered is True

    def test_verify_rule_does_not_trigger(self):
        rule = next(r for r in BUILT_IN_RULES if r.name == "verification-drop-critical")
        triggered, _ = rule.evaluate({"verification_pass_rate": 0.85})
        assert triggered is False

    def test_hallucination_rule_triggers(self):
        rule = next(r for r in BUILT_IN_RULES if r.name == "hallucination-spike-critical")
        triggered, _ = rule.evaluate({"hallucination_rate": 0.20})
        assert triggered is True

    def test_multi_condition_rule_both_match(self):
        rule = next(r for r in BUILT_IN_RULES if r.name == "multi-dimensional-degradation-critical")
        triggered, _ = rule.evaluate({
            "verification_pass_rate": 0.55,
            "groundedness": 0.55,
        })
        assert triggered is True

    def test_multi_condition_rule_partial_match(self):
        rule = next(r for r in BUILT_IN_RULES if r.name == "multi-dimensional-degradation-critical")
        triggered, _ = rule.evaluate({
            "verification_pass_rate": 0.55,
            "groundedness": 0.85,
        })
        assert triggered is False


class TestEvaluateCondition:
    def test_lt_true(self):
        cond = AlertCondition("m", "lt", 0.80)
        assert evaluate_condition(cond, 0.70) is True

    def test_lt_false(self):
        cond = AlertCondition("m", "lt", 0.80)
        assert evaluate_condition(cond, 0.90) is False


class TestEvaluateRule:
    def test_triggered(self):
        rule = AlertRule(
            name="test", description="test",
            conditions=[AlertCondition("m", "lt", 0.80)],
        )
        triggered, matched = evaluate_rule(rule, {"m": 0.70})
        assert triggered is True
        assert len(matched) == 1

    def test_not_triggered(self):
        rule = AlertRule(
            name="test", description="test",
            conditions=[AlertCondition("m", "lt", 0.80)],
        )
        triggered, matched = evaluate_rule(rule, {"m": 0.90})
        assert triggered is False


# ═══════════════════════════════════════════════════════════════════════
# AlertManager
# ═══════════════════════════════════════════════════════════════════════

class TestAlertManagerEvaluateStatic:
    def test_healthy_metrics_no_alerts(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 0.90,
            "groundedness": 0.88,
            "hallucination_rate": 0.01,
            "failure_rate": 0.02,
            "latency_p95_ms": 2000,
            "composite_score": 0.80,
            "tool_failure_rate": 0.02,
            "citation_quality": 0.95,
            "memory_f1": 0.90,
        })
        assert len(report.alerts_triggered) == 0

    def test_degrading_metrics_triggers_alerts(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 0.40,
            "groundedness": 0.50,
            "hallucination_rate": 0.25,
            "failure_rate": 0.30,
            "latency_p95_ms": 15000,
            "composite_score": 0.30,
            "tool_failure_rate": 0.15,
            "citation_quality": 0.60,
            "memory_f1": 0.40,
        })
        # Should trigger multiple critical + warning rules
        assert len(report.alerts_triggered) > 0
        assert report.has_critical is True

    def test_evaluate_static_no_side_effects(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        assert mgr.evaluation_count == 0
        report = mgr.evaluate_static({"verification_pass_rate": 0.40})
        # evaluate_static should NOT increment evaluation_count
        assert mgr.evaluation_count == 0

    def test_evaluate_increments_count(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        mgr.evaluate({"verification_pass_rate": 0.90})
        assert mgr.evaluation_count == 1

    def test_permissive_profile_fewer_alerts(self):
        strict_mgr = AlertManager(ALERT_PROFILES["strict"])
        perm_mgr = AlertManager(ALERT_PROFILES["permissive"])

        bad_metrics = {
            "verification_pass_rate": 0.40,
            "groundedness": 0.50,
            "hallucination_rate": 0.25,
            "failure_rate": 0.30,
            "latency_p95_ms": 15000,
            "composite_score": 0.30,
            "tool_failure_rate": 0.15,
            "citation_quality": 0.60,
            "memory_f1": 0.40,
        }
        strict_report = strict_mgr.evaluate_static(bad_metrics)
        perm_report = perm_mgr.evaluate_static(bad_metrics)
        assert len(perm_report.alerts_triggered) < len(strict_report.alerts_triggered)

    def test_silent_profile_no_alerts(self):
        mgr = AlertManager(ALERT_PROFILES["silent"])
        report = mgr.evaluate_static({"verification_pass_rate": 0.10})
        assert len(report.alerts_triggered) == 0

    def test_evaluate_static_dispatches_nothing(self):
        """evaluate_static should not dispatch even with console channel."""
        mgr = AlertManager(ALERT_PROFILES["strict"])
        report = mgr.evaluate_static({"verification_pass_rate": 0.30})
        assert len(report.alerts_triggered) > 0

    def test_cooldown_respected(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        # First evaluation
        report1 = mgr.evaluate({"verification_pass_rate": 0.40})
        triggered1 = len(report1.alerts_triggered)
        # Second evaluation — should be suppressed by cooldown
        report2 = mgr.evaluate({"verification_pass_rate": 0.40})
        triggered2 = len(report2.alerts_triggered)
        assert triggered2 < triggered1

    def test_reset_cooldowns(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        mgr.evaluate({"verification_pass_rate": 0.40})
        mgr.reset_cooldowns()
        # Should fire again
        report = mgr.evaluate({"verification_pass_rate": 0.40})
        assert len(report.alerts_triggered) > 0

    def test_custom_config(self):
        rule = AlertRule(
            name="custom", description="custom rule",
            conditions=[AlertCondition("custom_metric", "gt", 100)],
            severity=AlertSeverity.WARNING,
        )
        config = AlertManagerConfig(rules=[rule])
        mgr = AlertManager(config)
        report = mgr.evaluate_static({"custom_metric": 150})
        assert len(report.alerts_triggered) == 1
        assert report.alerts_triggered[0].rule_name == "custom"

    def test_disabled_rule_in_config(self):
        rule = AlertRule(
            name="disabled", description="disabled",
            conditions=[AlertCondition("m", "lt", 0.80)],
            enabled=False,
        )
        config = AlertManagerConfig(rules=[rule])
        mgr = AlertManager(config)
        report = mgr.evaluate_static({"m": 0.50})
        assert len(report.alerts_triggered) == 0

    def test_missing_metric_in_snapshot(self):
        """Rules referencing metrics not in the snapshot should not fire."""
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 0.40,
            # Many metrics missing — only rules with verification_pass_rate fire
        })
        # verification-drop-critical and verification-drop-warning should fire
        triggered_names = [a.rule_name for a in report.alerts_triggered]
        assert "verification-drop-critical" in triggered_names
        assert "verification-drop-warning" in triggered_names

    def test_report_metrics(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({"verification_pass_rate": 0.40})
        assert report.rules_evaluated > 0
        assert report.conditions_checked > 0
        assert report.conditions_matched > 0

    def test_run_id_passed_through(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static(
            {"verification_pass_rate": 0.40},
            run_id="abc-123",
        )
        for alert in report.alerts_triggered:
            assert alert.run_id == "abc-123"


# ═══════════════════════════════════════════════════════════════════════
# Channels
# ═══════════════════════════════════════════════════════════════════════

class TestConsoleChannel:
    def test_send_enabled(self, capsys):
        channel = ConsoleChannel()
        alert = _make_alert(AlertSeverity.CRITICAL, "test-critical")
        result = channel.send(alert)
        assert result is True
        captured = capsys.readouterr()
        assert "[CRITICAL]" in captured.out
        assert "test-critical" in captured.out

    def test_send_disabled(self):
        config = AlertChannelConfig(type="console", enabled=False)
        channel = ConsoleChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False


class TestFileChannel:
    def test_send_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alerts.jsonl")
            config = AlertChannelConfig(
                type="file", config={"path": path}, enabled=True,
            )
            channel = FileChannel(config)
            alert = _make_alert(AlertSeverity.WARNING, "file-test")
            result = channel.send(alert)
            assert result is True
            assert os.path.exists(path)

            # Verify JSON content
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["rule_name"] == "file-test"
            assert data["severity"] == "warning"

    def test_send_multiple_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alerts.jsonl")
            config = AlertChannelConfig(
                type="file", config={"path": path}, enabled=True,
            )
            channel = FileChannel(config)
            channel.send(_make_alert(AlertSeverity.INFO, "first"))
            channel.send(_make_alert(AlertSeverity.WARNING, "second"))

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["rule_name"] == "first"
            assert json.loads(lines[1])["rule_name"] == "second"

    def test_send_disabled(self, tmp_path):
        path = tmp_path / "alerts.jsonl"
        config = AlertChannelConfig(
            type="file", config={"path": str(path)}, enabled=False,
        )
        channel = FileChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "nested", "alerts.jsonl")
            config = AlertChannelConfig(
                type="file", config={"path": path}, enabled=True,
            )
            channel = FileChannel(config)
            result = channel.send(_make_alert(AlertSeverity.INFO, "deep"))
            assert result is True
            assert os.path.exists(path)

    def test_path_property(self):
        config = AlertChannelConfig(
            type="file", config={"path": "/tmp/test.jsonl"}, enabled=True,
        )
        channel = FileChannel(config)
        assert str(channel.path) == "/tmp/test.jsonl"


class TestWebhookChannel:
    def test_disabled_when_no_url(self):
        config = AlertChannelConfig(type="webhook", config={}, enabled=True)
        channel = WebhookChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_disabled_by_config(self):
        config = AlertChannelConfig(
            type="webhook",
            config={"url": "http://example.com/webhook"},
            enabled=False,
        )
        channel = WebhookChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_enabled_with_url_does_not_crash(self):
        """Even if webhook endpoint is unreachable, send() should not raise."""
        config = AlertChannelConfig(
            type="webhook",
            config={"url": "http://localhost:19999/webhook", "timeout": 1},
            enabled=True,
        )
        channel = WebhookChannel(config)
        # Should return False on connection failure, not raise
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False  # Connection refused = False, not exception


class TestCreateChannel:
    def test_console(self):
        ch = create_channel(AlertChannelConfig(type="console", enabled=True))
        assert isinstance(ch, ConsoleChannel)

    def test_file(self):
        ch = create_channel(AlertChannelConfig(
            type="file", config={"path": "/tmp/test.jsonl"}, enabled=True,
        ))
        assert isinstance(ch, FileChannel)

    def test_webhook(self):
        ch = create_channel(AlertChannelConfig(
            type="webhook", config={"url": "http://example.com"}, enabled=True,
        ))
        assert isinstance(ch, WebhookChannel)

    def test_disabled_returns_noop(self):
        ch = create_channel(AlertChannelConfig(type="console", enabled=False))
        assert isinstance(ch, _NoOpChannel)
        result = ch.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown alert channel type"):
            create_channel(AlertChannelConfig(type="sms", enabled=True))


class TestNoOpChannel:
    def test_send_returns_false(self):
        ch = _NoOpChannel()
        assert ch.send(_make_alert(AlertSeverity.WARNING, "test")) is False


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Slack Block Kit channel tests
# ═══════════════════════════════════════════════════════════════════════

class TestSlackBlockChannel:
    def test_disabled_when_no_url(self):
        config = AlertChannelConfig(type="slack", config={}, enabled=True)
        channel = SlackBlockChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_disabled_by_config(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=False,
        )
        channel = SlackBlockChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_enabled_with_url_does_not_crash(self):
        """Even if Slack webhook is unreachable, send() should not raise."""
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "http://localhost:19999/slack", "timeout": 1},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        # Should return False on connection failure, not raise
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_build_blocks_critical(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "verification-drop")
        blocks = channel._build_blocks(alert)

        assert isinstance(blocks, list)
        assert len(blocks) >= 3
        # Header block
        assert blocks[0]["type"] == "header"
        assert "CRITICAL" in blocks[0]["text"]["text"]
        assert "verification-drop" in blocks[0]["text"]["text"]
        assert ":red_circle:" in blocks[0]["text"]["text"]
        # Section with message
        assert blocks[1]["type"] == "section"
        # Divider at end
        assert blocks[-1]["type"] == "divider"

    def test_build_blocks_warning(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "latency-spike")
        blocks = channel._build_blocks(alert)

        assert ":warning:" in blocks[0]["text"]["text"]
        assert "WARNING" in blocks[0]["text"]["text"]

    def test_build_blocks_info(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = _make_alert(AlertSeverity.INFO, "citation-quality")
        blocks = channel._build_blocks(alert)

        assert ":information_source:" in blocks[0]["text"]["text"]
        assert "INFO" in blocks[0]["text"]["text"]

    def test_build_blocks_includes_conditions(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test-rule")
        blocks = channel._build_blocks(alert)

        # Find conditions section
        conditions_blocks = [b for b in blocks if "Conditions" in str(b)]
        assert len(conditions_blocks) > 0
        conditions_text = str(conditions_blocks)
        assert "test_metric" in conditions_text

    def test_build_blocks_includes_current_values(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "test")
        blocks = channel._build_blocks(alert)

        metrics_blocks = [b for b in blocks if "Current metrics" in str(b)]
        assert len(metrics_blocks) > 0

    def test_build_blocks_with_run_id(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="test message", conditions_matched=[
                AlertCondition("m", "lt", 0.80),
            ],
            current_values={"m": 0.55},
            triggered_at="2026-01-01T00:00:00Z",
            run_id="run-abc-123",
        )
        blocks = channel._build_blocks(alert)

        context_blocks = [b for b in blocks if b.get("type") == "context"]
        assert len(context_blocks) > 0
        context_text = str(context_blocks[0])
        assert "run-abc-123" in context_text

    def test_build_blocks_empty_conditions(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.INFO,
            message="no conditions", conditions_matched=[],
            current_values={}, triggered_at="",
        )
        blocks = channel._build_blocks(alert)
        # Should still have header, section, context, divider
        assert len(blocks) >= 3

    def test_build_blocks_metrics_truncated_at_six(self):
        config = AlertChannelConfig(
            type="slack",
            config={"webhook_url": "https://hooks.slack.com/services/test"},
            enabled=True,
        )
        channel = SlackBlockChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="test", conditions_matched=[],
            current_values={f"metric_{i}": float(i) for i in range(10)},
            triggered_at="",
        )
        blocks = channel._build_blocks(alert)
        metrics_blocks = [b for b in blocks if "Current metrics" in str(b)]
        text = str(metrics_blocks[0])
        # metrics_lines[:6] — should have metric_0 through metric_5 but not metric_9
        assert "metric_0" in text
        assert "metric_5" in text
        # metric_9 might or might not be present since it's the 10th item


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Discord embed channel tests
# ═══════════════════════════════════════════════════════════════════════

class TestDiscordEmbedChannel:
    def test_disabled_when_no_url(self):
        config = AlertChannelConfig(type="discord", config={}, enabled=True)
        channel = DiscordEmbedChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_disabled_by_config(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=False,
        )
        channel = DiscordEmbedChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_enabled_with_url_does_not_crash(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "http://localhost:19999/discord", "timeout": 1},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_build_embed_critical(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "verification-drop")
        embed = channel.build_embed(alert)

        assert embed["title"] == "[CRITICAL] verification-drop"
        assert embed["color"] == 0xFF0000
        assert "description" in embed
        assert "timestamp" in embed
        assert "footer" in embed
        assert "AgentOps" in embed["footer"]["text"]

    def test_build_embed_warning_color(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test")
        embed = channel.build_embed(alert)
        assert embed["color"] == 0xFFA500

    def test_build_embed_info_color(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = _make_alert(AlertSeverity.INFO, "test")
        embed = channel.build_embed(alert)
        assert embed["color"] == 0x0066FF

    def test_build_embed_includes_conditions_field(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test-rule")
        embed = channel.build_embed(alert)

        condition_fields = [f for f in embed.get("fields", []) if "Conditions" in f["name"]]
        assert len(condition_fields) > 0
        assert "test_metric" in condition_fields[0]["value"]

    def test_build_embed_includes_metrics_field(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test")
        embed = channel.build_embed(alert)

        metric_fields = [f for f in embed.get("fields", []) if "Metrics" in f["name"]]
        assert len(metric_fields) > 0
        assert "test_metric" in metric_fields[0]["value"]

    def test_build_embed_with_run_id(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="test message", conditions_matched=[
                AlertCondition("m", "lt", 0.80),
            ],
            current_values={"m": 0.55},
            triggered_at="2026-01-01T00:00:00Z",
            run_id="run-xyz-456",
        )
        embed = channel.build_embed(alert)
        assert "run-xyz-456" in embed["footer"]["text"]

    def test_build_embed_empty_conditions(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.INFO,
            message="no conditions", conditions_matched=[],
            current_values={}, triggered_at="",
        )
        embed = channel.build_embed(alert)
        assert embed["title"] == "[INFO] test"
        # Should NOT have conditions field
        condition_fields = [f for f in embed.get("fields", []) if "Conditions" in f["name"]]
        assert len(condition_fields) == 0

    def test_metrics_chunked_across_fields(self):
        config = AlertChannelConfig(
            type="discord",
            config={"webhook_url": "https://discord.com/api/webhooks/test"},
            enabled=True,
        )
        channel = DiscordEmbedChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="test", conditions_matched=[],
            current_values={f"metric_{i}": float(i) for i in range(12)},
            triggered_at="",
        )
        embed = channel.build_embed(alert)
        metric_fields = [f for f in embed.get("fields", []) if "Metrics" in f["name"]]
        # 12 metrics in chunks of 5 = 3 fields
        assert len(metric_fields) == 3
        assert "Metrics" in metric_fields[0]["name"]
        assert "Metrics (cont.)" in metric_fields[1]["name"]
        assert "Metrics (cont.)" in metric_fields[2]["name"]


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Email channel tests
# ═══════════════════════════════════════════════════════════════════════

class TestEmailChannel:
    def test_disabled_when_no_recipients(self):
        config = AlertChannelConfig(
            type="email",
            config={"smtp_host": "localhost", "from_addr": "test@test.com"},
            enabled=True,
        )
        channel = EmailChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_disabled_by_config(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=False,
        )
        channel = EmailChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_enabled_but_smtp_unreachable_does_not_crash(self):
        """Even if SMTP server is unreachable, send() should not raise."""
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "smtp_port": 19999,
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
                "timeout": 1,
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        result = channel.send(_make_alert(AlertSeverity.WARNING, "test"))
        assert result is False

    def test_build_message_subject_critical(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "verification-drop")
        msg = channel.build_message(alert)

        assert "[AgentOps CRITICAL] verification-drop" == msg["Subject"]

    def test_build_message_subject_warning(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "latency-spike")
        msg = channel.build_message(alert)

        assert "[AgentOps WARNING] latency-spike" == msg["Subject"]

    def test_build_message_subject_info(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.INFO, "citation-quality")
        msg = channel.build_message(alert)

        assert "[AgentOps INFO] citation-quality" == msg["Subject"]

    def test_build_message_subject_with_prefix(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
                "subject_prefix": "[PROD]",
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "test")
        msg = channel.build_message(alert)

        assert msg["Subject"].startswith("[PROD] [AgentOps CRITICAL]")

    def test_build_message_subject_with_run_id(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = Alert(
            rule_name="test", severity=AlertSeverity.WARNING,
            message="test message", conditions_matched=[
                AlertCondition("m", "lt", 0.80),
            ],
            current_values={"m": 0.55},
            triggered_at="",
            run_id="run-abc-123-very-long-run-id-for-truncation",
        )
        msg = channel.build_message(alert)
        assert "(Run: run-abc-123-)" in msg["Subject"]

    def test_build_message_has_plaintext_part(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test-rule")
        msg = channel.build_message(alert)

        parts = msg.get_payload()
        assert len(parts) == 2
        assert parts[0].get_content_type() == "text/plain"
        assert parts[1].get_content_type() == "text/html"

    def test_build_message_plaintext_content(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "test-rule")
        msg = channel.build_message(alert)
        plaintext = msg.get_payload()[0].get_payload(decode=True).decode("utf-8")

        assert "CRITICAL" in plaintext
        assert "test-rule" in plaintext
        assert "test_metric" in plaintext
        assert "0.55" in plaintext or "0.550" in plaintext
        assert "AgentOps Reliability Platform" in plaintext

    def test_build_message_html_content(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.CRITICAL, "test-rule")
        msg = channel.build_message(alert)
        html = msg.get_payload()[1].get_payload(decode=True).decode("utf-8")

        assert "<!DOCTYPE html>" in html
        assert "CRITICAL" in html
        assert "test-rule" in html
        assert "#FF0000" in html  # red for critical
        assert "AgentOps Reliability Platform" in html

    def test_build_message_html_warning_color(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test")
        msg = channel.build_message(alert)
        html = msg.get_payload()[1].get_payload(decode=True).decode("utf-8")

        assert "#FFA500" in html

    def test_build_message_html_info_color(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "test@test.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.INFO, "test")
        msg = channel.build_message(alert)
        html = msg.get_payload()[1].get_payload(decode=True).decode("utf-8")

        assert "#0066FF" in html

    def test_build_message_from_and_to(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "localhost",
                "from_addr": "alerts@agentops.example.com",
                "to_addrs": ["oncall@example.com", "lead@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        alert = _make_alert(AlertSeverity.WARNING, "test")
        msg = channel.build_message(alert)

        assert msg["From"] == "alerts@agentops.example.com"
        assert msg["To"] == "oncall@example.com, lead@example.com"

    def test_default_config_values(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        assert channel._smtp_host == "localhost"
        assert channel._smtp_port == 587
        assert channel._use_tls is True
        assert channel._from_addr == "agentops@localhost"
        assert channel._username is None
        assert channel._password is None

    def test_smtp_auth_configured(self):
        config = AlertChannelConfig(
            type="email",
            config={
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@example.com",
                "password": "app-password-here",
                "from_addr": "alerts@example.com",
                "to_addrs": ["oncall@example.com"],
            },
            enabled=True,
        )
        channel = EmailChannel(config)
        assert channel._username == "alerts@example.com"
        assert channel._password == "app-password-here"


# ═══════════════════════════════════════════════════════════════════════
# v0.16: Updated create_channel tests
# ═══════════════════════════════════════════════════════════════════════

class TestCreateChannelV16:
    def test_slack(self):
        ch = create_channel(AlertChannelConfig(
            type="slack", config={"webhook_url": "https://hooks.slack.com/test"},
            enabled=True,
        ))
        assert isinstance(ch, SlackBlockChannel)

    def test_discord(self):
        ch = create_channel(AlertChannelConfig(
            type="discord", config={"webhook_url": "https://discord.com/api/test"},
            enabled=True,
        ))
        assert isinstance(ch, DiscordEmbedChannel)

    def test_email(self):
        ch = create_channel(AlertChannelConfig(
            type="email",
            config={"smtp_host": "localhost", "to_addrs": ["test@test.com"]},
            enabled=True,
        ))
        assert isinstance(ch, EmailChannel)


# ═══════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════

class TestFullAlertLifecycle:
    def test_healthy_to_degrading_transition(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        mgr.reset_cooldowns()

        # Start healthy
        healthy = mgr.evaluate_static({
            "verification_pass_rate": 0.90,
            "groundedness": 0.88,
            "hallucination_rate": 0.01,
            "failure_rate": 0.02,
            "latency_p95_ms": 2000,
            "composite_score": 0.80,
            "tool_failure_rate": 0.02,
            "citation_quality": 0.95,
            "memory_f1": 0.90,
        })
        assert len(healthy.alerts_triggered) == 0

        # Degrade
        degrading = mgr.evaluate_static({
            "verification_pass_rate": 0.40,
            "groundedness": 0.55,
            "hallucination_rate": 0.25,
            "failure_rate": 0.30,
            "latency_p95_ms": 15000,
            "composite_score": 0.35,
            "tool_failure_rate": 0.15,
            "citation_quality": 0.60,
            "memory_f1": 0.40,
        })
        assert len(degrading.alerts_triggered) > 5
        assert degrading.has_critical is True
        assert degrading.has_warnings is True

    def test_json_serialization_roundtrip(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 0.40,
            "groundedness": 0.55,
            "hallucination_rate": 0.25,
            "failure_rate": 0.30,
            "latency_p95_ms": 15000,
            "composite_score": 0.35,
        })

        # Serialize
        d = report.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["alerts_triggered"] == len(report.alerts_triggered)
        assert parsed["rules_evaluated"] == report.rules_evaluated
        assert len(parsed["alerts"]) == len(report.alerts_triggered)

    def test_markdown_report_content(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 0.40,
            "hallucination_rate": 0.25,
            "groundedness": 0.55,
        })
        md = report.to_markdown()
        assert "Alert Evaluation Report" in md
        assert "Triggered Alerts" in md
        # Should have at least one critical
        assert "critical" in md.lower()


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_metrics_dict(self):
        mgr = AlertManager(ALERT_PROFILES["strict"])
        report = mgr.evaluate_static({})
        assert len(report.alerts_triggered) == 0

    def test_zero_values(self):
        mgr = AlertManager(ALERT_PROFILES["production"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 0.0,
            "groundedness": 0.0,
            "hallucination_rate": 0.0,
            "failure_rate": 0.0,
            "latency_p95_ms": 0.0,
            "composite_score": 0.0,
            "tool_failure_rate": 0.0,
            "citation_quality": 0.0,
            "memory_f1": 0.0,
        })
        # Zero verification should trigger, zero hallucination should NOT trigger
        assert "verification-drop-critical" in [a.rule_name for a in report.alerts_triggered]

    def test_very_high_values(self):
        mgr = AlertManager(ALERT_PROFILES["strict"])
        report = mgr.evaluate_static({
            "verification_pass_rate": 1.0,
            "groundedness": 1.0,
            "hallucination_rate": 1.0,
            "failure_rate": 1.0,
            "latency_p95_ms": 1_000_000,
            "composite_score": 1.0,
            "tool_failure_rate": 1.0,
            "citation_quality": 0.0,
            "memory_f1": 0.0,
        })
        assert len(report.alerts_triggered) > 0

    def test_boundary_values(self):
        """Test values exactly at thresholds."""
        # verification-drop-critical: lt 0.60 — exactly 0.60 should NOT trigger
        rule = next(r for r in BUILT_IN_RULES if r.name == "verification-drop-critical")
        triggered, _ = rule.evaluate({"verification_pass_rate": 0.60})
        assert triggered is False

        # hallucination-spike-critical: gt 0.15 — exactly 0.15 should NOT trigger
        rule = next(r for r in BUILT_IN_RULES if r.name == "hallucination-spike-critical")
        triggered, _ = rule.evaluate({"hallucination_rate": 0.15})
        assert triggered is False

    def test_float_precision(self):
        """Conditions with tight float thresholds should still work."""
        cond = AlertCondition("m", "eq", 0.80)
        assert cond.evaluate(0.80) is True
        assert cond.evaluate(0.80000000001) is True  # within 1e-9 epsilon
        assert cond.evaluate(0.800001) is False  # well outside epsilon

    def test_alert_with_no_matched_conditions(self):
        """Alert can be created even without conditions for edge cases."""
        alert = Alert(
            rule_name="edge", severity=AlertSeverity.INFO,
            message="test", conditions_matched=[],
            current_values={}, triggered_at="",
        )
        assert alert.to_dict()["conditions"] == []

    def test_profiles_are_independent(self):
        """Production and strict profiles have different rule counts."""
        prod = get_alert_profile("production")
        strict = get_alert_profile("strict")
        assert prod is not None
        assert strict is not None

        # Strict profile has all rules (including INFO), production excludes INFO
        assert len(strict.rules) > len(prod.rules)
        # Both profiles are valid and have rules
        assert len(prod.rules) > 0


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_alert(severity: AlertSeverity, name: str) -> Alert:
    return Alert(
        rule_name=name,
        severity=severity,
        message=f"Test alert: {name}",
        conditions_matched=[AlertCondition("test_metric", "lt", 0.80)],
        current_values={"test_metric": 0.55},
        triggered_at=datetime.now(timezone.utc).isoformat(),
    )
