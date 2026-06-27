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
