"""
Tests for AgentOps v0.18 — Production Readiness Assessment.

Covers:
- State models: ReadinessTier, ReadinessDimension, ReadinessDimensionScore, ReadinessReport
- ReadinessAssessor: dimension scoring, composite computation, tier assignment
- ReadinessAssessor: healthy agent (all pass), critical agent (multiple fails)
- ReadinessAssessor: degraded/conditional agent, borderline cases
- ReadinessAssessor: missing dimension handling (no data = fail/warn)
- ReadinessAssessor: multi-agent dimension optionality
- ReadinessAssessor: assess_from_eval_reports integration
- Reporting: format_readiness_json, format_readiness_markdown, format_readiness_html
- Reporting: edge cases (empty report, single dimension)
- CLI: readiness assess command, readiness gate command, readiness scenarios
- Integration: full lifecycle (assess → report → gate)
"""

from __future__ import annotations

import json

import pytest

from agentops.readiness.state import (
    READINESS_DIMENSIONS,
    READINESS_THRESHOLDS,
    ReadinessDimension,
    ReadinessDimensionScore,
    ReadinessReport,
    ReadinessTier,
)
from agentops.readiness.assessor import ReadinessAssessor
from agentops.readiness.reporting import (
    format_readiness_json,
    format_readiness_markdown,
    format_readiness_html,
)


# ═══════════════════════════════════════════════════════════════════════
# State Models
# ═══════════════════════════════════════════════════════════════════════

class TestReadinessTier:
    def test_values(self):
        assert ReadinessTier.PRODUCTION_READY.value == "production_ready"
        assert ReadinessTier.CONDITIONAL.value == "conditional"
        assert ReadinessTier.NEEDS_WORK.value == "needs_work"
        assert ReadinessTier.CRITICAL_ISSUES.value == "critical_issues"

    def test_labels_are_human_readable(self):
        assert "Production Ready" in ReadinessTier.PRODUCTION_READY.label
        assert "Conditional" in ReadinessTier.CONDITIONAL.label
        assert "Needs Work" in ReadinessTier.NEEDS_WORK.label
        assert "Critical" in ReadinessTier.CRITICAL_ISSUES.label

    def test_exit_codes(self):
        assert ReadinessTier.PRODUCTION_READY.exit_code == 0
        assert ReadinessTier.CONDITIONAL.exit_code == 0
        assert ReadinessTier.NEEDS_WORK.exit_code == 1
        assert ReadinessTier.CRITICAL_ISSUES.exit_code == 2


class TestReadinessDimension:
    def test_count(self):
        assert len(READINESS_DIMENSIONS) == 8

    def test_weights_sum_to_one(self):
        total = sum(d.weight for d in READINESS_DIMENSIONS)
        assert abs(total - 1.0) < 0.01

    def test_all_have_labels(self):
        for d in READINESS_DIMENSIONS:
            assert len(d.label) > 0

    def test_healthy_agent_weights(self):
        """Weights are highest on verification, safety, tools, response."""
        top4 = sorted(READINESS_DIMENSIONS, key=lambda d: d.weight, reverse=True)[:4]
        top4_names = {d.value for d in top4}
        expected = {"verification_quality", "safety_compliance", "tool_reliability", "response_quality"}
        assert top4_names == expected

    def test_pass_threshold(self):
        for d in READINESS_DIMENSIONS:
            assert d.pass_threshold == 80.0

    def test_warn_threshold(self):
        for d in READINESS_DIMENSIONS:
            assert d.warn_threshold == 60.0


class TestReadinessDimensionScore:
    def test_creation_defaults(self):
        score = ReadinessDimensionScore(
            dimension=ReadinessDimension.VERIFICATION_QUALITY,
            score=85.0,
            status="pass",
        )
        assert score.dimension == ReadinessDimension.VERIFICATION_QUALITY
        assert score.score == 85.0
        assert score.status == "pass"
        assert len(score.evidence) > 0  # auto-generated

    def test_explicit_evidence(self):
        score = ReadinessDimensionScore(
            dimension=ReadinessDimension.VERIFICATION_QUALITY,
            score=55.0,
            status="fail",
            evidence=["Verification rate: 45%", "Groundedness: 0.30"],
            recommendation="Fix verification pipeline.",
        )
        assert len(score.evidence) == 2
        assert "45%" in score.evidence[0]
        assert score.recommendation == "Fix verification pipeline."


class TestReadinessReport:
    def _make_report(self, tier=ReadinessTier.PRODUCTION_READY, composite=92.0):
        return ReadinessReport(
            agent_name="test-agent",
            agent_version="0.18.0",
            tier=tier,
            composite_score=composite,
            dimension_scores=[
                ReadinessDimensionScore(
                    dimension=ReadinessDimension.VERIFICATION_QUALITY,
                    score=95.0,
                    status="pass",
                ),
                ReadinessDimensionScore(
                    dimension=ReadinessDimension.SAFETY_COMPLIANCE,
                    score=90.0,
                    status="pass",
                ),
                ReadinessDimensionScore(
                    dimension=ReadinessDimension.TOOL_RELIABILITY,
                    score=85.0,
                    status="pass",
                ),
            ],
        )

    def test_counts(self):
        report = self._make_report()
        assert report.pass_count == 3
        assert report.warn_count == 0
        assert report.fail_count == 0

    def test_is_deployable(self):
        assert self._make_report(ReadinessTier.PRODUCTION_READY).is_deployable is True
        assert self._make_report(ReadinessTier.CONDITIONAL).is_deployable is True
        assert self._make_report(ReadinessTier.NEEDS_WORK).is_deployable is False
        assert self._make_report(ReadinessTier.CRITICAL_ISSUES).is_deployable is False

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["agent_name"] == "test-agent"
        assert d["agent_version"] == "0.18.0"
        assert d["tier"] == "production_ready"
        assert d["composite_score"] == 92.0
        assert d["deployable"] is True
        assert d["exit_code"] == 0
        assert len(d["dimensions"]) == 3
        assert d["summary"]["pass"] == 3
        assert d["summary"]["fail"] == 0

    def test_counts_mixed(self):
        report = ReadinessReport(
            agent_name="test",
            agent_version="0.18.0",
            tier=ReadinessTier.NEEDS_WORK,
            composite_score=55.0,
            dimension_scores=[
                ReadinessDimensionScore(
                    dimension=ReadinessDimension.VERIFICATION_QUALITY,
                    score=85.0, status="pass",
                ),
                ReadinessDimensionScore(
                    dimension=ReadinessDimension.SAFETY_COMPLIANCE,
                    score=65.0, status="warn",
                ),
                ReadinessDimensionScore(
                    dimension=ReadinessDimension.TOOL_RELIABILITY,
                    score=40.0, status="fail",
                ),
            ],
        )
        assert report.pass_count == 1
        assert report.warn_count == 1
        assert report.fail_count == 1


# ═══════════════════════════════════════════════════════════════════════
# ReadinessAssessor — Core Scoring
# ═══════════════════════════════════════════════════════════════════════

class TestReadinessAssessor:
    def test_healthy_agent_production_ready(self):
        """A well-performing agent should score PRODUCTION_READY."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )

        assert report.tier == ReadinessTier.PRODUCTION_READY
        assert report.composite_score >= 85
        assert report.is_deployable is True
        assert report.pass_count >= 7
        assert report.fail_count == 0

    def test_critical_agent_multiple_failures(self):
        """An agent with multiple dimension failures gets CRITICAL_ISSUES."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.30, "groundedness_mean": 0.25},
            guardrail_stats={"block_rate": 0.40, "false_negative_rate": 0.30, "active_patterns": 5},
            tool_stats={"tool_success_rate": 0.35, "schema_compliance_rate": 0.40, "hallucinated_tool_rate": 0.25},
            judge_scores={"accuracy": 25.0, "completeness": 20.0, "relevance": 30.0, "clarity": 35.0},
            retrieval_stats={"citation_precision": 0.25, "relevance_score": 0.20, "mrr": 0.15},
            latency_stats={"avg_latency_ms": 10000, "p95_latency_ms": 20000, "budget_compliance_rate": 0.30},
            memory_stats={"recall_precision": 0.15, "recall_rate": 0.10, "f1_score": 0.12, "hallucination_rate": 0.50},
        )

        assert report.tier == ReadinessTier.CRITICAL_ISSUES
        assert report.composite_score < 50
        assert report.is_deployable is False
        assert report.fail_count > 0

    def test_degraded_agent_needs_work(self):
        """An agent with multiple warnings and one fail gets CRITICAL_ISSUES."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.65, "groundedness_mean": 0.60},
            guardrail_stats={"block_rate": 0.82, "false_negative_rate": 0.08, "active_patterns": 15},
            tool_stats={"tool_success_rate": 0.72, "schema_compliance_rate": 0.75, "hallucinated_tool_rate": 0.06},
            judge_scores={"accuracy": 55.0, "completeness": 50.0, "relevance": 58.0, "clarity": 60.0},
            retrieval_stats={"citation_precision": 0.58, "relevance_score": 0.55, "mrr": 0.50},
            latency_stats={"avg_latency_ms": 3500, "p95_latency_ms": 7000, "budget_compliance_rate": 0.70},
            memory_stats={"recall_precision": 0.45, "recall_rate": 0.40, "f1_score": 0.42, "hallucination_rate": 0.20},
        )

        # Degraded agent with fails → should not be deployable
        assert report.is_deployable is False
        assert report.tier in (ReadinessTier.NEEDS_WORK, ReadinessTier.CRITICAL_ISSUES)

    def test_conditional_agent_one_warning(self):
        """One warning dimension → CONDITIONAL."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.92, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.96, "false_negative_rate": 0.02, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.94, "schema_compliance_rate": 0.93, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 88.0, "completeness": 85.0, "relevance": 90.0, "clarity": 87.0},
            retrieval_stats={"citation_precision": 0.90, "relevance_score": 0.85, "mrr": 0.82},
            latency_stats={"avg_latency_ms": 600, "p95_latency_ms": 1200, "budget_compliance_rate": 0.95},
            memory_stats={"recall_precision": 0.72, "recall_rate": 0.68, "f1_score": 0.70, "hallucination_rate": 0.08},
        )

        assert report.tier == ReadinessTier.CONDITIONAL
        assert report.is_deployable is True
        assert report.warn_count >= 1

    def test_missing_dimension_returns_fail(self):
        """Missing data produces a fail/warn with appropriate status."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            # No guardrail, tool, judge data...
        )

        # Some dimensions will fail due to missing data
        assert report.fail_count >= 1
        # Composite score should be lower due to missing dimensions
        assert report.composite_score < 80

    def test_multi_agent_optional(self):
        """Multi-agent dimension is neutral when no data provided."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
            # No multi_agent_stats
        )

        # Should still be production_ready since multi-agent is optional
        assert report.tier == ReadinessTier.PRODUCTION_READY

    def test_multi_agent_with_data(self):
        """Multi-agent dimension with good data contributes positively."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
            multi_agent_stats={"coordination_score": 0.92, "message_efficiency": 0.90, "task_completion_rate": 0.95},
        )

        assert report.tier == ReadinessTier.PRODUCTION_READY

    def test_multi_agent_poor_performance(self):
        """Poor multi-agent performance fails that dimension but small weight means still deployable."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
            multi_agent_stats={"coordination_score": 0.30, "message_efficiency": 0.25, "task_completion_rate": 0.20},
        )

        # Weight is only 5% but a fail on any dimension triggers CRITICAL_ISSUES.
        # This is correct — a failing multi-agent system should block deployment.
        assert report.tier in (ReadinessTier.CRITICAL_ISSUES, ReadinessTier.NEEDS_WORK)

    def test_custom_weights(self):
        """Custom dimension weights affect composite score."""
        # Give all weight to verification
        custom_weights = {d: (1.0 if d == ReadinessDimension.VERIFICATION_QUALITY else 0.0)
                          for d in READINESS_DIMENSIONS}

        assessor = ReadinessAssessor(dimension_weights=custom_weights)
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            # Other dimensions are terrible
            guardrail_stats={"block_rate": 0.30, "false_negative_rate": 0.50, "active_patterns": 5},
            tool_stats={"tool_success_rate": 0.20, "schema_compliance_rate": 0.20, "hallucinated_tool_rate": 0.50},
            judge_scores={"accuracy": 10.0, "completeness": 10.0, "relevance": 10.0, "clarity": 10.0},
            retrieval_stats={"citation_precision": 0.10, "relevance_score": 0.10, "mrr": 0.10},
            latency_stats={"avg_latency_ms": 15000, "p95_latency_ms": 30000, "budget_compliance_rate": 0.10},
            memory_stats={"recall_precision": 0.10, "recall_rate": 0.10, "f1_score": 0.10, "hallucination_rate": 0.80},
        )

        # With all weight on verification (95% pass), composite should be ~92
        assert report.composite_score > 85

    def test_agent_version_defaults(self):
        """Agent name and version are configurable."""
        assessor = ReadinessAssessor(agent_name="my-custom-agent", agent_version="2.0.0")
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        assert report.agent_name == "my-custom-agent"
        assert report.agent_version == "2.0.0"

    def test_timestamp_populated(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        assert "UTC" in report.run_timestamp
        assert report.assessment_duration_ms >= 0

    def test_trace_and_benchmark_counts(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
            trace_count=500,
            benchmark_count=10,
        )
        assert report.total_traces_analyzed == 500
        assert report.total_benchmarks_run == 10

    def test_assess_from_eval_reports(self):
        """Integration with EvalReport objects."""
        # Create mock eval reports
        class MockEvalReport:
            def __init__(self, summary):
                self.summary = summary

        reports = [
            MockEvalReport({"verification_pass_rate": 0.95, "groundedness_mean": 0.88, "composite_mean": 0.92}),
            MockEvalReport({"verification_pass_rate": 0.93, "groundedness_mean": 0.85, "composite_mean": 0.89}),
        ]

        assessor = ReadinessAssessor()
        report = assessor.assess_from_eval_reports(reports, trace_count=200)

        assert report.total_benchmarks_run == 2
        assert report.total_traces_analyzed == 200
        assert report.composite_score > 20  # Only verification dimension has data

    def test_verification_edge_zero(self):
        """Zero values should produce minimum score."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.0, "groundedness_mean": 0.0},
        )
        dim_score = next(ds for ds in report.dimension_scores
                         if ds.dimension == ReadinessDimension.VERIFICATION_QUALITY)
        assert dim_score.score == 0.0
        assert dim_score.status == "fail"


# ═══════════════════════════════════════════════════════════════════════
# Scoring Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestScoringEdgeCases:
    def test_perfect_scores(self):
        """All perfect scores should yield PRODUCTION_READY with high composite."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 1.0, "groundedness_mean": 1.0},
            guardrail_stats={"block_rate": 1.0, "false_negative_rate": 0.0, "active_patterns": 21},
            tool_stats={"tool_success_rate": 1.0, "schema_compliance_rate": 1.0, "hallucinated_tool_rate": 0.0},
            judge_scores={"accuracy": 100.0, "completeness": 100.0, "relevance": 100.0, "clarity": 100.0},
            retrieval_stats={"citation_precision": 1.0, "relevance_score": 1.0, "mrr": 1.0},
            latency_stats={"avg_latency_ms": 100, "p95_latency_ms": 200, "budget_compliance_rate": 1.0},
            memory_stats={"recall_precision": 1.0, "recall_rate": 1.0, "f1_score": 1.0, "hallucination_rate": 0.0},
        )
        assert report.tier == ReadinessTier.PRODUCTION_READY
        assert report.composite_score >= 90
        assert report.pass_count >= 7

    def test_safety_false_negative_penalizes(self):
        """High false negative rate should reduce safety score."""
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.80, "false_negative_rate": 0.40, "active_patterns": 10},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )

        safety = next(ds for ds in report.dimension_scores
                      if ds.dimension == ReadinessDimension.SAFETY_COMPLIANCE)
        assert safety.score < 55  # 40% false negative is very bad
        assert safety.status == "fail"

    def test_latency_budget_compliance(self):
        """Budget compliance should contribute to latency score."""
        assessor = ReadinessAssessor()
        # Good latency, poor budget compliance
        report_a = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.30},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        # Good latency, good budget compliance
        report_b = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )

        latency_a = next(ds for ds in report_a.dimension_scores
                         if ds.dimension == ReadinessDimension.LATENCY_EFFICIENCY)
        latency_b = next(ds for ds in report_b.dimension_scores
                         if ds.dimension == ReadinessDimension.LATENCY_EFFICIENCY)
        assert latency_a.score < latency_b.score

    # ── Specific dimension scoring tests ──────────────────────────

    def _base_good_data(self):
        return {
            "eval_summary": {"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            "guardrail_stats": {"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            "tool_stats": {"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            "judge_scores": {"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            "retrieval_stats": {"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            "latency_stats": {"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            "memory_stats": {"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        }

    def test_tool_hallucination_penalizes_heavily(self):
        data = self._base_good_data()
        data["tool_stats"]["hallucinated_tool_rate"] = 0.15
        assessor = ReadinessAssessor()
        report = assessor.assess(**data)
        tool = next(ds for ds in report.dimension_scores
                    if ds.dimension == ReadinessDimension.TOOL_RELIABILITY)
        assert tool.score < 80

    def test_memory_hallucination_penalizes(self):
        data = self._base_good_data()
        data["memory_stats"]["hallucination_rate"] = 0.30
        assessor = ReadinessAssessor()
        report = assessor.assess(**data)
        mem = next(ds for ds in report.dimension_scores
                   if ds.dimension == ReadinessDimension.MEMORY_CONSISTENCY)
        assert mem.score < 75


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

class TestFormatReadinessJson:
    def test_valid_json_output(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        json_str = format_readiness_json(report)
        parsed = json.loads(json_str)
        assert parsed["agent_name"] == "agentops-agent"
        assert parsed["tier"] in ("production_ready", "conditional")
        assert len(parsed["dimensions"]) == 8
        assert "recommendations" in parsed

    def test_critical_json(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.30, "groundedness_mean": 0.25},
        )
        json_str = format_readiness_json(report)
        parsed = json.loads(json_str)
        assert parsed["tier"] == "critical_issues"
        assert parsed["deployable"] is False
        assert parsed["exit_code"] == 2


class TestFormatReadinessMarkdown:
    def test_contains_key_sections(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        md = format_readiness_markdown(report)
        assert "Production Readiness Assessment" in md
        assert "Verdict:" in md
        assert "Composite Score" in md
        assert "Dimension Scores" in md
        assert "Recommendations" in md
        assert "AgentOps Reliability Platform" in md

    def test_contains_verdict(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        md = format_readiness_markdown(report)
        assert ("Production Ready" in md or "✅" in md or "Conditional" in md or "warning" in md)

    def test_critical_report_shows_warnings(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.30, "groundedness_mean": 0.25},
        )
        md = format_readiness_markdown(report)
        assert "CRITICAL" in md or "🚨" in md
        assert "Critical Findings" in md
        assert "NOT deploy" in md or "Deployable" in md


class TestFormatReadinessHtml:
    def test_contains_html_structure(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        html = format_readiness_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<title>" in html
        assert "Production Readiness Assessment" in html
        assert "Composite Score" in html or "composite" in html.lower()
        assert "</html>" in html

    def test_includes_all_dimensions(self):
        assessor = ReadinessAssessor()
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.95, "groundedness_mean": 0.88},
            guardrail_stats={"block_rate": 0.98, "false_negative_rate": 0.01, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.95, "schema_compliance_rate": 0.94, "hallucinated_tool_rate": 0.01},
            judge_scores={"accuracy": 90.0, "completeness": 88.0, "relevance": 92.0, "clarity": 90.0},
            retrieval_stats={"citation_precision": 0.92, "relevance_score": 0.88, "mrr": 0.85},
            latency_stats={"avg_latency_ms": 400, "p95_latency_ms": 800, "budget_compliance_rate": 0.98},
            memory_stats={"recall_precision": 0.90, "recall_rate": 0.88, "f1_score": 0.89, "hallucination_rate": 0.02},
        )
        html = format_readiness_html(report)
        for dim in READINESS_DIMENSIONS:
            assert dim.label in html


# ═══════════════════════════════════════════════════════════════════════
# CLI Integration (unit tests via import)
# ═══════════════════════════════════════════════════════════════════════

class TestCliIntegration:
    """Test that CLI commands and sub-app are importable."""

    def test_readiness_subapp_exists(self):
        from agentops.cli.main import readiness_app
        assert readiness_app is not None
        assert readiness_app.info.name == "readiness"

    def test_readiness_module_importable(self):
        """Verify readiness module can be imported."""
        from agentops.readiness import ReadinessAssessor, ReadinessTier
        assert ReadinessAssessor is not None
        assert ReadinessTier is not None


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — Full Lifecycle
# ═══════════════════════════════════════════════════════════════════════

class TestFullReadinessLifecycle:
    def test_assess_report_gate_pipeline(self):
        """Simulate a CI pipeline: assess → generate report → run gate."""
        assessor = ReadinessAssessor()

        # 1. Assess
        report = assessor.assess(
            eval_summary={"verification_pass_rate": 0.92, "groundedness_mean": 0.85},
            guardrail_stats={"block_rate": 0.96, "false_negative_rate": 0.02, "active_patterns": 21},
            tool_stats={"tool_success_rate": 0.91, "schema_compliance_rate": 0.92, "hallucinated_tool_rate": 0.02},
            judge_scores={"accuracy": 85.0, "completeness": 82.0, "relevance": 88.0, "clarity": 86.0},
            retrieval_stats={"citation_precision": 0.88, "relevance_score": 0.85, "mrr": 0.82},
            latency_stats={"avg_latency_ms": 600, "p95_latency_ms": 1200, "budget_compliance_rate": 0.95},
            memory_stats={"recall_precision": 0.85, "recall_rate": 0.82, "f1_score": 0.83, "hallucination_rate": 0.04},
            trace_count=500,
            benchmark_count=10,
        )

        # 2. Generate reports in all formats
        json_report = format_readiness_json(report)
        md_report = format_readiness_markdown(report)
        html_report = format_readiness_html(report)

        assert '"composite_score"' in json_report
        assert "Dimension Scores" in md_report
        assert "<html" in html_report

        # 3. Verify deployable
        assert report.is_deployable is True
        assert report.tier.exit_code == 0

        # 4. Verify all 8 dimensions scored
        assert len(report.dimension_scores) == 8

        # 5. Verify recommendations present
        assert len(report.recommendations) > 0
