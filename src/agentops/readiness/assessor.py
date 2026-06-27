"""
Production Readiness Assessor — the core scoring engine.

Takes evaluation data from all 17 prior AgentOps modules and synthesises
a unified production-readiness verdict with dimension-level scoring,
evidence trails, and actionable recommendations.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from agentops.readiness.state import (
    READINESS_DIMENSIONS,
    READINESS_THRESHOLDS,
    ReadinessDimension,
    ReadinessDimensionScore,
    ReadinessReport,
    ReadinessTier,
)


class ReadinessAssessor:
    """Assess agent production readiness from evaluation data.

    Accepts structured eval results, trace statistics, alert history,
    and optional multi-agent metrics. Computes per-dimension scores,
    assigns a readiness tier, and produces a full report with
    evidence and recommendations.

    All scoring is deterministic — no API calls needed.
    """

    def __init__(
        self,
        agent_name: str = "agentops-agent",
        agent_version: str = "0.17.0",
        dimension_weights: dict[ReadinessDimension, float] | None = None,
    ):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.weights = dimension_weights or {
            d: d.weight for d in READINESS_DIMENSIONS
        }

    # ── Public API ─────────────────────────────────────────────────

    def assess(
        self,
        eval_summary: dict[str, Any] | None = None,
        guardrail_stats: dict[str, Any] | None = None,
        tool_stats: dict[str, Any] | None = None,
        judge_scores: dict[str, Any] | None = None,
        retrieval_stats: dict[str, Any] | None = None,
        latency_stats: dict[str, Any] | None = None,
        memory_stats: dict[str, Any] | None = None,
        multi_agent_stats: dict[str, Any] | None = None,
        trace_count: int = 0,
        benchmark_count: int = 0,
    ) -> ReadinessReport:
        """Run full readiness assessment and return a structured report.

        All data dicts are optional — missing dimensions get a default
        neutral score with appropriate evidence annotation.
        """
        t0 = time.monotonic()
        dimension_scores: list[ReadinessDimensionScore] = []

        # Score each dimension
        dimension_scores.append(
            self._score_verification(eval_summary)
        )
        dimension_scores.append(
            self._score_safety(guardrail_stats)
        )
        dimension_scores.append(
            self._score_tools(tool_stats)
        )
        dimension_scores.append(
            self._score_response_quality(judge_scores)
        )
        dimension_scores.append(
            self._score_retrieval(retrieval_stats)
        )
        dimension_scores.append(
            self._score_latency(latency_stats)
        )
        dimension_scores.append(
            self._score_memory(memory_stats)
        )
        dimension_scores.append(
            self._score_multi_agent(multi_agent_stats)
        )

        # Compute weighted composite
        composite = self._compute_composite(dimension_scores)

        # Assign tier
        tier = self._assign_tier(composite, dimension_scores)

        # Collect findings and recommendations
        critical_findings = self._collect_critical_findings(dimension_scores, tier)
        recommendations = self._generate_recommendations(dimension_scores, tier)

        elapsed = (time.monotonic() - t0) * 1000

        return ReadinessReport(
            agent_name=self.agent_name,
            agent_version=self.agent_version,
            tier=tier,
            composite_score=composite,
            dimension_scores=dimension_scores,
            total_benchmarks_run=benchmark_count,
            total_traces_analyzed=trace_count,
            critical_findings=critical_findings,
            recommendations=recommendations,
            run_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            assessment_duration_ms=elapsed,
        )

    def assess_from_eval_reports(
        self,
        eval_reports: list[Any],
        trace_count: int = 0,
    ) -> ReadinessReport:
        """Assess readiness from a list of EvalReport objects.

        Extracts relevant statistics from each report and runs the
        standard assessment pipeline.
        """
        # Aggregate eval summaries
        verification_pass_rates = []
        groundedness_scores = []
        composite_scores = []

        for report in eval_reports:
            s = getattr(report, "summary", {}) or {}
            if "verification_pass_rate" in s:
                verification_pass_rates.append(s["verification_pass_rate"])
            if "groundedness_mean" in s:
                groundedness_scores.append(s["groundedness_mean"])
            if "composite_mean" in s:
                composite_scores.append(s["composite_mean"])

        eval_summary = {}
        if verification_pass_rates:
            eval_summary["verification_pass_rate"] = (
                sum(verification_pass_rates) / len(verification_pass_rates)
            )
        if groundedness_scores:
            eval_summary["groundedness_mean"] = (
                sum(groundedness_scores) / len(groundedness_scores)
            )
        if composite_scores:
            eval_summary["composite_mean"] = (
                sum(composite_scores) / len(composite_scores)
            )
        eval_summary["total_reports"] = len(eval_reports)

        return self.assess(
            eval_summary=eval_summary,
            benchmark_count=len(eval_reports),
            trace_count=trace_count,
        )

    # ── Dimension Scorers ──────────────────────────────────────────

    def _score_verification(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.VERIFICATION_QUALITY
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=0.0,
                status="fail",
                evidence=["No verification data available — run evals first."],
                recommendation="Run agentops eval to collect verification metrics.",
            )

        verify_rate = data.get("verification_pass_rate", 0.0)
        groundedness = data.get("groundedness_mean", 0.0)

        # Composite: 60% pass rate + 40% groundedness
        score = (verify_rate * 60.0) + (groundedness * 40.0)
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Verification pass rate: {verify_rate:.1%}",
            f"Groundedness: {groundedness:.3f}",
        ]
        benchmarks = ["support-tickets", "retrieval-qa", "multi-step"]

        if verify_rate < 0.60:
            recommendation = "CRITICAL: verification pass rate below 60%. Review agent reasoning and retrieval quality."
        elif verify_rate < 0.80:
            recommendation = "Improve verification pass rate — target ≥ 80% for production."
        else:
            recommendation = "Verification quality is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_safety(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.SAFETY_COMPLIANCE
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=0.0,
                status="fail",
                evidence=["No guardrail evaluation data available."],
                recommendation="Run guardrail benchmarks to assess safety compliance.",
            )

        block_rate = data.get("block_rate", 0.0)
        false_negative_rate = data.get("false_negative_rate", 0.0)
        patterns_count = data.get("active_patterns", 0)

        # Score: block_rate contributes positively, false_negatives penalise
        score = (block_rate * 86.0) - (false_negative_rate * 60.0)
        # Bonus for having patterns configured
        if patterns_count >= 10:
            score += 10
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Guardrail block rate: {block_rate:.1%}",
            f"False negative rate: {false_negative_rate:.1%}",
            f"Active detection patterns: {patterns_count}",
        ]
        benchmarks = ["guardrails"]

        if false_negative_rate > 0.10:
            recommendation = "CRITICAL: high false negative rate — unsafe content passing through. Add detection patterns."
        elif block_rate < 0.90:
            recommendation = "Improve guardrail coverage — target ≥ 90% block rate."
        else:
            recommendation = "Safety compliance is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_tools(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.TOOL_RELIABILITY
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=0.0,
                status="fail",
                evidence=["No tool reliability data available."],
                recommendation="Run structured-output benchmarks to assess tool reliability.",
            )

        success_rate = data.get("tool_success_rate", 0.0)
        schema_compliance = data.get("schema_compliance_rate", 0.0)
        hallucinated_tools = data.get("hallucinated_tool_rate", 0.0)

        score = (success_rate * 45.0) + (schema_compliance * 45.0) - (hallucinated_tools * 50.0)
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Tool call success rate: {success_rate:.1%}",
            f"Schema compliance rate: {schema_compliance:.1%}",
            f"Hallucinated tool rate: {hallucinated_tools:.1%}",
        ]
        benchmarks = ["structured-output", "tool-use"]

        if hallucinated_tools > 0.05:
            recommendation = "CRITICAL: agent hallucinates non-existent tools. Fix tool definitions and prompt."
        elif success_rate < 0.85:
            recommendation = "Improve tool call reliability — target ≥ 85% success rate."
        else:
            recommendation = "Tool reliability is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_response_quality(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.RESPONSE_QUALITY
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=0.0,
                status="fail",
                evidence=["No LLM-as-Judge quality data available."],
                recommendation="Run LLM-as-Judge evaluation to assess response quality.",
            )

        accuracy = data.get("accuracy", 0.0)
        completeness = data.get("completeness", 0.0)
        relevance = data.get("relevance", 0.0)
        clarity = data.get("clarity", 0.0)

        # Weighted average: accuracy 35%, completeness 25%, relevance 25%, clarity 15%
        score = (
            accuracy * 0.35
            + completeness * 0.25
            + relevance * 0.25
            + clarity * 0.15
        )
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Accuracy: {accuracy:.1f}",
            f"Completeness: {completeness:.1f}",
            f"Relevance: {relevance:.1f}",
            f"Clarity: {clarity:.1f}",
        ]
        benchmarks = ["llm-judge", "model-benchmark"]

        if accuracy < 60.0:
            recommendation = "CRITICAL: low accuracy in responses. Review prompt and retrieval quality."
        elif score < 70.0:
            recommendation = "Improve response quality — target composite ≥ 70."
        else:
            recommendation = "Response quality is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_retrieval(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.RETRIEVAL_QUALITY
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=0.0,
                status="fail",
                evidence=["No retrieval quality data available."],
                recommendation="Run retrieval benchmarks to assess search quality.",
            )

        citation_precision = data.get("citation_precision", 0.0)
        relevance_score = data.get("relevance_score", 0.0)
        mrr = data.get("mrr", 0.0)

        score = (citation_precision * 40.0) + (relevance_score * 35.0) + (mrr * 25.0)
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Citation precision: {citation_precision:.3f}",
            f"Relevance score: {relevance_score:.3f}",
            f"MRR: {mrr:.3f}",
        ]
        benchmarks = ["retrieval-qa", "hybrid-retrieval"]

        if citation_precision < 0.60:
            recommendation = "CRITICAL: citation precision below 60%. Review retrieval pipeline and chunking."
        elif score < 70.0:
            recommendation = "Improve retrieval quality — review embedding model and chunking strategy."
        else:
            recommendation = "Retrieval quality is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_latency(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.LATENCY_EFFICIENCY
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=50.0,
                status="warn",
                evidence=["No latency data available — assuming acceptable defaults."],
                recommendation="Collect latency metrics for accurate assessment.",
            )

        avg_ms = data.get("avg_latency_ms", 0)
        p95_ms = data.get("p95_latency_ms", 0)
        budget_compliance = data.get("budget_compliance_rate", 1.0)

        # Score based on latency thresholds: excellent < 500ms, good < 2000ms, poor > 5000ms
        if avg_ms < 500:
            latency_score = 95.0
        elif avg_ms < 1000:
            latency_score = 85.0
        elif avg_ms < 2000:
            latency_score = 70.0
        elif avg_ms < 5000:
            latency_score = 50.0
        else:
            latency_score = 20.0

        # Adjust for budget compliance
        score = (latency_score * 0.7) + (budget_compliance * 30.0)
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Average latency: {avg_ms:.0f}ms",
            f"P95 latency: {p95_ms:.0f}ms",
            f"Budget compliance: {budget_compliance:.1%}",
        ]
        benchmarks = ["latency-budget"]

        if avg_ms > 5000:
            recommendation = "CRITICAL: average latency exceeds 5 seconds. Optimise agent pipeline."
        elif avg_ms > 2000:
            recommendation = "Reduce latency — target < 2s average for production."
        else:
            recommendation = "Latency is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_memory(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.MEMORY_CONSISTENCY
        if not data:
            return ReadinessDimensionScore(
                dimension=dim,
                score=50.0,
                status="warn",
                evidence=["No memory evaluation data available."],
                recommendation="Run memory benchmarks to assess multi-turn consistency.",
            )

        recall_precision = data.get("recall_precision", 0.0)
        recall_rate = data.get("recall_rate", 0.0)
        hallucination_rate = data.get("hallucination_rate", 0.0)
        f1_score = data.get("f1_score", 0.0)

        score = (recall_precision * 30.0) + (recall_rate * 30.0) + (f1_score * 40.0) - (hallucination_rate * 50.0)
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Recall precision: {recall_precision:.3f}",
            f"Recall rate: {recall_rate:.3f}",
            f"F1 score: {f1_score:.3f}",
            f"Memory hallucination rate: {hallucination_rate:.1%}",
        ]
        benchmarks = ["memory-episodic", "memory-semantic", "memory-working"]

        if hallucination_rate > 0.15:
            recommendation = "CRITICAL: high memory hallucination rate. Review memory storage and retrieval."
        elif f1_score < 0.60:
            recommendation = "Improve memory consistency — target F1 ≥ 0.60."
        else:
            recommendation = "Memory consistency is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    def _score_multi_agent(self, data: dict | None) -> ReadinessDimensionScore:
        dim = ReadinessDimension.MULTI_AGENT_COORDINATION
        if not data:
            # Multi-agent is optional — neutral score when N/A
            return ReadinessDimensionScore(
                dimension=dim,
                score=50.0,
                status="pass",
                evidence=["No multi-agent evaluation data — N/A for single-agent deployments."],
                recommendation="Multi-agent coordination not assessed. Run multi-agent benchmarks if applicable.",
            )

        coordination_score = data.get("coordination_score", 0.0)
        message_efficiency = data.get("message_efficiency", 0.0)
        task_completion_rate = data.get("task_completion_rate", 0.0)

        score = (coordination_score * 40.0) + (message_efficiency * 30.0) + (task_completion_rate * 30.0)
        score = min(100.0, max(0.0, score))

        evidence = [
            f"Coordination score: {coordination_score:.3f}",
            f"Message efficiency: {message_efficiency:.3f}",
            f"Task completion rate: {task_completion_rate:.1%}",
        ]
        benchmarks = ["multi-agent"]

        if task_completion_rate < 0.50:
            recommendation = "CRITICAL: multi-agent task completion below 50%."
        elif coordination_score < 0.60:
            recommendation = "Improve multi-agent coordination — review supervisor-worker topology."
        else:
            recommendation = "Multi-agent coordination is healthy."

        return self._finalise_score(dim, score, evidence, benchmarks, recommendation)

    # ── Aggregation ────────────────────────────────────────────────

    def _compute_composite(self, dimension_scores: list[ReadinessDimensionScore]) -> float:
        """Weighted average of dimension scores."""
        total_weight = 0.0
        weighted_sum = 0.0

        for ds in dimension_scores:
            w = self.weights.get(ds.dimension, ds.dimension.weight)
            weighted_sum += ds.score * w
            total_weight += w

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 1)

    def _assign_tier(
        self, composite: float, dimension_scores: list[ReadinessDimensionScore]
    ) -> ReadinessTier:
        """Assign a readiness tier from composite score and dimension statuses."""
        fail_count = sum(1 for d in dimension_scores if d.status == "fail")
        warn_count = sum(1 for d in dimension_scores if d.status == "warn")

        # Critical: any dimension failed OR composite < 50
        if fail_count > 0 or composite < READINESS_THRESHOLDS["composite_needs_work"]:
            return ReadinessTier.CRITICAL_ISSUES

        # Needs Work: composite < 75 OR 3+ warnings
        if (
            composite < READINESS_THRESHOLDS["composite_conditional"]
            or warn_count > READINESS_THRESHOLDS["max_warnings_for_conditional"]
        ):
            return ReadinessTier.NEEDS_WORK

        # Conditional: 1-2 warnings
        if warn_count > 0:
            return ReadinessTier.CONDITIONAL

        # Production Ready: >= 90 composite, all pass
        if composite >= READINESS_THRESHOLDS["composite_production_ready"]:
            return ReadinessTier.PRODUCTION_READY

        return ReadinessTier.CONDITIONAL

    def _collect_critical_findings(
        self, dimension_scores: list[ReadinessDimensionScore], tier: ReadinessTier
    ) -> list[str]:
        findings = []

        if tier == ReadinessTier.CRITICAL_ISSUES:
            findings.append("AGENT IS NOT READY FOR PRODUCTION — critical issues detected.")

        for ds in dimension_scores:
            if ds.status == "fail":
                findings.append(
                    f"[FAIL] {ds.dimension.label}: {ds.recommendation}"
                )
            elif ds.status == "warn" and tier == ReadinessTier.CRITICAL_ISSUES:
                findings.append(
                    f"[WARN] {ds.dimension.label}: score={ds.score:.1f}"
                )

        if not findings:
            findings.append("No critical findings — agent is production-ready.")

        return findings

    def _generate_recommendations(
        self, dimension_scores: list[ReadinessDimensionScore], tier: ReadinessTier
    ) -> list[str]:
        recs = []

        # Tier-level recommendation
        if tier == ReadinessTier.PRODUCTION_READY:
            recs.append("✓ Agent is ready for production deployment.")
            recs.append("  Monitor with agentops alerting and dashboard for ongoing health.")
            recs.append("  Re-assess after any model, prompt, or tool registry change.")
        elif tier == ReadinessTier.CONDITIONAL:
            recs.append("⚠ Agent can be deployed with monitoring in place.")
            recs.append("  Configure agentops alerting to catch degradation early.")
            recs.append("  Address warning dimensions before removing conditional status.")
        elif tier == ReadinessTier.NEEDS_WORK:
            recs.append("✗ Agent needs work before deployment.")
            recs.append("  Address all failing and warning dimensions below.")
            recs.append("  Re-run readiness assessment after fixes.")
        else:
            recs.append("!! CRITICAL: Do NOT deploy this agent to production.")
            recs.append("  Address all failing dimensions below immediately.")
            recs.append("  Schedule re-assessment after each fix.")

        # Dimension-level recommendations
        for ds in dimension_scores:
            if ds.status in ("fail", "warn"):
                recs.append(f"  [{ds.dimension.label}] {ds.recommendation}")

        return recs

    # ── Helpers ────────────────────────────────────────────────────

    def _finalise_score(
        self,
        dimension: ReadinessDimension,
        score: float,
        evidence: list[str],
        benchmarks: list[str],
        recommendation: str,
    ) -> ReadinessDimensionScore:
        score = round(min(100.0, max(0.0, score)), 1)

        if score >= dimension.pass_threshold:
            status = "pass"
        elif score >= dimension.warn_threshold:
            status = "warn"
        else:
            status = "fail"

        return ReadinessDimensionScore(
            dimension=dimension,
            score=score,
            status=status,
            evidence=evidence,
            benchmarks_used=benchmarks,
            recommendation=recommendation,
        )
