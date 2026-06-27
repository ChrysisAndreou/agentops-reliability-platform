"""
State models for production readiness assessment.

Defines the 8 dimensions, 4 readiness tiers, scoring thresholds,
and structured report types that form the readiness assessment framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReadinessTier(str, Enum):
    """Production readiness classification tiers.

    PRODUCTION_READY — all dimensions green; safe to deploy.
    CONDITIONAL      — minor concerns; deploy with monitoring.
    NEEDS_WORK       — significant gaps; fix before deploying.
    CRITICAL_ISSUES  — multiple failures; do NOT deploy.
    """

    PRODUCTION_READY = "production_ready"
    CONDITIONAL = "conditional"
    NEEDS_WORK = "needs_work"
    CRITICAL_ISSUES = "critical_issues"

    @property
    def label(self) -> str:
        return {
            ReadinessTier.PRODUCTION_READY: "Production Ready",
            ReadinessTier.CONDITIONAL: "Conditional — Deploy with Monitoring",
            ReadinessTier.NEEDS_WORK: "Needs Work — Fix Before Deploying",
            ReadinessTier.CRITICAL_ISSUES: "Critical Issues — Do NOT Deploy",
        }[self]

    @property
    def exit_code(self) -> int:
        """CI-friendly exit codes."""
        return {
            ReadinessTier.PRODUCTION_READY: 0,
            ReadinessTier.CONDITIONAL: 0,
            ReadinessTier.NEEDS_WORK: 1,
            ReadinessTier.CRITICAL_ISSUES: 2,
        }[self]


class ReadinessDimension(str, Enum):
    """The 8 dimensions of agent production readiness."""

    VERIFICATION_QUALITY = "verification_quality"
    SAFETY_COMPLIANCE = "safety_compliance"
    TOOL_RELIABILITY = "tool_reliability"
    RESPONSE_QUALITY = "response_quality"
    RETRIEVAL_QUALITY = "retrieval_quality"
    LATENCY_EFFICIENCY = "latency_efficiency"
    MEMORY_CONSISTENCY = "memory_consistency"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"

    @property
    def label(self) -> str:
        return {
            ReadinessDimension.VERIFICATION_QUALITY: "Verification Quality",
            ReadinessDimension.SAFETY_COMPLIANCE: "Safety Compliance",
            ReadinessDimension.TOOL_RELIABILITY: "Tool Reliability",
            ReadinessDimension.RESPONSE_QUALITY: "Response Quality",
            ReadinessDimension.RETRIEVAL_QUALITY: "Retrieval Quality",
            ReadinessDimension.LATENCY_EFFICIENCY: "Latency & Efficiency",
            ReadinessDimension.MEMORY_CONSISTENCY: "Memory Consistency",
            ReadinessDimension.MULTI_AGENT_COORDINATION: "Multi-Agent Coordination",
        }[self]

    @property
    def weight(self) -> float:
        """Default weight in composite score (sum = 1.0)."""
        return {
            ReadinessDimension.VERIFICATION_QUALITY: 0.20,
            ReadinessDimension.SAFETY_COMPLIANCE: 0.18,
            ReadinessDimension.TOOL_RELIABILITY: 0.15,
            ReadinessDimension.RESPONSE_QUALITY: 0.15,
            ReadinessDimension.RETRIEVAL_QUALITY: 0.12,
            ReadinessDimension.LATENCY_EFFICIENCY: 0.08,
            ReadinessDimension.MEMORY_CONSISTENCY: 0.07,
            ReadinessDimension.MULTI_AGENT_COORDINATION: 0.05,
        }[self]

    @property
    def pass_threshold(self) -> float:
        """Score >= this is a PASS."""
        return 80.0

    @property
    def warn_threshold(self) -> float:
        """Score >= this but < pass_threshold is a WARN."""
        return 60.0


# Convenience lists
READINESS_DIMENSIONS: list[ReadinessDimension] = list(ReadinessDimension)
READINESS_THRESHOLDS: dict[str, Any] = {
    "composite_production_ready": 85.0,
    "composite_conditional": 75.0,
    "composite_needs_work": 50.0,
    "max_warnings_for_conditional": 2,
}


@dataclass
class ReadinessDimensionScore:
    """Score for a single readiness dimension with evidence."""

    dimension: ReadinessDimension
    score: float  # 0-100
    status: str  # "pass", "warn", "fail"
    evidence: list[str] = field(default_factory=list)
    benchmarks_used: list[str] = field(default_factory=list)
    recommendation: str = ""

    def __post_init__(self):
        if not self.evidence:
            self.evidence = [f"{self.dimension.label}: score={self.score:.1f}"]


@dataclass
class ReadinessReport:
    """Complete production readiness assessment report."""

    agent_name: str
    agent_version: str
    tier: ReadinessTier
    composite_score: float  # 0-100 weighted average
    dimension_scores: list[ReadinessDimensionScore]
    total_benchmarks_run: int = 0
    total_traces_analyzed: int = 0
    critical_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    run_timestamp: str = ""
    assessment_duration_ms: float = 0.0

    @property
    def pass_count(self) -> int:
        return sum(1 for d in self.dimension_scores if d.status == "pass")

    @property
    def warn_count(self) -> int:
        return sum(1 for d in self.dimension_scores if d.status == "warn")

    @property
    def fail_count(self) -> int:
        return sum(1 for d in self.dimension_scores if d.status == "fail")

    @property
    def is_deployable(self) -> bool:
        return self.tier in (ReadinessTier.PRODUCTION_READY, ReadinessTier.CONDITIONAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "tier": self.tier.value,
            "tier_label": self.tier.label,
            "composite_score": self.composite_score,
            "deployable": self.is_deployable,
            "exit_code": self.tier.exit_code,
            "dimensions": [
                {
                    "dimension": d.dimension.value,
                    "label": d.dimension.label,
                    "score": d.score,
                    "status": d.status,
                    "weight": d.dimension.weight,
                    "evidence": d.evidence,
                    "benchmarks_used": d.benchmarks_used,
                    "recommendation": d.recommendation,
                }
                for d in self.dimension_scores
            ],
            "summary": {
                "total_dimensions": len(self.dimension_scores),
                "pass": self.pass_count,
                "warn": self.warn_count,
                "fail": self.fail_count,
            },
            "total_benchmarks_run": self.total_benchmarks_run,
            "total_traces_analyzed": self.total_traces_analyzed,
            "critical_findings": self.critical_findings,
            "recommendations": self.recommendations,
            "run_timestamp": self.run_timestamp,
            "assessment_duration_ms": self.assessment_duration_ms,
        }
