"""
Compliance Evaluation Benchmarks — scenario-based EU AI Act compliance testing.

Provides structured evaluation scenarios across risk tiers, sector-specific
compliance checks, and integration with existing AgentOps evaluation harness.
Each scenario defines an agent system configuration, expected risk tier, and
required compliance status.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

from agentops.compliance.classifier import RiskTier, RiskClassification
from agentops.compliance.auditor import (
    ComplianceStatus,
    ComplianceReport,
    ComplianceAuditor,
    FindingSeverity,
)


class ComplianceScenarioDomain(str, Enum):
    """Sector domains for compliance evaluation scenarios."""
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    RECRUITMENT = "recruitment"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    GENERAL_PURPOSE = "general_purpose"
    CODING_ASSISTANT = "coding_assistant"
    SOCIAL_SCORING = "social_scoring"
    MANIPULATIVE = "manipulative"
    CHILD_TARGETED = "child_targeted"
    LAW_ENFORCEMENT = "law_enforcement"


@dataclass
class ComplianceScenario:
    """A compliance evaluation scenario with expected outcomes."""
    name: str
    domain: ComplianceScenarioDomain
    description: str
    # System configuration
    makes_autonomous_decisions: bool = False
    processes_personal_data: bool = False
    processes_special_category_data: bool = False
    affects_legal_rights: bool = False
    affects_physical_safety: bool = False
    involves_profiling: bool = False
    involves_manipulation: bool = False
    involves_social_scoring: bool = False
    targets_vulnerable_population: bool = False
    is_safety_component: bool = False
    has_human_oversight: bool = True
    is_critical_system: bool = False
    # Expected outcomes
    expected_risk_tier: RiskTier = RiskTier.MINIMAL
    expected_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    expected_min_score: float = 80.0


@dataclass
class ComplianceEvalResult:
    """Result of evaluating a single compliance scenario."""
    scenario: ComplianceScenario
    passed: bool
    actual_risk_tier: RiskTier
    expected_risk_tier: RiskTier
    expected_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    actual_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    actual_score: float = 0.0
    report: Optional[ComplianceReport] = None
    errors: List[str] = field(default_factory=list)

    @property
    def risk_tier_correct(self) -> bool:
        return self.actual_risk_tier == self.expected_risk_tier

    @property
    def status_acceptable(self) -> bool:
        return self.actual_status in (
            self.expected_status,
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.COMPLIANT_WITH_OBSERVATIONS,
        )


@dataclass
class ComplianceBenchmark:
    """A collection of compliance evaluation scenarios."""
    name: str
    description: str
    scenarios: List[ComplianceScenario] = field(default_factory=list)


class ComplianceEvaluator:
    """
    Evaluates compliance audit accuracy across defined scenarios.

    Usage:
        >>> evaluator = ComplianceEvaluator()
        >>> results = evaluator.evaluate(COMPLIANCE_BENCHMARK)
        >>> print(evaluator.format_results(results))
    """

    def __init__(self, auditor: Optional[ComplianceAuditor] = None):
        self.auditor = auditor or ComplianceAuditor()

    def evaluate(
        self,
        benchmark: ComplianceBenchmark,
    ) -> List[ComplianceEvalResult]:
        """Run all scenarios in a benchmark and collect results."""
        results = []
        for scenario in benchmark.scenarios:
            report = self.auditor.audit(
                domain=scenario.domain.value,
                description=scenario.description,
                makes_autonomous_decisions=scenario.makes_autonomous_decisions,
                processes_personal_data=scenario.processes_personal_data,
                processes_special_category_data=scenario.processes_special_category_data,
                affects_legal_rights=scenario.affects_legal_rights,
                affects_physical_safety=scenario.affects_physical_safety,
                involves_profiling=scenario.involves_profiling,
                involves_manipulation=scenario.involves_manipulation,
                involves_social_scoring=scenario.involves_social_scoring,
                targets_vulnerable_population=scenario.targets_vulnerable_population,
                is_safety_component=scenario.is_safety_component,
                has_human_oversight=scenario.has_human_oversight,
                is_critical_system=scenario.is_critical_system,
            )
            errors = []

            # Validate risk tier
            tier_correct = report.risk_tier == scenario.expected_risk_tier
            if not tier_correct:
                errors.append(
                    f"Risk tier mismatch: expected {scenario.expected_risk_tier.value}, "
                    f"got {report.risk_tier.value}"
                )

            # Validate score threshold
            score_ok = report.compliance_score >= scenario.expected_min_score
            if not score_ok:
                errors.append(
                    f"Score below threshold: expected >= {scenario.expected_min_score}, "
                    f"got {report.compliance_score:.0f}"
                )

            passed = tier_correct and score_ok and (
                report.status == scenario.expected_status
                or report.status in (
                    ComplianceStatus.COMPLIANT,
                    ComplianceStatus.COMPLIANT_WITH_OBSERVATIONS,
                )
            )

            results.append(
                ComplianceEvalResult(
                    scenario=scenario,
                    passed=passed,
                    actual_risk_tier=report.risk_tier,
                    expected_risk_tier=scenario.expected_risk_tier,
                    actual_status=report.status,
                    actual_score=report.compliance_score,
                    report=report,
                    errors=errors,
                )
            )
        return results

    def format_results(self, results: List[ComplianceEvalResult]) -> str:
        """Format evaluation results as a readable markdown report."""
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        tier_correct = sum(1 for r in results if r.risk_tier_correct)

        lines = [
            "# EU AI Act Compliance Benchmark Results",
            "",
            f"**Passed:** {passed}/{total} ({passed/total*100:.0f}%)" if total > 0 else "**No results**",
            f"**Risk Tier Accuracy:** {tier_correct}/{total} ({tier_correct/total*100:.0f}%)" if total > 0 else "",
            "",
            "| # | Scenario | Domain | Expected | Actual | Score | Pass |",
            "|---|----------|--------|----------|--------|-------|------|",
        ]

        for i, r in enumerate(results, 1):
            icon = "✓" if r.passed else "✗"
            errors = "; ".join(r.errors) if r.errors else ""
            extra = f" ({errors})" if errors else ""
            lines.append(
                f"| {i} | {r.scenario.name} | {r.scenario.domain.value} | "
                f"{r.expected_risk_tier.value} | {r.actual_risk_tier.value} | "
                f"{r.actual_score:.0f} | {icon}{extra} |"
            )

        lines.extend([
            "",
            "## Risk Tier Distribution",
            "",
        ])
        from collections import Counter
        tier_counts = Counter(r.actual_risk_tier for r in results)
        for tier in RiskTier:
            count = tier_counts.get(tier, 0)
            lines.append(f"- **{tier.value}:** {count} scenarios")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standard Compliance Benchmark
# ---------------------------------------------------------------------------

COMPLIANCE_BENCHMARK = ComplianceBenchmark(
    name="EU AI Act Standard Compliance Benchmark",
    description="Comprehensive compliance evaluation covering all risk tiers and key sectors",
    scenarios=[
        # --- Minimal risk scenarios ---
        ComplianceScenario(
            name="General Coding Assistant",
            domain=ComplianceScenarioDomain.CODING_ASSISTANT,
            description="AI coding assistant with no personal data access",
            makes_autonomous_decisions=False,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.MINIMAL,
            expected_status=ComplianceStatus.COMPLIANT,
        ),
        ComplianceScenario(
            name="Spam Filter",
            domain=ComplianceScenarioDomain.GENERAL_PURPOSE,
            description="Email spam filter with no profiling of individuals",
            makes_autonomous_decisions=True,
            processes_personal_data=False,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.MINIMAL,
            expected_status=ComplianceStatus.COMPLIANT,
        ),

        # --- Limited risk scenarios ---
        ComplianceScenario(
            name="Unsupervised Chatbot with PII",
            domain=ComplianceScenarioDomain.GENERAL_PURPOSE,
            description="Customer chatbot processing PII without human oversight",
            makes_autonomous_decisions=False,
            processes_personal_data=True,
            has_human_oversight=False,
            expected_risk_tier=RiskTier.LIMITED,
            expected_status=ComplianceStatus.COMPLIANT_WITH_OBSERVATIONS,
        ),
        ComplianceScenario(
            name="Autonomous Content Generator No Oversight",
            domain=ComplianceScenarioDomain.GENERAL_PURPOSE,
            description="AI content generator without human review",
            makes_autonomous_decisions=True,
            processes_personal_data=False,
            has_human_oversight=False,
            expected_risk_tier=RiskTier.LIMITED,
            expected_status=ComplianceStatus.COMPLIANT_WITH_OBSERVATIONS,
        ),

        # --- High risk scenarios ---
        ComplianceScenario(
            name="Clinical Decision Support",
            domain=ComplianceScenarioDomain.HEALTHCARE,
            description="Clinical decision support agent operating in a hospital",
            makes_autonomous_decisions=True,
            processes_personal_data=True,
            processes_special_category_data=True,
            affects_legal_rights=True,
            affects_physical_safety=True,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.HIGH,
            expected_status=ComplianceStatus.NON_COMPLIANT,
            expected_min_score=50.0,
        ),
        ComplianceScenario(
            name="AI Recruitment Screener",
            domain=ComplianceScenarioDomain.RECRUITMENT,
            description="AI agent screening job applicants and ranking candidates",
            makes_autonomous_decisions=True,
            processes_personal_data=True,
            affects_legal_rights=True,
            involves_profiling=True,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.HIGH,
            expected_status=ComplianceStatus.NON_COMPLIANT,
            expected_min_score=50.0,
        ),
        ComplianceScenario(
            name="Educational Assessment Agent",
            domain=ComplianceScenarioDomain.EDUCATION,
            description="AI agent determining student admissions and grading",
            makes_autonomous_decisions=True,
            processes_personal_data=True,
            affects_legal_rights=True,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.HIGH,
            expected_status=ComplianceStatus.NON_COMPLIANT,
            expected_min_score=50.0,
        ),
        ComplianceScenario(
            name="Credit Scoring Agent",
            domain=ComplianceScenarioDomain.FINANCE,
            description="AI agent determining creditworthiness for loan applications",
            makes_autonomous_decisions=True,
            processes_personal_data=True,
            affects_legal_rights=True,
            involves_profiling=True,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.HIGH,
            expected_status=ComplianceStatus.NON_COMPLIANT,
            expected_min_score=50.0,
        ),
        ComplianceScenario(
            name="Critical Infrastructure Monitoring",
            domain=ComplianceScenarioDomain.CRITICAL_INFRASTRUCTURE,
            description="Agent monitoring power grid with autonomous control capability",
            makes_autonomous_decisions=True,
            affects_physical_safety=True,
            is_critical_system=True,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.HIGH,
            expected_status=ComplianceStatus.NON_COMPLIANT,
            expected_min_score=50.0,
        ),
        ComplianceScenario(
            name="Safety Component — Medical Device",
            domain=ComplianceScenarioDomain.HEALTHCARE,
            description="AI agent serving as safety component in a medical device",
            makes_autonomous_decisions=True,
            processes_special_category_data=True,
            affects_physical_safety=True,
            is_safety_component=True,
            has_human_oversight=True,
            expected_risk_tier=RiskTier.HIGH,
            expected_status=ComplianceStatus.NON_COMPLIANT,
            expected_min_score=50.0,
        ),

        # --- Prohibited scenarios ---
        ComplianceScenario(
            name="Social Scoring System",
            domain=ComplianceScenarioDomain.SOCIAL_SCORING,
            description="Government social scoring system evaluating citizen trustworthiness",
            involves_social_scoring=True,
            makes_autonomous_decisions=True,
            processes_personal_data=True,
            has_human_oversight=False,
            expected_risk_tier=RiskTier.UNACCEPTABLE,
            expected_status=ComplianceStatus.PROHIBITED,
            expected_min_score=0.0,
        ),
        ComplianceScenario(
            name="Child Manipulation Chatbot",
            domain=ComplianceScenarioDomain.MANIPULATIVE,
            description="AI chatbot designed to manipulate children into purchases",
            involves_manipulation=True,
            targets_vulnerable_population=True,
            makes_autonomous_decisions=True,
            has_human_oversight=False,
            expected_risk_tier=RiskTier.UNACCEPTABLE,
            expected_status=ComplianceStatus.PROHIBITED,
            expected_min_score=0.0,
        ),
    ],
)


def evaluate_compliance(
    benchmark: Optional[ComplianceBenchmark] = None,
) -> List[ComplianceEvalResult]:
    """Run compliance evaluation benchmark.

    Args:
        benchmark: Benchmark to evaluate (defaults to COMPLIANCE_BENCHMARK)

    Returns:
        List of ComplianceEvalResult
    """
    evaluator = ComplianceEvaluator()
    return evaluator.evaluate(benchmark or COMPLIANCE_BENCHMARK)


def format_compliance_eval_report(
    results: Optional[List[ComplianceEvalResult]] = None,
) -> str:
    """Format compliance evaluation results.

    Args:
        results: Evaluation results (defaults to running the benchmark)

    Returns:
        Markdown-formatted evaluation report
    """
    if results is None:
        results = evaluate_compliance()
    evaluator = ComplianceEvaluator()
    return evaluator.format_results(results)
