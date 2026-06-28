"""
Compliance Auditing Engine — systematic EU AI Act compliance evaluation.

Performs automated compliance audits of AI agent systems against EU AI Act
requirements. Evaluates risk classification, conformity assessment coverage,
documentation completeness, and produces structured compliance reports with
pass/fail gating, non-conformity findings, and remediation guidance.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

from agentops.compliance.classifier import (
    RiskTier,
    RiskClassification,
    RiskClassifier,
    AnnexIIICategory,
)
from agentops.compliance.requirements import (
    ConformityRequirement,
    RequirementCategory,
    RequirementStatus,
    RequirementsProfile,
    get_requirements_for_tier,
    build_requirements_checklist,
)
from agentops.compliance.documentation import (
    TechnicalDocumentation,
    validate_documentation,
)


class FindingSeverity(str, Enum):
    """Severity of a compliance finding."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


class ComplianceStatus(str, Enum):
    """Overall compliance status after audit."""
    COMPLIANT = "compliant"
    COMPLIANT_WITH_OBSERVATIONS = "compliant_with_observations"
    NON_COMPLIANT = "non_compliant"
    PROHIBITED = "prohibited"


@dataclass
class Finding:
    """A single compliance finding identified during audit."""
    id: str
    severity: FindingSeverity
    category: RequirementCategory
    article: str
    requirement_id: str
    description: str
    evidence: str = ""
    remediation: str = ""


@dataclass
class ComplianceReport:
    """Complete compliance audit report."""
    status: ComplianceStatus
    risk_tier: RiskTier
    risk_classification: Optional[RiskClassification] = None
    requirements_profile: Optional[RequirementsProfile] = None
    documentation_completeness: float = 0.0
    findings: List[Finding] = field(default_factory=list)
    passed_requirements: int = 0
    failed_requirements: int = 0
    partial_requirements: int = 0
    total_requirements: int = 0
    compliance_score: float = 0.0  # 0.0-100.0
    audit_timestamp: str = ""
    auditor_version: str = "v0.28"

    @property
    def is_compliant(self) -> bool:
        return self.status in (ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_OBSERVATIONS)

    @property
    def critical_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == FindingSeverity.CRITICAL]

    @property
    def major_findings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == FindingSeverity.MAJOR]

    def summary(self) -> str:
        """One-line compliance summary."""
        return (
            f"{self.status.value.upper()}: {self.risk_tier.value} risk, "
            f"score={self.compliance_score:.0f}/100, "
            f"{self.passed_requirements}/{self.total_requirements} passed, "
            f"{len(self.findings)} findings"
        )


class ComplianceAuditor:
    """
    Audits AI agent systems for EU AI Act compliance.

    Performs a multi-phase audit:
    1. Risk classification — determine the applicable risk tier
    2. Requirements evaluation — check conformity assessment coverage
    3. Documentation review — verify Annex IV documentation completeness
    4. Findings compilation — aggregate non-conformities with remediation

    Usage:
        >>> auditor = ComplianceAuditor()
        >>> report = auditor.audit(
        ...     domain="healthcare",
        ...     description="Clinical decision support agent",
        ...     makes_autonomous_decisions=True,
        ...     technical_documentation=doc,
        ...     evidence_available={"risk_assessment_document": True, ...},
        ...     has_human_oversight=True,
        ... )
        >>> print(report.summary())
        COMPLIANT_WITH_OBSERVATIONS: high risk, score=82/100, 15/18 passed, 3 findings
    """

    def audit(
        self,
        domain: str = "",
        description: str = "",
        # Risk classification inputs
        makes_autonomous_decisions: bool = False,
        processes_personal_data: bool = False,
        processes_special_category_data: bool = False,
        affects_legal_rights: bool = False,
        affects_physical_safety: bool = False,
        involves_profiling: bool = False,
        involves_manipulation: bool = False,
        involves_social_scoring: bool = False,
        targets_vulnerable_population: bool = False,
        is_safety_component: bool = False,
        has_human_oversight: bool = True,
        is_critical_system: bool = False,
        # Documentation
        technical_documentation: Optional[TechnicalDocumentation] = None,
        # Evidence
        evidence_available: Optional[Dict[str, bool]] = None,
    ) -> ComplianceReport:
        """
        Perform a full compliance audit of an AI agent system.

        Returns a ComplianceReport with status, findings, and remediation guidance.
        """
        findings: List[Finding] = []
        finding_counter = [0]

        def _add_finding(
            severity: FindingSeverity,
            category: RequirementCategory,
            article: str,
            req_id: str,
            description: str,
            remediation: str = "",
        ):
            finding_counter[0] += 1
            findings.append(
                Finding(
                    id=f"F-{finding_counter[0]:03d}",
                    severity=severity,
                    category=category,
                    article=article,
                    requirement_id=req_id,
                    description=description,
                    remediation=remediation,
                )
            )

        # Phase 1: Risk classification
        classifier = RiskClassifier()
        classification = classifier.classify(
            domain=domain,
            description=description,
            makes_autonomous_decisions=makes_autonomous_decisions,
            processes_personal_data=processes_personal_data,
            processes_special_category_data=processes_special_category_data,
            affects_legal_rights=affects_legal_rights,
            affects_physical_safety=affects_physical_safety,
            involves_profiling=involves_profiling,
            involves_manipulation=involves_manipulation,
            involves_social_scoring=involves_social_scoring,
            targets_vulnerable_population=targets_vulnerable_population,
            is_safety_component=is_safety_component,
            has_human_oversight=has_human_oversight,
            is_critical_system=is_critical_system,
        )

        if classification.tier == RiskTier.UNACCEPTABLE:
            _add_finding(
                FindingSeverity.CRITICAL,
                RequirementCategory.TRANSPARENCY,
                "Article 5",
                "UN-PH-01",
                f"System classified as UNACCEPTABLE risk: {classification.rationale}",
                "System must not be deployed. Withdraw immediately from the EU market.",
            )
            return ComplianceReport(
                status=ComplianceStatus.PROHIBITED,
                risk_tier=RiskTier.UNACCEPTABLE,
                risk_classification=classification,
                findings=findings,
                passed_requirements=0,
                failed_requirements=1,
                total_requirements=1,
                compliance_score=0.0,
                audit_timestamp=datetime.now().isoformat(),
            )

        # Phase 2: Requirements evaluation
        evidence = evidence_available or {}
        profile = build_requirements_checklist(classification.tier, evidence)

        for req in profile.requirements:
            if req.status == RequirementStatus.FAIL:
                severity = (
                    FindingSeverity.CRITICAL
                    if req.category in (
                        RequirementCategory.RISK_MANAGEMENT,
                        RequirementCategory.DATA_GOVERNANCE,
                        RequirementCategory.TECHNICAL_DOCUMENTATION,
                    )
                    else FindingSeverity.MAJOR
                )
                _add_finding(
                    severity,
                    req.category,
                    req.article,
                    req.id,
                    f"Failed requirement: {req.description}",
                    f"Provide evidence: {', '.join(req.evidence_required)}",
                )
            elif req.status == RequirementStatus.PARTIAL:
                _add_finding(
                    FindingSeverity.MINOR,
                    req.category,
                    req.article,
                    req.id,
                    f"Partially satisfied: {req.description}",
                    f"Complete evidence: {', '.join(req.evidence_required)}",
                )

        # Phase 3: Documentation review
        doc_score = 0.0
        if technical_documentation is not None:
            doc_score = technical_documentation.overall_completeness()
            validation = validate_documentation(technical_documentation)

            for error in validation.get("errors", []):
                _add_finding(
                    FindingSeverity.MAJOR,
                    RequirementCategory.TECHNICAL_DOCUMENTATION,
                    "Article 11 / Annex IV",
                    "DOC-ERR",
                    f"Documentation error: {error}",
                    "Update technical documentation to include the missing element.",
                )

            for warning in validation.get("warnings", []):
                _add_finding(
                    FindingSeverity.MINOR,
                    RequirementCategory.TECHNICAL_DOCUMENTATION,
                    "Article 11 / Annex IV",
                    "DOC-WARN",
                    f"Documentation gap: {warning}",
                    "Consider adding this section to technical documentation.",
                )

            if doc_score < 0.95 and classification.tier == RiskTier.HIGH:
                _add_finding(
                    FindingSeverity.MAJOR,
                    RequirementCategory.TECHNICAL_DOCUMENTATION,
                    "Article 11",
                    "DOC-CMP",
                    f"Technical documentation only {doc_score:.1%} complete — "
                    f"threshold for high-risk systems is 95%",
                    "Complete all Annex IV documentation sections.",
                )
        elif classification.tier == RiskTier.HIGH:
            _add_finding(
                FindingSeverity.CRITICAL,
                RequirementCategory.TECHNICAL_DOCUMENTATION,
                "Article 11",
                "DOC-MIS",
                "No technical documentation provided — required for high-risk systems",
                "Generate Annex IV technical documentation immediately.",
            )

        # Phase 4: Status determination
        critical_count = len([f for f in findings if f.severity == FindingSeverity.CRITICAL])
        major_count = len([f for f in findings if f.severity == FindingSeverity.MAJOR])
        minor_count = len([f for f in findings if f.severity == FindingSeverity.MINOR])

        # Compliance score: 100 - deductions
        deductions = 0
        deductions += critical_count * 25
        deductions += major_count * 10
        deductions += minor_count * 3
        score = max(0.0, 100.0 - deductions)

        if critical_count > 0:
            status = ComplianceStatus.NON_COMPLIANT
        elif major_count > 0:
            status = ComplianceStatus.NON_COMPLIANT
        elif minor_count > 0:
            status = ComplianceStatus.COMPLIANT_WITH_OBSERVATIONS
        else:
            status = ComplianceStatus.COMPLIANT

        return ComplianceReport(
            status=status,
            risk_tier=classification.tier,
            risk_classification=classification,
            requirements_profile=profile,
            documentation_completeness=doc_score,
            findings=findings,
            passed_requirements=profile.passed,
            failed_requirements=profile.failed,
            partial_requirements=profile.partial,
            total_requirements=profile.total,
            compliance_score=score,
            audit_timestamp=datetime.now().isoformat(),
        )


def audit_agent_system(**kwargs) -> ComplianceReport:
    """Convenience function for one-shot compliance audit.

    Args:
        **kwargs: All parameters accepted by ComplianceAuditor.audit()

    Returns:
        ComplianceReport
    """
    return ComplianceAuditor().audit(**kwargs)


def format_compliance_report(report: ComplianceReport) -> str:
    """Format a ComplianceReport as a human-readable Markdown report.

    Args:
        report: The ComplianceReport to format

    Returns:
        Markdown-formatted compliance report string
    """
    lines = [
        "# EU AI Act Compliance Audit Report",
        "",
        f"**Audit Date:** {report.audit_timestamp[:19] if report.audit_timestamp else 'N/A'}",
        f"**Auditor Version:** {report.auditor_version}",
        f"**Regulatory Basis:** Regulation (EU) 2024/1689",
        "",
        f"## Overall Status: {report.status.value.upper()}",
        "",
        f"- **Risk Tier:** {report.risk_tier.value}",
        f"- **Compliance Score:** {report.compliance_score:.0f}/100",
        f"- **Requirements:** {report.passed_requirements}/{report.total_requirements} passed",
        f"- **Documentation Completeness:** {report.documentation_completeness:.1%}",
        f"- **Findings:** {len(report.findings)} total "
        f"({len(report.critical_findings)} critical, "
        f"{len(report.major_findings)} major, "
        f"{len([f for f in report.findings if f.severity == FindingSeverity.MINOR])} minor, "
        f"{len([f for f in report.findings if f.severity == FindingSeverity.OBSERVATION])} observations)",
    ]

    if report.risk_classification:
        rc = report.risk_classification
        lines.extend([
            "",
            "## Risk Classification",
            "",
            f"- **Tier:** {rc.tier.value}",
            f"- **Score:** {rc.score:.2f}",
            f"- **Rationale:** {rc.rationale}",
            f"- **Conformity Assessment Required:** {rc.requires_conformity_assessment}",
            f"- **CE Marking Required:** {rc.requires_ce_marking}",
            f"- **Transparency Required:** {rc.requires_transparency}",
        ])
        if rc.applicable_articles:
            lines.append(f"- **Applicable Articles:** {', '.join(rc.applicable_articles)}")
        if rc.annex_iii_categories:
            lines.append(f"- **Annex III Categories:** {', '.join(c.value for c in rc.annex_iii_categories)}")

    if report.findings:
        lines.extend(["", "## Findings", ""])
        for f in report.findings:
            icon = {
                FindingSeverity.CRITICAL: "🔴",
                FindingSeverity.MAJOR: "🟠",
                FindingSeverity.MINOR: "🟡",
                FindingSeverity.OBSERVATION: "🔵",
            }.get(f.severity, "⚪")
            lines.extend([
                f"### {icon} {f.id} — {f.severity.value.upper()}",
                f"",
                f"- **Category:** {f.category.value}",
                f"- **Article:** {f.article}",
                f"- **Requirement:** {f.requirement_id}",
                f"",
                f"{f.description}",
                f"",
            ])
            if f.remediation:
                lines.extend([
                    f"**Remediation:** {f.remediation}",
                    f"",
                ])
    else:
        lines.extend(["", "## Findings", "", "✅ No findings — fully compliant.", ""])

    if report.requirements_profile:
        lines.extend(["", "## Requirements Breakdown", ""])
        for cat, reqs in report.requirements_profile.by_category().items():
            passed = sum(1 for r in reqs if r.status == RequirementStatus.PASS)
            total = len(reqs)
            lines.append(f"- **{cat.value.replace('_', ' ').title()}:** {passed}/{total}")

    lines.extend([
        "",
        "---",
        "",
        "*This report was generated automatically by AgentOps Compliance Auditor v0.28.*",
        "*It does not constitute legal advice. Consult qualified legal counsel for regulatory compliance.*",
    ])

    return "\n".join(lines)
