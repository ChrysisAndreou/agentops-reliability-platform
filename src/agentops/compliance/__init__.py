"""
EU AI Act Compliance Framework — regulatory compliance for AI agent systems.

Implements the EU Artificial Intelligence Act (Regulation 2024/1689) risk-based
compliance framework for AI agent systems deployed in the European market.
Provides automated risk classification, conformity assessment, technical
documentation generation, and compliance auditing with structured reporting.

This module addresses a critical gap: as the EU AI Act enters enforcement
(August 2026 for high-risk systems), organizations deploying AI agents in
Europe need systematic compliance tooling. No other open-source agent
evaluation platform provides EU AI Act compliance assessment.

Regulatory basis: Regulation (EU) 2024/1689 (EU AI Act), Articles 5, 6, 13,
14, 52; Annexes III, IV, V.

Modules:
    classifier: Risk classification engine — maps agent system characteristics
                to EU AI Act risk tiers (unacceptable/high/limited/minimal) using
                Article 6 criteria and Annex III categories.
    requirements: Conformity assessment requirements by risk level — data
                  governance, transparency, human oversight, accuracy, robustness,
                  record-keeping, and CE marking requirements per Article 13-14.
    documentation: Automated technical documentation generation per Annex IV
                   requirements — system description, intended purpose, training
                   data, performance metrics, risk assessment, and human oversight
                   measures.
    auditor: Compliance auditing engine — systematic evaluation of agent systems
             against EU AI Act requirements with pass/fail gating, non-conformity
             tracking, and structured compliance reports with remediation guidance.
    eval: Compliance evaluation benchmarks — scenario-based testing across
          risk tiers, sector-specific compliance checks, and integration with
          existing AgentOps evaluation harness.

Example usage:
    >>> from agentops.compliance import ComplianceAuditor, RiskClassifier
    >>> classifier = RiskClassifier()
    >>> result = classifier.classify(
    ...     domain="healthcare",
    ...     has_human_oversight=True,
    ...     processes_personal_data=True,
    ... )
    >>> print(f"Risk tier: {result.tier}, Score: {result.score}")
    >>> auditor = ComplianceAuditor()
    >>> report = auditor.audit(agent_spec, risk_tier=result.tier)
    >>> print(report.summary())
"""

from agentops.compliance.classifier import (
    AnnexIIICategory,
    RiskFactor,
    RiskTier,
    RiskClassification,
    RiskClassifier,
    classify_agent_system,
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
    DocSection,
    DocRequirement,
    DocStatus,
    generate_technical_documentation,
    validate_documentation,
)

from agentops.compliance.auditor import (
    Finding,
    FindingSeverity,
    ComplianceStatus,
    ComplianceReport,
    ComplianceAuditor,
    audit_agent_system,
    format_compliance_report,
)

from agentops.compliance.eval import (
    ComplianceScenario,
    ComplianceBenchmark,
    ComplianceEvalResult,
    ComplianceEvaluator,
    evaluate_compliance,
    format_compliance_eval_report,
)

__all__ = [
    # Classifier
    "AnnexIIICategory",
    "RiskFactor",
    "RiskTier",
    "RiskClassification",
    "RiskClassifier",
    "classify_agent_system",
    # Requirements
    "ConformityRequirement",
    "RequirementCategory",
    "RequirementStatus",
    "RequirementsProfile",
    "get_requirements_for_tier",
    "build_requirements_checklist",
    # Documentation
    "TechnicalDocumentation",
    "DocSection",
    "DocRequirement",
    "DocStatus",
    "generate_technical_documentation",
    "validate_documentation",
    # Auditor
    "Finding",
    "FindingSeverity",
    "ComplianceStatus",
    "ComplianceReport",
    "ComplianceAuditor",
    "audit_agent_system",
    "format_compliance_report",
    # Evaluation
    "ComplianceScenario",
    "ComplianceBenchmark",
    "ComplianceEvalResult",
    "ComplianceEvaluator",
    "evaluate_compliance",
    "format_compliance_eval_report",
]
