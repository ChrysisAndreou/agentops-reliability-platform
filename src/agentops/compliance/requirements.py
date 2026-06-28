"""
Conformity Assessment Requirements — risk-level-specific obligations.

Implements Articles 9-15 and Annex IV of the EU AI Act, providing structured
requirements checklists for each risk tier. Each requirement maps to specific
articles and includes verification criteria for automated compliance checking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

from agentops.compliance.classifier import RiskTier


class RequirementCategory(str, Enum):
    """Categories of EU AI Act conformity requirements."""
    RISK_MANAGEMENT = "risk_management"
    DATA_GOVERNANCE = "data_governance"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    RECORD_KEEPING = "record_keeping"
    TRANSPARENCY = "transparency"
    HUMAN_OVERSIGHT = "human_oversight"
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    CYBERSECURITY = "cybersecurity"
    QUALITY_MANAGEMENT = "quality_management"
    CE_MARKING = "ce_marking"
    POST_MARKET = "post_market"
    REGISTRATION = "registration"


class RequirementStatus(str, Enum):
    """Status of a conformity requirement check."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


@dataclass
class ConformityRequirement:
    """A single EU AI Act conformity requirement."""
    id: str
    category: RequirementCategory
    article: str
    description: str
    verification_criteria: str
    risk_tiers: List[RiskTier] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    status: RequirementStatus = RequirementStatus.NOT_CHECKED
    notes: str = ""


@dataclass
class RequirementsProfile:
    """Complete set of requirements for a given risk tier."""
    risk_tier: RiskTier
    requirements: List[ConformityRequirement] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.requirements)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.requirements if r.status == RequirementStatus.PASS)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.requirements if r.status == RequirementStatus.FAIL)
    
    @property
    def partial(self) -> int:
        return sum(1 for r in self.requirements if r.status == RequirementStatus.PARTIAL)
    
    @property
    def compliance_ratio(self) -> float:
        applicable = self.passed + self.failed + self.partial
        if applicable == 0:
            return 1.0
        # Partial counts as 0.5
        return (self.passed + 0.5 * self.partial) / applicable
    
    def by_category(self) -> Dict[RequirementCategory, List[ConformityRequirement]]:
        result: Dict[RequirementCategory, List[ConformityRequirement]] = {}
        for req in self.requirements:
            result.setdefault(req.category, []).append(req)
        return result


# ---------------------------------------------------------------------------
# Requirements definitions per risk tier
# ---------------------------------------------------------------------------

_HIGH_RISK_REQUIREMENTS = [
    # Risk management system (Article 9)
    ConformityRequirement(
        id="HR-RM-01",
        category=RequirementCategory.RISK_MANAGEMENT,
        article="Article 9",
        description="Establish, implement, document, and maintain a risk management system",
        verification_criteria="Risk management system must cover the entire lifecycle, "
                             "identify foreseeable risks, estimate and evaluate risks, "
                             "and adopt risk management measures",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["risk_assessment_document", "risk_mitigation_plan"],
    ),
    ConformityRequirement(
        id="HR-RM-02",
        category=RequirementCategory.RISK_MANAGEMENT,
        article="Article 9(4)",
        description="Eliminate or reduce risks as far as possible through adequate design",
        verification_criteria="Design decisions documented with risk-reduction rationale; "
                             "residual risk acknowledged and justified",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["design_risk_rationale", "residual_risk_acknowledgment"],
    ),

    # Data governance (Article 10)
    ConformityRequirement(
        id="HR-DG-01",
        category=RequirementCategory.DATA_GOVERNANCE,
        article="Article 10(2)",
        description="Training, validation, and testing datasets shall be subject to "
                     "appropriate data governance and management practices",
        verification_criteria="Datasets must be relevant, representative, free from "
                             "errors, and complete. Provenance and characteristics documented.",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["data_governance_policy", "dataset_documentation", "bias_assessment"],
    ),
    ConformityRequirement(
        id="HR-DG-02",
        category=RequirementCategory.DATA_GOVERNANCE,
        article="Article 10(5)",
        description="Processing of special categories of personal data for bias "
                     "monitoring, detection, and correction",
        verification_criteria="If special category data is processed, GDPR safeguards "
                             "must apply, including purpose limitation and data minimization",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["dpia", "gdpr_compliance_check", "bias_correction_procedure"],
    ),

    # Technical documentation (Article 11)
    ConformityRequirement(
        id="HR-TD-01",
        category=RequirementCategory.TECHNICAL_DOCUMENTATION,
        article="Article 11",
        description="Draw up technical documentation before the system is placed on "
                     "the market or put into service",
        verification_criteria="Documentation must cover all Annex IV elements: system "
                             "description, intended purpose, design specifications, "
                             "training methodologies, performance metrics",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["technical_documentation", "annex_iv_checklist"],
    ),
    ConformityRequirement(
        id="HR-TD-02",
        category=RequirementCategory.TECHNICAL_DOCUMENTATION,
        article="Article 11(1)",
        description="Keep technical documentation up to date throughout the system's lifetime",
        verification_criteria="Version control with change history; documentation "
                             "updated within 30 days of any material change",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["documentation_version_history", "change_log"],
    ),

    # Record-keeping (Article 12)
    ConformityRequirement(
        id="HR-RK-01",
        category=RequirementCategory.RECORD_KEEPING,
        article="Article 12",
        description="Automatically record events (logs) while the system is operating",
        verification_criteria="Logging must cover: time of each use, reference database "
                             "used, input data, identification of natural persons involved, "
                             "and results of log verification",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["logging_system_description", "log_sample", "log_retention_policy"],
    ),
    ConformityRequirement(
        id="HR-RK-02",
        category=RequirementCategory.RECORD_KEEPING,
        article="Article 12(1)",
        description="Ensure traceability of the AI system's functioning throughout its lifetime",
        verification_criteria="Trace must cover all system decisions, intermediate states, "
                             "and model versioning; logs must be tamper-evident",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["trace_sample", "trace_retention_policy", "tamper_evidence_mechanism"],
    ),

    # Transparency (Article 13)
    ConformityRequirement(
        id="HR-TR-01",
        category=RequirementCategory.TRANSPARENCY,
        article="Article 13",
        description="Design and develop high-risk AI systems to ensure sufficiently "
                     "transparent operation",
        verification_criteria="Users must be able to interpret system output. Operating "
                             "instructions must include: identity of provider, capabilities "
                             "and limitations, foreseeable risks, human oversight measures, "
                             "and expected lifetime",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["user_instructions", "limitations_disclosure", "oversight_procedures"],
    ),

    # Human oversight (Article 14)
    ConformityRequirement(
        id="HR-HO-01",
        category=RequirementCategory.HUMAN_OVERSIGHT,
        article="Article 14",
        description="Design AI systems to enable effective human oversight",
        verification_criteria="Human overseers must be able to: fully understand the "
                             "system's capabilities/limitations, remain aware of automation "
                             "bias, correctly interpret output, decide not to use the system, "
                             "and intervene or override decisions",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["human_oversight_procedure", "override_mechanism", "training_materials"],
    ),
    ConformityRequirement(
        id="HR-HO-02",
        category=RequirementCategory.HUMAN_OVERSIGHT,
        article="Article 14(4)",
        description="Implement human-in-the-loop, human-on-the-loop, or human-in-command "
                     "measures as appropriate",
        verification_criteria="For agentic systems: human-in-the-loop for high-stakes "
                             "decisions, human-on-the-loop with override capability for "
                             "routine decisions, documented escalation procedures",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["oversight_architecture", "escalation_procedures", "override_log"],
    ),

    # Accuracy, robustness, cybersecurity (Article 15)
    ConformityRequirement(
        id="HR-AR-01",
        category=RequirementCategory.ACCURACY,
        article="Article 15(1)",
        description="Design AI systems to achieve an appropriate level of accuracy",
        verification_criteria="Accuracy metrics must be defined and measured against "
                             "declared performance levels; accuracy must be maintained "
                             "over the system's lifetime",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["accuracy_metrics", "performance_report", "accuracy_monitoring_plan"],
    ),
    ConformityRequirement(
        id="HR-AR-02",
        category=RequirementCategory.ROBUSTNESS,
        article="Article 15(3)",
        description="Design AI systems to be resilient to errors, faults, and "
                     "inconsistencies, including adversarial manipulation",
        verification_criteria="System must handle erroneous inputs gracefully, resist "
                             "adversarial attacks, and have defined fallback plans for "
                             "unexpected situations",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["robustness_test_results", "adversarial_test_results", "fallback_plan"],
    ),
    ConformityRequirement(
        id="HR-AR-03",
        category=RequirementCategory.CYBERSECURITY,
        article="Article 15(4)",
        description="Design AI systems to be resilient against unauthorized third-party "
                     "attempts to alter their use, outputs, or performance",
        verification_criteria="Security measures must address: model poisoning, data "
                             "poisoning, adversarial examples, and model extraction attacks",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["security_assessment", "penetration_test_results", "security_update_policy"],
    ),

    # Quality management system (Article 17)
    ConformityRequirement(
        id="HR-QM-01",
        category=RequirementCategory.QUALITY_MANAGEMENT,
        article="Article 17",
        description="Implement a quality management system for regulatory compliance",
        verification_criteria="QMS must cover: regulatory compliance strategy, design "
                             "control procedures, technical documentation management, "
                             "post-market monitoring, incident reporting, and communication "
                             "with authorities",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["qms_documentation", "compliance_strategy", "incident_report_procedure"],
    ),

    # CE marking (Article 48)
    ConformityRequirement(
        id="HR-CE-01",
        category=RequirementCategory.CE_MARKING,
        article="Article 48",
        description="Affix CE marking to indicate conformity with the Regulation",
        verification_criteria="CE marking must be affixed visibly, legibly, and indelibly "
                             "to the AI system or its packaging; EU Declaration of Conformity "
                             "must be drawn up",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["eu_declaration_of_conformity", "ce_marking_placement"],
    ),

    # Post-market monitoring (Article 61)
    ConformityRequirement(
        id="HR-PM-01",
        category=RequirementCategory.POST_MARKET,
        article="Article 61",
        description="Establish and document a post-market monitoring system",
        verification_criteria="Monitoring must collect and analyze data on system "
                             "performance throughout its lifetime, with incident "
                             "reporting within 72 hours for serious incidents",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["post_market_monitoring_plan", "incident_report_template"],
    ),

    # Registration (Article 51)
    ConformityRequirement(
        id="HR-RG-01",
        category=RequirementCategory.REGISTRATION,
        article="Article 51",
        description="Register the high-risk AI system in the EU database",
        verification_criteria="Registration must include: provider details, system "
                             "description, intended purpose, risk category, conformity "
                             "assessment body, and performance metrics",
        risk_tiers=[RiskTier.HIGH],
        evidence_required=["eu_database_registration", "registration_confirmation"],
    ),
]

_LIMITED_RISK_REQUIREMENTS = [
    ConformityRequirement(
        id="LR-TR-01",
        category=RequirementCategory.TRANSPARENCY,
        article="Article 52(1)",
        description="Inform users they are interacting with an AI system",
        verification_criteria="Clear disclosure at the start of interaction, in a way "
                             "that is easily noticeable and understandable",
        risk_tiers=[RiskTier.LIMITED],
        evidence_required=["ai_disclosure_notice", "user_interaction_log"],
    ),
    ConformityRequirement(
        id="LR-TR-02",
        category=RequirementCategory.TRANSPARENCY,
        article="Article 52(3)",
        description="Disclose when content has been artificially generated or manipulated",
        verification_criteria="Generated content must be marked as AI-generated; "
                             "metadata must include generation provenance",
        risk_tiers=[RiskTier.LIMITED],
        evidence_required=["content_disclosure_mechanism", "provenance_metadata_schema"],
    ),
]

_MINIMAL_RISK_REQUIREMENTS = [
    ConformityRequirement(
        id="MR-VL-01",
        category=RequirementCategory.TRANSPARENCY,
        article="Recital 28",
        description="Voluntary compliance encouraged — no mandatory obligations",
        verification_criteria="No mandatory requirements apply. Voluntary adoption "
                             "of codes of conduct is encouraged by recital 28.",
        risk_tiers=[RiskTier.MINIMAL],
        evidence_required=[],
    ),
]

# ---- Unacceptable risk systems should not be deployed ----
_UNACCEPTABLE_REQUIREMENTS = [
    ConformityRequirement(
        id="UN-PH-01",
        category=RequirementCategory.TRANSPARENCY,
        article="Article 5",
        description="PROHIBITED: This AI practice is prohibited under Article 5 "
                     "and must not be placed on the market, put into service, or used",
        verification_criteria="System must not be deployed. If already deployed, "
                             "must be withdrawn immediately.",
        risk_tiers=[RiskTier.UNACCEPTABLE],
        evidence_required=["withdrawal_confirmation"],
    ),
]


def get_requirements_for_tier(risk_tier: RiskTier) -> RequirementsProfile:
    """Return the conformity assessment requirements for a given risk tier.

    Args:
        risk_tier: The EU AI Act risk tier

    Returns:
        RequirementsProfile with all applicable requirements for that tier
    """
    tier_map = {
        RiskTier.HIGH: _HIGH_RISK_REQUIREMENTS,
        RiskTier.LIMITED: _LIMITED_RISK_REQUIREMENTS,
        RiskTier.MINIMAL: _MINIMAL_RISK_REQUIREMENTS,
        RiskTier.UNACCEPTABLE: _UNACCEPTABLE_REQUIREMENTS,
    }
    reqs = tier_map.get(risk_tier, [])
    return RequirementsProfile(risk_tier=risk_tier, requirements=list(reqs))


def build_requirements_checklist(
    risk_tier: RiskTier,
    evidence_available: Optional[Dict[str, bool]] = None,
) -> RequirementsProfile:
    """Build a requirements checklist with pre-filled evidence availability.

    Args:
        risk_tier: The EU AI Act risk tier
        evidence_available: Optional dict of {evidence_id: is_available}

    Returns:
        RequirementsProfile with initial status set based on evidence availability
    """
    profile = get_requirements_for_tier(risk_tier)
    if evidence_available is None:
        return profile

    for req in profile.requirements:
        all_available = all(
            evidence_available.get(ev, False)
            for ev in req.evidence_required
        )
        any_available = any(
            evidence_available.get(ev, False)
            for ev in req.evidence_required
        )
        if not req.evidence_required:
            req.status = RequirementStatus.NOT_CHECKED
        elif all_available:
            req.status = RequirementStatus.PASS
        elif any_available:
            req.status = RequirementStatus.PARTIAL
        else:
            req.status = RequirementStatus.FAIL

    return profile
