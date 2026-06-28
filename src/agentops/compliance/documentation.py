"""
Automated Technical Documentation — Annex IV compliance.

Generates structured technical documentation for AI agent systems as required
by Annex IV of the EU AI Act. Covers system description, intended purpose,
design specifications, training methodologies, performance metrics, risk
assessment, and human oversight measures.

The documentation engine produces structured, versioned, and auditable
technical documentation suitable for notified body review and regulatory
submission.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime


class DocSection(str, Enum):
    """Annex IV documentation sections."""
    GENERAL_DESCRIPTION = "general_description"
    INTENDED_PURPOSE = "intended_purpose"
    DESIGN_SPECIFICATION = "design_specification"
    DEVELOPMENT_METHODOLOGY = "development_methodology"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_ASSESSMENT = "risk_assessment"
    HUMAN_OVERSIGHT = "human_oversight"
    CHANGES_AND_VERSIONING = "changes_and_versioning"


class DocRequirement(str, Enum):
    """Specific documentation requirements per Annex IV."""
    PROVIDER_IDENTITY = "provider_identity"
    SYSTEM_NAME_VERSION = "system_name_version"
    INTENDED_PURPOSE_STATEMENT = "intended_purpose_statement"
    INTENDED_USERS = "intended_users"
    GEOGRAPHIC_SCOPE = "geographic_scope"
    ARCHITECTURE_DESCRIPTION = "architecture_description"
    INPUT_OUTPUT_SPEC = "input_output_spec"
    SYSTEM_BOUNDARIES = "system_boundaries"
    LOGIC_AND_ALGORITHMS = "logic_and_algorithms"
    TRAINING_METHODOLOGY = "training_methodology"
    TRAINING_DATA_DESCRIPTION = "training_data_description"
    VALIDATION_PROCEDURE = "validation_procedure"
    ACCURACY_METRICS = "accuracy_metrics"
    ROBUSTNESS_METRICS = "robustness_metrics"
    BIAS_ASSESSMENT = "bias_assessment"
    RISK_IDENTIFICATION = "risk_identification"
    RISK_MITIGATION = "risk_mitigation"
    RESIDUAL_RISK = "residual_risk"
    OVERSIGHT_MEASURES = "oversight_measures"
    OVERRIDE_MECHANISMS = "override_mechanisms"
    ESCALATION_PROCEDURES = "escalation_procedures"
    CHANGE_LOG = "change_log"
    VERSION_HISTORY = "version_history"


class DocStatus(str, Enum):
    """Status of a documentation requirement."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class TechnicalDocumentation:
    """Annex IV technical documentation for an AI agent system."""
    # General description (Annex IV, Section 1)
    provider_name: str = ""
    provider_address: str = ""
    provider_contact: str = ""
    system_name: str = ""
    system_version: str = ""
    system_description: str = ""

    # Intended purpose (Annex IV, Section 1(b))
    intended_purpose: str = ""
    intended_users: List[str] = field(default_factory=list)
    geographic_scope: List[str] = field(default_factory=list)
    prohibited_uses: List[str] = field(default_factory=list)

    # Design specification (Annex IV, Section 1(c))
    architecture_description: str = ""
    input_specification: str = ""
    output_specification: str = ""
    system_boundaries: str = ""
    logic_description: str = ""
    dependencies: List[str] = field(default_factory=list)

    # Development methodology (Annex IV, Section 1(d))
    training_methodology: str = ""
    training_data_description: str = ""
    training_data_provenance: str = ""
    data_preprocessing: str = ""
    validation_procedure: str = ""
    testing_methodology: str = ""

    # Performance metrics (Annex IV, Section 1(e))
    accuracy_metrics: str = ""
    robustness_metrics: str = ""
    bias_assessment: str = ""
    fairness_metrics: str = ""
    known_limitations: List[str] = field(default_factory=list)

    # Risk assessment (Annex IV, Section 1(f))
    risk_identification: str = ""
    risk_mitigation: str = ""
    residual_risk: str = ""

    # Human oversight (Annex IV, Section 1(g))
    oversight_measures: str = ""
    override_mechanisms: str = ""
    escalation_procedures: str = ""
    training_requirements: str = ""

    # Metadata
    generated_at: str = ""
    document_version: str = "1.0"
    last_updated: str = ""
    status_overrides: Dict[DocRequirement, DocStatus] = field(default_factory=dict)

    def completeness_report(self) -> Dict[DocSection, Dict[str, int]]:
        """Generate a completeness report for all Annex IV sections.

        Returns:
            Dict mapping each DocSection to {total, complete, partial, missing, score}
        """
        section_map = {
            DocSection.GENERAL_DESCRIPTION: [
                DocRequirement.PROVIDER_IDENTITY,
                DocRequirement.SYSTEM_NAME_VERSION,
                DocRequirement.INTENDED_PURPOSE_STATEMENT,
            ],
            DocSection.INTENDED_PURPOSE: [
                DocRequirement.INTENDED_PURPOSE_STATEMENT,
                DocRequirement.INTENDED_USERS,
                DocRequirement.GEOGRAPHIC_SCOPE,
            ],
            DocSection.DESIGN_SPECIFICATION: [
                DocRequirement.ARCHITECTURE_DESCRIPTION,
                DocRequirement.INPUT_OUTPUT_SPEC,
                DocRequirement.SYSTEM_BOUNDARIES,
                DocRequirement.LOGIC_AND_ALGORITHMS,
            ],
            DocSection.DEVELOPMENT_METHODOLOGY: [
                DocRequirement.TRAINING_METHODOLOGY,
                DocRequirement.TRAINING_DATA_DESCRIPTION,
                DocRequirement.VALIDATION_PROCEDURE,
            ],
            DocSection.PERFORMANCE_METRICS: [
                DocRequirement.ACCURACY_METRICS,
                DocRequirement.ROBUSTNESS_METRICS,
                DocRequirement.BIAS_ASSESSMENT,
            ],
            DocSection.RISK_ASSESSMENT: [
                DocRequirement.RISK_IDENTIFICATION,
                DocRequirement.RISK_MITIGATION,
                DocRequirement.RESIDUAL_RISK,
            ],
            DocSection.HUMAN_OVERSIGHT: [
                DocRequirement.OVERSIGHT_MEASURES,
                DocRequirement.OVERRIDE_MECHANISMS,
                DocRequirement.ESCALATION_PROCEDURES,
            ],
            DocSection.CHANGES_AND_VERSIONING: [
                DocRequirement.CHANGE_LOG,
                DocRequirement.VERSION_HISTORY,
            ],
        }

        report = {}
        for section, reqs in section_map.items():
            statuses = [self._check_requirement(r) for r in reqs]
            complete = sum(1 for s in statuses if s == DocStatus.COMPLETE)
            partial = sum(1 for s in statuses if s == DocStatus.PARTIAL)
            missing = sum(1 for s in statuses if s == DocStatus.MISSING)
            na = sum(1 for s in statuses if s == DocStatus.NOT_APPLICABLE)
            applicable = len(reqs) - na
            score = (complete + 0.5 * partial) / applicable if applicable > 0 else 1.0
            report[section] = {
                "total": len(reqs),
                "applicable": applicable,
                "complete": complete,
                "partial": partial,
                "missing": missing,
                "score": score,
            }
        return report

    def overall_completeness(self) -> float:
        """Return the overall documentation completeness score (0.0-1.0)."""
        report = self.completeness_report()
        if not report:
            return 0.0
        scores = [r["score"] for r in report.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def is_complete(self, threshold: float = 0.95) -> bool:
        """Check if documentation meets the completeness threshold."""
        return self.overall_completeness() >= threshold

    def _check_requirement(self, req: DocRequirement) -> DocStatus:
        """Check whether a specific documentation requirement is satisfied."""
        if req in self.status_overrides:
            return self.status_overrides[req]

        field_map = {
            DocRequirement.PROVIDER_IDENTITY: bool(self.provider_name and self.provider_contact),
            DocRequirement.SYSTEM_NAME_VERSION: bool(self.system_name and self.system_version),
            DocRequirement.INTENDED_PURPOSE_STATEMENT: bool(self.intended_purpose),
            DocRequirement.INTENDED_USERS: bool(self.intended_users),
            DocRequirement.GEOGRAPHIC_SCOPE: bool(self.geographic_scope),
            DocRequirement.ARCHITECTURE_DESCRIPTION: bool(self.architecture_description),
            DocRequirement.INPUT_OUTPUT_SPEC: bool(self.input_specification and self.output_specification),
            DocRequirement.SYSTEM_BOUNDARIES: bool(self.system_boundaries),
            DocRequirement.LOGIC_AND_ALGORITHMS: bool(self.logic_description),
            DocRequirement.TRAINING_METHODOLOGY: bool(self.training_methodology),
            DocRequirement.TRAINING_DATA_DESCRIPTION: bool(self.training_data_description),
            DocRequirement.VALIDATION_PROCEDURE: bool(self.validation_procedure),
            DocRequirement.ACCURACY_METRICS: bool(self.accuracy_metrics),
            DocRequirement.ROBUSTNESS_METRICS: bool(self.robustness_metrics),
            DocRequirement.BIAS_ASSESSMENT: bool(self.bias_assessment),
            DocRequirement.RISK_IDENTIFICATION: bool(self.risk_identification),
            DocRequirement.RISK_MITIGATION: bool(self.risk_mitigation),
            DocRequirement.RESIDUAL_RISK: bool(self.residual_risk),
            DocRequirement.OVERSIGHT_MEASURES: bool(self.oversight_measures),
            DocRequirement.OVERRIDE_MECHANISMS: bool(self.override_mechanisms),
            DocRequirement.ESCALATION_PROCEDURES: bool(self.escalation_procedures),
            DocRequirement.CHANGE_LOG: bool(self.last_updated),
            DocRequirement.VERSION_HISTORY: bool(self.document_version),
        }

        filled = field_map.get(req, False)
        return DocStatus.COMPLETE if filled else DocStatus.MISSING

    def format_markdown(self) -> str:
        """Render technical documentation as Annex IV-compliant Markdown."""
        completeness = self.completeness_report()
        overall = self.overall_completeness()

        lines = [
            f"# Technical Documentation — {self.system_name or 'AI Agent System'}",
            f"",
            f"**Document Version:** {self.document_version}",
            f"**Generated:** {self.generated_at or datetime.now().isoformat()}",
            f"**Last Updated:** {self.last_updated or 'N/A'}",
            f"**Overall Completeness:** {overall:.1%}",
            f"",
            f"## 1. General Description",
            f"",
            f"- **Provider:** {self.provider_name or '[NOT PROVIDED]'}",
            f"- **Contact:** {self.provider_contact or '[NOT PROVIDED]'}",
            f"- **System:** {self.system_name or '[NOT PROVIDED]'} v{self.system_version or 'N/A'}",
            f"",
            f"{self.system_description or '[No system description provided]'}",
            f"",
            f"## 2. Intended Purpose",
            f"",
            f"{self.intended_purpose or '[No intended purpose statement provided]'}",
            f"",
            f"### Intended Users",
            f"",
        ]
        if self.intended_users:
            for user in self.intended_users:
                lines.append(f"- {user}")
        else:
            lines.append("- [Not specified]")

        lines.extend([
            f"",
            f"### Geographic Scope",
            f"",
        ])
        if self.geographic_scope:
            for geo in self.geographic_scope:
                lines.append(f"- {geo}")
        else:
            lines.append("- [Not specified]")

        if self.prohibited_uses:
            lines.extend([f"", f"### Prohibited Uses", f""])
            for use in self.prohibited_uses:
                lines.append(f"- {use}")

        lines.extend([
            f"",
            f"## 3. Design Specification",
            f"",
            f"### Architecture",
            f"",
            f"{self.architecture_description or '[No architecture description provided]'}",
            f"",
            f"### Input Specification",
            f"",
            f"{self.input_specification or '[Not provided]'}",
            f"",
            f"### Output Specification",
            f"",
            f"{self.output_specification or '[Not provided]'}",
            f"",
            f"### System Boundaries",
            f"",
            f"{self.system_boundaries or '[Not defined]'}",
            f"",
            f"### Logic & Algorithms",
            f"",
            f"{self.logic_description or '[Not provided]'}",
            f"",
            f"## 4. Development Methodology",
            f"",
            f"### Training Methodology",
            f"",
            f"{self.training_methodology or '[Not provided]'}",
            f"",
            f"### Training Data",
            f"",
            f"{self.training_data_description or '[Not provided]'}",
            f"",
            f"**Data Provenance:** {self.training_data_provenance or '[Not provided]'}",
            f"",
            f"### Validation Procedure",
            f"",
            f"{self.validation_procedure or '[Not provided]'}",
            f"",
            f"## 5. Performance Metrics",
            f"",
            f"### Accuracy",
            f"",
            f"{self.accuracy_metrics or '[Not provided]'}",
            f"",
            f"### Robustness",
            f"",
            f"{self.robustness_metrics or '[Not provided]'}",
            f"",
            f"### Bias Assessment",
            f"",
            f"{self.bias_assessment or '[Not provided]'}",
            f"",
            f"### Known Limitations",
            f"",
        ])
        if self.known_limitations:
            for lim in self.known_limitations:
                lines.append(f"- {lim}")
        else:
            lines.append("- [None documented]")

        lines.extend([
            f"",
            f"## 6. Risk Assessment",
            f"",
            f"### Risk Identification",
            f"",
            f"{self.risk_identification or '[Not performed]'}",
            f"",
            f"### Risk Mitigation",
            f"",
            f"{self.risk_mitigation or '[Not documented]'}",
            f"",
            f"### Residual Risk",
            f"",
            f"{self.residual_risk or '[Not assessed]'}",
            f"",
            f"## 7. Human Oversight",
            f"",
            f"### Oversight Measures",
            f"",
            f"{self.oversight_measures or '[Not documented]'}",
            f"",
            f"### Override Mechanisms",
            f"",
            f"{self.override_mechanisms or '[Not documented]'}",
            f"",
            f"### Escalation Procedures",
            f"",
            f"{self.escalation_procedures or '[Not documented]'}",
            f"",
            f"## 8. Changes & Versioning",
            f"",
            f"- **Current Version:** {self.document_version}",
            f"- **Last Updated:** {self.last_updated or 'N/A'}",
            f"",
            f"---",
            f"",
            f"### Documentation Completeness Summary",
            f"",
            f"| Section | Score | Status |",
            f"|---------|-------|--------|",
        ])

        for section, report in completeness.items():
            status = "✓" if report["score"] >= 0.95 else ("⚠" if report["score"] >= 0.5 else "✗")
            label = section.value.replace("_", " ").title()
            lines.append(f"| {label} | {report['score']:.1%} | {status} |")

        lines.extend([
            f"",
            f"**Overall:** {overall:.1%}",
        ])

        return "\n".join(lines)


def generate_technical_documentation(**kwargs) -> TechnicalDocumentation:
    """Generate Annex IV-compliant technical documentation.

    Accepts all fields defined in TechnicalDocumentation as keyword arguments.
    Fields not provided default to empty strings.

    Returns:
        TechnicalDocumentation with generated_at set to now
    """
    now = datetime.now().isoformat()
    doc = TechnicalDocumentation(
        provider_name=kwargs.get("provider_name", ""),
        provider_address=kwargs.get("provider_address", ""),
        provider_contact=kwargs.get("provider_contact", ""),
        system_name=kwargs.get("system_name", ""),
        system_version=kwargs.get("system_version", ""),
        system_description=kwargs.get("system_description", ""),
        intended_purpose=kwargs.get("intended_purpose", ""),
        intended_users=kwargs.get("intended_users", []),
        geographic_scope=kwargs.get("geographic_scope", []),
        prohibited_uses=kwargs.get("prohibited_uses", []),
        architecture_description=kwargs.get("architecture_description", ""),
        input_specification=kwargs.get("input_specification", ""),
        output_specification=kwargs.get("output_specification", ""),
        system_boundaries=kwargs.get("system_boundaries", ""),
        logic_description=kwargs.get("logic_description", ""),
        dependencies=kwargs.get("dependencies", []),
        training_methodology=kwargs.get("training_methodology", ""),
        training_data_description=kwargs.get("training_data_description", ""),
        training_data_provenance=kwargs.get("training_data_provenance", ""),
        data_preprocessing=kwargs.get("data_preprocessing", ""),
        validation_procedure=kwargs.get("validation_procedure", ""),
        testing_methodology=kwargs.get("testing_methodology", ""),
        accuracy_metrics=kwargs.get("accuracy_metrics", ""),
        robustness_metrics=kwargs.get("robustness_metrics", ""),
        bias_assessment=kwargs.get("bias_assessment", ""),
        fairness_metrics=kwargs.get("fairness_metrics", ""),
        known_limitations=kwargs.get("known_limitations", []),
        risk_identification=kwargs.get("risk_identification", ""),
        risk_mitigation=kwargs.get("risk_mitigation", ""),
        residual_risk=kwargs.get("residual_risk", ""),
        oversight_measures=kwargs.get("oversight_measures", ""),
        override_mechanisms=kwargs.get("override_mechanisms", ""),
        escalation_procedures=kwargs.get("escalation_procedures", ""),
        training_requirements=kwargs.get("training_requirements", ""),
        generated_at=now,
        last_updated=now,
    )
    return doc


def validate_documentation(doc: TechnicalDocumentation) -> Dict[str, List[str]]:
    """Validate technical documentation for Annex IV compliance.

    Returns:
        Dict with 'errors' (blocking) and 'warnings' (advisory) lists
    """
    errors = []
    warnings = []

    # Critical fields for Annex IV
    critical = [
        ("provider_name", "Provider identity (Article 11, Annex IV 1(a))"),
        ("system_name", "System name and version (Annex IV 1(a))"),
        ("intended_purpose", "Intended purpose statement (Annex IV 1(b))"),
        ("architecture_description", "Architecture description (Annex IV 1(c))"),
    ]
    for field, label in critical:
        if not getattr(doc, field):
            errors.append(f"Missing: {label}")

    # Warning-level fields
    advisory = [
        ("training_methodology", "Training methodology (Annex IV 1(d))"),
        ("validation_procedure", "Validation procedure (Annex IV 1(d))"),
        ("accuracy_metrics", "Accuracy metrics (Annex IV 1(e))"),
        ("risk_identification", "Risk assessment (Annex IV 1(f))"),
        ("oversight_measures", "Human oversight measures (Annex IV 1(g))"),
    ]
    for field, label in advisory:
        if not getattr(doc, field):
            warnings.append(f"Recommended: {label}")

    return {"errors": errors, "warnings": warnings}
