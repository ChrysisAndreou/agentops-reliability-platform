"""
EU AI Act Risk Classification Engine.

Implements Article 6 and Annex III of Regulation (EU) 2024/1689 for classifying
AI agent systems into risk tiers: unacceptable, high, limited, and minimal.

The classification is rule-based and deterministic, producing a structured
RiskClassification with the rationale, applicable articles, and confidence level.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class RiskTier(str, Enum):
    """EU AI Act risk tiers per Articles 5-6 and recitals 26-29."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


class AnnexIIICategory(str, Enum):
    """High-risk categories from Annex III of the EU AI Act."""
    BIOMETRICS = "biometrics"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION = "education"
    EMPLOYMENT = "employment"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION = "migration"
    DEMOCRATIC_PROCESSES = "democratic_processes"
    HEALTHCARE = "healthcare"  # Subset of essential services, significant enough to call out


class RiskFactor(str, Enum):
    """Factors that contribute to risk classification."""
    SAFETY_COMPONENT = "safety_component"
    PROCESSES_PERSONAL_DATA = "processes_personal_data"
    PROCESSES_SPECIAL_CATEGORY_DATA = "processes_special_category_data"
    MAKES_AUTONOMOUS_DECISIONS = "makes_autonomous_decisions"
    AFFECTS_LEGAL_RIGHTS = "affects_legal_rights"
    AFFECTS_PHYSICAL_SAFETY = "affects_physical_safety"
    PROFILING = "profiling"
    MANIPULATION = "manipulation"
    SOCIAL_SCORING = "social_scoring"
    VULNERABLE_POPULATION = "vulnerable_population"
    CRITICAL_SYSTEM = "critical_system"
    HAS_HUMAN_OVERSIGHT = "has_human_oversight"


@dataclass
class RiskClassification:
    """Result of AI Act risk classification."""
    tier: RiskTier
    score: float  # 0.0 (minimal) to 1.0 (unacceptable)
    factors: List[RiskFactor] = field(default_factory=list)
    applicable_articles: List[str] = field(default_factory=list)
    annex_iii_categories: List[AnnexIIICategory] = field(default_factory=list)
    rationale: str = ""
    requires_conformity_assessment: bool = False
    requires_ce_marking: bool = False
    requires_notified_body: bool = False
    requires_transparency: bool = False
    prohibition_risk: bool = False


class RiskClassifier:
    """
    Classifies AI agent systems according to the EU AI Act risk framework.

    The classifier evaluates an agent system's characteristics — domain, data
    processing patterns, autonomy level, oversight mechanisms — and maps them
    to the appropriate risk tier with supporting rationale.

    Classification rules are derived from the Regulation's Articles 5
    (prohibited practices), Article 6 (high-risk classification rules), and
    Annex III (high-risk AI systems).

    Usage:
        >>> classifier = RiskClassifier()
        >>> result = classifier.classify(
        ...     domain="healthcare",
        ...     description="Clinical decision support agent",
        ...     makes_autonomous_decisions=True,
        ...     processes_personal_data=True,
        ...     has_human_oversight=True,
        ... )
        >>> print(result.tier)
        RiskTier.HIGH
    """

    # Domain → Annex III category mapping
    DOMAIN_TO_ANNEX_III = {
        "healthcare": [AnnexIIICategory.HEALTHCARE, AnnexIIICategory.ESSENTIAL_SERVICES],
        "medical": [AnnexIIICategory.HEALTHCARE, AnnexIIICategory.ESSENTIAL_SERVICES],
        "finance": [AnnexIIICategory.ESSENTIAL_SERVICES],
        "banking": [AnnexIIICategory.ESSENTIAL_SERVICES],
        "insurance": [AnnexIIICategory.ESSENTIAL_SERVICES],
        "education": [AnnexIIICategory.EDUCATION],
        "edtech": [AnnexIIICategory.EDUCATION],
        "hr": [AnnexIIICategory.EMPLOYMENT],
        "recruitment": [AnnexIIICategory.EMPLOYMENT],
        "employment": [AnnexIIICategory.EMPLOYMENT],
        "energy": [AnnexIIICategory.CRITICAL_INFRASTRUCTURE],
        "transport": [AnnexIIICategory.CRITICAL_INFRASTRUCTURE],
        "water": [AnnexIIICategory.CRITICAL_INFRASTRUCTURE],
        "telecom": [AnnexIIICategory.CRITICAL_INFRASTRUCTURE],
        "law_enforcement": [AnnexIIICategory.LAW_ENFORCEMENT],
        "policing": [AnnexIIICategory.LAW_ENFORCEMENT],
        "migration": [AnnexIIICategory.MIGRATION],
        "border_control": [AnnexIIICategory.MIGRATION],
        "elections": [AnnexIIICategory.DEMOCRATIC_PROCESSES],
        "biometrics": [AnnexIIICategory.BIOMETRICS],
    }

    def classify(
        self,
        domain: str = "",
        description: str = "",
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
    ) -> RiskClassification:
        """
        Classify an AI agent system according to EU AI Act risk tiers.

        Args:
            domain: Application domain (healthcare, finance, education, etc.)
            description: Free-text description of the system
            makes_autonomous_decisions: Whether the agent makes decisions
                                        without human review
            processes_personal_data: Whether the agent processes personal data
                                     under GDPR
            processes_special_category_data: Whether the agent processes special
                                             category data (health, biometrics, etc.)
            affects_legal_rights: Whether decisions affect legal rights/obligations
            affects_physical_safety: Whether the system can cause physical harm
            involves_profiling: Whether the system profiles individuals
            involves_manipulation: Whether the system manipulates user behavior
            involves_social_scoring: Whether the system performs social scoring
            targets_vulnerable_population: Whether the system targets vulnerable
                                          populations (children, elderly, disabled)
            is_safety_component: Whether the system is a safety component of a
                                 product subject to EU harmonization legislation
            has_human_oversight: Whether human oversight mechanisms exist
            is_critical_system: Whether the system is part of critical infrastructure

        Returns:
            RiskClassification with tier, score, rationale, and requirements
        """
        factors: List[RiskFactor] = []
        annex_categories: List[AnnexIIICategory] = []
        score = 0.0

        # --- Prohibited practices (Article 5) ---
        if involves_social_scoring:
            factors.append(RiskFactor.SOCIAL_SCORING)
            score += 0.35

        if involves_manipulation and targets_vulnerable_population:
            factors.append(RiskFactor.MANIPULATION)
            factors.append(RiskFactor.VULNERABLE_POPULATION)
            score += 0.30

        # --- High-risk classification (Article 6 + Annex III) ---
        domain_cats = self.DOMAIN_TO_ANNEX_III.get(domain.lower(), [])
        if domain_cats:
            annex_categories.extend(domain_cats)
            score += 0.15 * len(set(domain_cats))  # Cap per unique category

        if is_safety_component:
            factors.append(RiskFactor.SAFETY_COMPONENT)
            score += 0.15

        if is_critical_system:
            factors.append(RiskFactor.CRITICAL_SYSTEM)
            score += 0.10

        if processes_special_category_data:
            factors.append(RiskFactor.PROCESSES_SPECIAL_CATEGORY_DATA)
            score += 0.12

        if processes_personal_data:
            factors.append(RiskFactor.PROCESSES_PERSONAL_DATA)
            score += 0.08

        # --- Additional risk factors ---
        if makes_autonomous_decisions:
            factors.append(RiskFactor.MAKES_AUTONOMOUS_DECISIONS)
            score += 0.10

        if affects_legal_rights:
            factors.append(RiskFactor.AFFECTS_LEGAL_RIGHTS)
            score += 0.10

        if affects_physical_safety:
            factors.append(RiskFactor.AFFECTS_PHYSICAL_SAFETY)
            score += 0.12

        if involves_profiling:
            factors.append(RiskFactor.PROFILING)
            score += 0.08

        if targets_vulnerable_population:
            factors.append(RiskFactor.VULNERABLE_POPULATION)
            score += 0.08

        # --- Mitigating factors ---
        if has_human_oversight:
            factors.append(RiskFactor.HAS_HUMAN_OVERSIGHT)
            score -= 0.05

        # --- Deterministic classification with hysteresis ---
        tier, articles, rationale_parts = self._determine_tier(
            score=score,
            involves_social_scoring=involves_social_scoring,
            involves_manipulation=involves_manipulation,
            targets_vulnerable_population=targets_vulnerable_population,
            domain_cats=domain_cats,
            is_safety_component=is_safety_component,
            is_critical_system=is_critical_system,
            makes_autonomous_decisions=makes_autonomous_decisions,
            processes_personal_data=processes_personal_data,
            domain=domain,
            description=description,
            factors=factors,
        )

        return RiskClassification(
            tier=tier,
            score=min(score, 1.0),
            factors=factors,
            applicable_articles=articles,
            annex_iii_categories=annex_categories,
            rationale="; ".join(rationale_parts),
            requires_conformity_assessment=tier in (RiskTier.HIGH, RiskTier.UNACCEPTABLE),
            requires_ce_marking=tier == RiskTier.HIGH,
            requires_notified_body=(
                tier == RiskTier.HIGH
                and not has_human_oversight
            ),
            requires_transparency=tier in (RiskTier.HIGH, RiskTier.LIMITED),
            prohibition_risk=(tier == RiskTier.UNACCEPTABLE),
        )

    def _determine_tier(
        self,
        score: float,
        involves_social_scoring: bool,
        involves_manipulation: bool,
        targets_vulnerable_population: bool,
        domain_cats: List[AnnexIIICategory],
        is_safety_component: bool,
        is_critical_system: bool,
        makes_autonomous_decisions: bool,
        processes_personal_data: bool,
        domain: str,
        description: str,
        factors: List[RiskFactor],
    ) -> tuple[RiskTier, List[str], List[str]]:
        """Deterministic tier assignment with hard rules from the Regulation."""

        articles = []
        rationale = []

        # Hard rules — these override score-based classification
        if involves_social_scoring:
            articles.append("Article 5(1)(c)")
            rationale.append("Social scoring by public authorities is prohibited under Article 5")
            return RiskTier.UNACCEPTABLE, articles, rationale

        if involves_manipulation and targets_vulnerable_population:
            articles.append("Article 5(1)(a)")
            rationale.append(
                "Manipulative AI targeting vulnerable populations is prohibited under Article 5"
            )
            return RiskTier.UNACCEPTABLE, articles, rationale

        # Safety component in regulated product → always high risk
        if is_safety_component:
            articles.append("Article 6(1)")
            rationale.append(
                "AI system is a safety component of a product subject to EU "
                "harmonization legislation — classified as high risk per Article 6(1)"
            )
            return RiskTier.HIGH, articles, rationale

        # Annex III domain + autonomous decisions → high risk
        if domain_cats and makes_autonomous_decisions:
            articles.append("Article 6(2)")
            articles.append(f"Annex III — {', '.join(c.value for c in domain_cats)}")
            rationale.append(
                f"Agent operates in an Annex III domain ({domain}) with autonomous "
                f"decision-making — high risk per Article 6(2)"
            )
            return RiskTier.HIGH, articles, rationale

        # Score-based classification with hysteresis
        if score >= 0.40:
            articles.append("Article 6(2)")
            if domain_cats:
                articles.append(f"Annex III — {', '.join(c.value for c in domain_cats)}")
            rationale.append(
                f"Composite risk score {score:.2f} exceeds high-risk threshold "
                f"(>= 0.40) — classified as high risk due to cumulative risk factors"
            )
            return RiskTier.HIGH, articles, rationale

        if score >= 0.08:
            articles.append("Article 52")
            rationale.append(
                f"Composite risk score {score:.2f} indicates limited risk — "
                f"transparency obligations apply per Article 52"
            )
            return RiskTier.LIMITED, articles, rationale

        # Default: minimal risk
        rationale.append(
            f"Composite risk score {score:.2f} indicates minimal risk — "
            f"no specific obligations under the EU AI Act"
        )
        return RiskTier.MINIMAL, articles, rationale


def classify_agent_system(**kwargs) -> RiskClassification:
    """Convenience function for one-shot risk classification.

    Args:
        **kwargs: All parameters accepted by RiskClassifier.classify()

    Returns:
        RiskClassification result
    """
    return RiskClassifier().classify(**kwargs)
