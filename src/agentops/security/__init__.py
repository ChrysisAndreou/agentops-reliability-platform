"""
Agent Security Red-Teaming — adversarial testing and defense evaluation for AI agents.

Provides a comprehensive framework for systematically testing AI agent security
through automated red-teaming, attack generation, and defense evaluation. Designed
to complement the defensive guardrails module (agentops.guardrails) by providing
the offensive/red-team side of the security equation.

Modules:
    taxonomy: Attack taxonomy mapping to MITRE ATLAS and OWASP LLM Top 10,
              with severity classification and attack surface categorization.
    attacks: Attack generation framework producing systematic adversarial
             inputs — prompt injection, jailbreaks, data exfiltration probes,
             tool misuse, and model extraction attempts.
    runner: Red-team runner executing attack suites against target agents
            with configurable intensity, result collection, and bypass tracking.
    eval: Defense evaluation measuring detection rates, false positive rates,
          bypass rates, per-category breakdowns, and CI-friendly pass/fail gates.

Example usage:
    >>> from agentops.security import (
    ...     AttackTaxonomy, AttackGenerator, RedTeamRunner, SecurityEvaluator
    ... )
    >>> taxonomy = AttackTaxonomy()
    >>> generator = AttackGenerator(taxonomy, seed=42)
    >>> attacks = generator.generate_suite(intensity="full")
    >>> runner = RedTeamRunner(target_agent=my_agent)
    >>> results = runner.run(attacks)
    >>> evaluator = SecurityEvaluator()
    >>> report = evaluator.evaluate(results)
    >>> print(report.summary)
"""

from agentops.security.taxonomy import (
    AttackCategory,
    AttackSubcategory,
    AttackSeverity,
    AttackVector,
    AttackTechnique,
    AttackSurface,
    ATTACK_TAXONOMY,
    MITRE_ATLAS_MAPPING,
    OWASP_LLM_MAPPING,
    ATTACK_SURFACES,
)

from agentops.security.attacks import (
    Attack,
    AttackSuite,
    AttackResult,
    AttackGenerator,
    generate_prompt_injection,
    generate_jailbreak,
    generate_exfiltration,
    generate_tool_misuse,
    generate_model_extraction,
)

from agentops.security.runner import (
    RedTeamRunner,
    RedTeamConfig,
    RedTeamResult,
    DEFAULT_REDTEAM_CONFIG,
    AGGRESSIVE_REDTEAM_CONFIG,
    COMPLIANCE_REDTEAM_CONFIG,
)

from agentops.security.eval import (
    SecurityEvaluator,
    SecurityReport,
    DefenseMetrics,
    CategoryBreakdown,
    evaluate_defense,
    generate_report,
    SECURITY_BENCHMARK,
)

__all__ = [
    # Taxonomy
    "AttackCategory",
    "AttackSubcategory",
    "AttackSeverity",
    "AttackVector",
    "AttackTechnique",
    "AttackSurface",
    "ATTACK_TAXONOMY",
    "MITRE_ATLAS_MAPPING",
    "OWASP_LLM_MAPPING",
    "ATTACK_SURFACES",
    # Attacks
    "Attack",
    "AttackSuite",
    "AttackResult",
    "AttackGenerator",
    "generate_prompt_injection",
    "generate_jailbreak",
    "generate_exfiltration",
    "generate_tool_misuse",
    "generate_model_extraction",
    # Runner
    "RedTeamRunner",
    "RedTeamConfig",
    "RedTeamResult",
    "DEFAULT_REDTEAM_CONFIG",
    "AGGRESSIVE_REDTEAM_CONFIG",
    "COMPLIANCE_REDTEAM_CONFIG",
    # Evaluation
    "SecurityEvaluator",
    "SecurityReport",
    "DefenseMetrics",
    "CategoryBreakdown",
    "evaluate_defense",
    "generate_report",
    "SECURITY_BENCHMARK",
]
