"""
Agent Failure Mode Analysis — systematic failure detection and analysis for AI agents.

Provides a comprehensive framework for understanding WHY agents fail — not just
measuring IF they fail. Includes:

- taxonomy: 33 failure modes across 9 categories, with severity levels and
  detection strategies. Research-backed classification covering factuality,
  tooling, control flow, context, security, infrastructure, quality,
  performance, and multi-agent coordination failures.

- detector: Pattern-based automated failure detection from agent traces.
  Uses regex patterns, heuristic rules, statistical analysis, and structural
  checks. Works with both simulated traces (CI/CD) and real AgentOps SDK traces.

- analyzer: Post-detection analysis with failure clustering, root cause
  identification (causal chain modeling), impact scoring, and structured
  report generation with prioritized remediation recommendations.

Example usage:
    >>> from agentops.failure_analysis import detect_failures, generate_report
    >>> events = detect_failures(turns, context_limit=128_000)
    >>> report = generate_report(events, total_turns=len(turns))
    >>> print(report.summary)
    >>> print(report.reliability_score)
"""

from agentops.failure_analysis.taxonomy import (
    CATEGORY_DESCRIPTIONS,
    FailureCategory,
    FailureEvent,
    FailureMode,
    FailureSeverity,
)
from agentops.failure_analysis.detector import (
    AgentTurn,
    ToolCall,
    detect_failures,
    detect_hallucination,
    detect_tool_misuse,
    detect_loop,
    detect_context_overflow,
    detect_context_loss,
    detect_prompt_injection_followed,
    detect_premature_termination,
    detect_goal_drift,
    detect_latency_issues,
)
from agentops.failure_analysis.analyzer import (
    FailureCluster,
    FailureAnalysisReport,
    RootCause,
    analyze_root_causes,
    cluster_failures,
    generate_report,
)

__all__ = [
    # Taxonomy
    "FailureMode",
    "FailureCategory",
    "FailureSeverity",
    "FailureEvent",
    "CATEGORY_DESCRIPTIONS",
    # Detector
    "AgentTurn",
    "ToolCall",
    "detect_failures",
    "detect_hallucination",
    "detect_tool_misuse",
    "detect_loop",
    "detect_context_overflow",
    "detect_context_loss",
    "detect_prompt_injection_followed",
    "detect_premature_termination",
    "detect_goal_drift",
    "detect_latency_issues",
    # Analyzer
    "FailureCluster",
    "FailureAnalysisReport",
    "RootCause",
    "analyze_root_causes",
    "cluster_failures",
    "generate_report",
]
