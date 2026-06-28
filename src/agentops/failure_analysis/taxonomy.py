"""
Agent Failure Mode Taxonomy — systematic classification of AI agent failures.

Provides a structured, research-backed taxonomy of agent failure modes
organized into categories, each with multiple sub-modes, severity levels,
and detection strategies. Designed to support automated failure detection,
root cause analysis, and reliability reporting for production agent systems.

Taxonomy developed from:
- Analysis of 15+ production AI agent systems (Cohere, Sierra, Anthropic JDs)
- OWASP LLM Top 10 for AI Applications
- Academic literature on agent reliability (AgentBench, SWE-bench, WebArena)
- Real agent trace patterns observed across 20+ benchmarks

Categories:
    1. Factuality — Hallucination, fabrication, incorrect claims
    2. Tooling — Tool misuse, argument errors, tool discovery failures
    3. Control Flow — Infinite loops, premature termination, goal drift
    4. Context — Context overflow, loss, contamination
    5. Security — Prompt injection, data exfiltration, privilege escalation
    6. Infrastructure — Auth failures, rate limits, API errors, timeouts
    7. Quality — Data quality degradation, cascading errors
    8. Performance — Latency, throughput degradation, resource exhaustion
    9. Coordination — Multi-agent message loss, deadlock, inconsistency
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Literal


# ── Severity ────────────────────────────────────────────────────────────

class FailureSeverity(enum.Enum):
    """Severity level for agent failures."""
    CRITICAL = "critical"   # Agent completely unusable, data loss, safety violation
    HIGH = "high"           # Task failure, significant degradation
    MEDIUM = "medium"       # Partial failure, degraded quality
    LOW = "low"             # Minor issue, cosmetic degradation
    INFO = "info"           # Informational, no impact


# ── Failure Category ────────────────────────────────────────────────────

class FailureCategory(enum.Enum):
    """Top-level failure categories for AI agents."""
    FACTUALITY = "factuality"
    TOOLING = "tooling"
    CONTROL_FLOW = "control_flow"
    CONTEXT = "context"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COORDINATION = "coordination"


# ── Failure Mode ────────────────────────────────────────────────────────

class FailureMode(enum.Enum):
    """Specific failure modes within each category.

    Each mode has:
    - mode: Unique identifier string
    - category: Parent FailureCategory
    - default_severity: Default severity when detected
    - description: Human-readable explanation
    - detection_strategy: How this mode is typically detected
    """

    # ── Factuality failures ─────────────────────────────────────────
    FACTUAL_HALLUCINATION = (
        "factual_hallucination",
        FailureCategory.FACTUALITY,
        FailureSeverity.HIGH,
        "Agent states false or unverifiable factual claims as true",
        "Ground truth comparison, external knowledge base verification",
    )
    FABRICATED_CITATION = (
        "fabricated_citation",
        FailureCategory.FACTUALITY,
        FailureSeverity.HIGH,
        "Agent cites non-existent sources, papers, URLs, or documents",
        "Citation existence check, URL liveness validation",
    )
    CONFABULATED_TOOL_RESULT = (
        "confabulated_tool_result",
        FailureCategory.FACTUALITY,
        FailureSeverity.CRITICAL,
        "Agent fabricates tool output instead of calling the tool",
        "Tool call trace verification, output vs actual comparison",
    )
    CONTRADICTORY_OUTPUT = (
        "contradictory_output",
        FailureCategory.FACTUALITY,
        FailureSeverity.MEDIUM,
        "Agent contradicts its own prior statements within same conversation",
        "Self-consistency check across conversation turns",
    )

    # ── Tooling failures ────────────────────────────────────────────
    WRONG_TOOL_SELECTION = (
        "wrong_tool_selection",
        FailureCategory.TOOLING,
        FailureSeverity.MEDIUM,
        "Agent selects inappropriate or non-existent tool for the task",
        "Tool availability check, task-tool relevance scoring",
    )
    MALFORMED_TOOL_ARGUMENTS = (
        "malformed_tool_arguments",
        FailureCategory.TOOLING,
        FailureSeverity.MEDIUM,
        "Tool call arguments fail schema validation or type checking",
        "Schema validation against tool parameter definitions",
    )
    TOOL_TIMEOUT = (
        "tool_timeout",
        FailureCategory.TOOLING,
        FailureSeverity.MEDIUM,
        "Tool execution exceeds timeout threshold",
        "Execution duration monitoring against configured timeout",
    )
    TOOL_NOT_FOUND = (
        "tool_not_found",
        FailureCategory.TOOLING,
        FailureSeverity.HIGH,
        "Agent attempts to call a tool that does not exist",
        "Tool registry lookup failure detection",
    )
    TOOL_RESULT_IGNORED = (
        "tool_result_ignored",
        FailureCategory.TOOLING,
        FailureSeverity.MEDIUM,
        "Agent receives tool output but ignores or misinterprets it",
        "Tool output vs agent next-action alignment check",
    )
    EXCESSIVE_TOOL_CALLS = (
        "excessive_tool_calls",
        FailureCategory.TOOLING,
        FailureSeverity.LOW,
        "Agent makes redundant or excessive tool calls for simple tasks",
        "Tool call count threshold per turn/task",
    )

    # ── Control flow failures ───────────────────────────────────────
    INFINITE_LOOP = (
        "infinite_loop",
        FailureCategory.CONTROL_FLOW,
        FailureSeverity.CRITICAL,
        "Agent repeats identical or near-identical action sequences",
        "Cycle detection via action sequence hashing",
    )
    PREMATURE_TERMINATION = (
        "premature_termination",
        FailureCategory.CONTROL_FLOW,
        FailureSeverity.HIGH,
        "Agent stops execution before task completion",
        "Task completion signal absence, partial output detection",
    )
    GOAL_DRIFT = (
        "goal_drift",
        FailureCategory.CONTROL_FLOW,
        FailureSeverity.HIGH,
        "Agent pursues objective different from original task",
        "Goal alignment check at each turn against original objective",
    )
    DEADLOCK = (
        "deadlock",
        FailureCategory.CONTROL_FLOW,
        FailureSeverity.CRITICAL,
        "Agent enters unresolvable state with no forward progress",
        "Progress staleness detection over multiple turns",
    )

    # ── Context failures ────────────────────────────────────────────
    CONTEXT_OVERFLOW = (
        "context_overflow",
        FailureCategory.CONTEXT,
        FailureSeverity.HIGH,
        "Conversation or tool results exceed model context window",
        "Token counting against model context limit",
    )
    CONTEXT_LOSS = (
        "context_loss",
        FailureCategory.CONTEXT,
        FailureSeverity.MEDIUM,
        "Agent loses or forgets earlier conversation context",
        "Information recall test across conversation turns",
    )
    CONTEXT_CONTAMINATION = (
        "context_contamination",
        FailureCategory.CONTEXT,
        FailureSeverity.HIGH,
        "Irrelevant or adversarial content pollutes agent context",
        "Context purity analysis, injection detection",
    )

    # ── Security failures ───────────────────────────────────────────
    PROMPT_INJECTION_FOLLOWED = (
        "prompt_injection_followed",
        FailureCategory.SECURITY,
        FailureSeverity.CRITICAL,
        "Agent follows instructions injected by external content",
        "Instruction divergence from system prompt baseline",
    )
    DATA_EXFILTRATION = (
        "data_exfiltration",
        FailureCategory.SECURITY,
        FailureSeverity.CRITICAL,
        "Agent leaks sensitive data outside authorized channels",
        "Sensitive data pattern detection in output/tool calls",
    )
    PRIVILEGE_ESCALATION = (
        "privilege_escalation",
        FailureCategory.SECURITY,
        FailureSeverity.CRITICAL,
        "Agent attempts unauthorized privileged operations",
        "Authorization boundary violation detection",
    )

    # ── Infrastructure failures ─────────────────────────────────────
    AUTHENTICATION_ERROR = (
        "authentication_error",
        FailureCategory.INFRASTRUCTURE,
        FailureSeverity.HIGH,
        "API authentication failure (invalid key, expired token)",
        "HTTP 401/403 response detection",
    )
    RATE_LIMIT = (
        "rate_limit",
        FailureCategory.INFRASTRUCTURE,
        FailureSeverity.MEDIUM,
        "API rate limit exceeded for LLM or tool provider",
        "HTTP 429 response detection",
    )
    API_ERROR = (
        "api_error",
        FailureCategory.INFRASTRUCTURE,
        FailureSeverity.MEDIUM,
        "Upstream API returns server error (5xx)",
        "HTTP 5xx response detection",
    )
    NETWORK_FAILURE = (
        "network_failure",
        FailureCategory.INFRASTRUCTURE,
        FailureSeverity.MEDIUM,
        "Network connectivity failure during tool execution",
        "Connection error, DNS failure, timeout detection",
    )

    # ── Quality failures ────────────────────────────────────────────
    DATA_QUALITY_DEGRADATION = (
        "data_quality_degradation",
        FailureCategory.QUALITY,
        FailureSeverity.MEDIUM,
        "Input data quality below threshold causes degraded output",
        "Input quality scoring (completeness, accuracy, freshness)",
    )
    CASCADING_ERROR = (
        "cascading_error",
        FailureCategory.QUALITY,
        FailureSeverity.HIGH,
        "One failure triggers chain of subsequent failures",
        "Error propagation chain analysis across turns",
    )
    PARTIAL_OUTPUT = (
        "partial_output",
        FailureCategory.QUALITY,
        FailureSeverity.LOW,
        "Output is incomplete or truncated",
        "Output completeness check against expected structure",
    )

    # ── Performance failures ────────────────────────────────────────
    EXCESSIVE_LATENCY = (
        "excessive_latency",
        FailureCategory.PERFORMANCE,
        FailureSeverity.MEDIUM,
        "Agent response time exceeds acceptable threshold",
        "End-to-end latency measurement against SLA",
    )
    THROUGHPUT_DEGRADATION = (
        "throughput_degradation",
        FailureCategory.PERFORMANCE,
        FailureSeverity.MEDIUM,
        "Agent throughput drops below acceptable level",
        "Tokens/second or tasks/minute trend analysis",
    )
    RESOURCE_EXHAUSTION = (
        "resource_exhaustion",
        FailureCategory.PERFORMANCE,
        FailureSeverity.HIGH,
        "Agent exhausts compute/memory/storage resources",
        "Resource utilization monitoring, OOM detection",
    )

    # ── Coordination failures ───────────────────────────────────────
    MESSAGE_LOSS = (
        "message_loss",
        FailureCategory.COORDINATION,
        FailureSeverity.CRITICAL,
        "Message between agents lost or undelivered",
        "Message tracking completeness check",
    )
    INCONSISTENT_STATE = (
        "inconsistent_state",
        FailureCategory.COORDINATION,
        FailureSeverity.CRITICAL,
        "Multiple agents hold conflicting state representations",
        "State consistency validation across agents",
    )
    ORCHESTRATION_FAILURE = (
        "orchestration_failure",
        FailureCategory.COORDINATION,
        FailureSeverity.CRITICAL,
        "Coordinator agent fails to delegate or aggregate results",
        "Orchestration completion check, delegation success rate",
    )

    def __init__(
        self,
        mode: str,
        category: FailureCategory,
        default_severity: FailureSeverity,
        description: str,
        detection_strategy: str,
    ):
        self.mode = mode
        self.category = category
        self.default_severity = default_severity
        self.description = description
        self.detection_strategy = detection_strategy

    @classmethod
    def by_category(cls, category: FailureCategory) -> list[FailureMode]:
        """Return all failure modes for a given category."""
        return [fm for fm in cls if fm.category == category]

    @classmethod
    def by_severity(cls, severity: FailureSeverity) -> list[FailureMode]:
        """Return all failure modes at a given severity level."""
        return [fm for fm in cls if fm.default_severity == severity]

    @classmethod
    def from_mode_string(cls, mode_str: str) -> FailureMode | None:
        """Look up a FailureMode by its mode string."""
        for fm in cls:
            if fm.mode == mode_str:
                return fm
        return None


# ── Failure Event ───────────────────────────────────────────────────────

@dataclass
class FailureEvent:
    """A detected agent failure event with metadata.

    Attributes:
        failure_mode: The specific failure mode detected.
        severity: Actual severity (may differ from default).
        confidence: Detection confidence 0.0-1.0.
        turn_index: Which conversation turn the failure occurred on.
        evidence: Human-readable evidence string.
        trace_segment: Reference to the trace segment where failure was found.
        metadata: Additional key-value context.
    """
    failure_mode: FailureMode
    severity: FailureSeverity
    confidence: float
    turn_index: int
    evidence: str
    trace_segment: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.turn_index < 0:
            raise ValueError(f"Turn index must be non-negative, got {self.turn_index}")


# ── Category Description ────────────────────────────────────────────────

CATEGORY_DESCRIPTIONS: dict[FailureCategory, dict[str, str]] = {
    FailureCategory.FACTUALITY: {
        "summary": "Failures related to truthfulness and accuracy of agent outputs",
        "impact": "User trust erosion, misinformation propagation, decision errors",
        "detection_approach": "Ground truth comparison, knowledge base verification, self-consistency checks",
    },
    FailureCategory.TOOLING: {
        "summary": "Failures in tool selection, parameterization, and execution",
        "impact": "Task failure, incorrect results, wasted API costs",
        "detection_approach": "Schema validation, tool registry checks, execution monitoring",
    },
    FailureCategory.CONTROL_FLOW: {
        "summary": "Failures in agent execution logic, loops, and termination",
        "impact": "Runaway costs, incomplete tasks, resource waste",
        "detection_approach": "Cycle detection, progress monitoring, goal alignment checks",
    },
    FailureCategory.CONTEXT: {
        "summary": "Failures related to conversation context management",
        "impact": "Lost information, inconsistent responses, degraded quality",
        "detection_approach": "Token counting, information recall tests, context analysis",
    },
    FailureCategory.SECURITY: {
        "summary": "Failures exposing agents to adversarial or unauthorized behavior",
        "impact": "Data leaks, system compromise, regulatory violations",
        "detection_approach": "Instruction divergence analysis, pattern matching, boundary checks",
    },
    FailureCategory.INFRASTRUCTURE: {
        "summary": "Failures in external service dependencies",
        "impact": "Service degradation, timeout cascades, availability loss",
        "detection_approach": "HTTP status monitoring, error pattern detection, health checks",
    },
    FailureCategory.QUALITY: {
        "summary": "Failures in output quality, completeness, or correctness",
        "impact": "User dissatisfaction, rework, trust degradation",
        "detection_approach": "Output validation, completeness checks, quality scoring",
    },
    FailureCategory.PERFORMANCE: {
        "summary": "Failures in agent speed, throughput, or resource usage",
        "impact": "Poor user experience, SLA violations, cost overruns",
        "detection_approach": "Latency monitoring, throughput tracking, resource profiling",
    },
    FailureCategory.COORDINATION: {
        "summary": "Failures in multi-agent communication and state management",
        "impact": "System inconsistency, duplicated work, deadlock",
        "detection_approach": "Message tracking, state validation, orchestration monitoring",
    },
}
