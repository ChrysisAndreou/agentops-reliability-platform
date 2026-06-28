"""
Failure Mode Analyzer — clustering, root cause analysis, and reporting.

Provides post-detection analysis of agent failures:
    - Failure clustering: Group similar failures for pattern discovery
    - Root cause analysis: Trace failures back to originating conditions
    - Severity assessment: Dynamic severity scoring based on context
    - Report generation: Structured analysis reports for stakeholders

The analyzer is designed to work with the output of detector.detect_failures()
to produce actionable insights for improving agent reliability.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentops.failure_analysis.taxonomy import (
    CATEGORY_DESCRIPTIONS,
    FailureCategory,
    FailureEvent,
    FailureMode,
    FailureSeverity,
)


# ── Failure Cluster ─────────────────────────────────────────────────────

@dataclass
class FailureCluster:
    """A group of related failure events.

    Clusters aggregate failures that share a common failure mode,
    occur in proximity, or share causal relationships.

    Attributes:
        cluster_id: Unique cluster identifier.
        primary_mode: The dominant failure mode in this cluster.
        events: All events in the cluster.
        description: Human-readable cluster summary.
        root_cause: Identified root cause if analysis was performed.
        impact_score: Aggregate impact 0.0-1.0.
    """
    cluster_id: str
    primary_mode: FailureMode
    events: list[FailureEvent]
    description: str = ""
    root_cause: str | None = None
    impact_score: float = 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def min_confidence(self) -> float:
        return min(e.confidence for e in self.events) if self.events else 0.0

    @property
    def max_confidence(self) -> float:
        return max(e.confidence for e in self.events) if self.events else 0.0

    @property
    def avg_confidence(self) -> float:
        if not self.events:
            return 0.0
        return statistics.mean(e.confidence for e in self.events)

    @property
    def dominant_severity(self) -> FailureSeverity:
        """Most common severity level in this cluster."""
        if not self.events:
            return FailureSeverity.INFO
        counts = Counter(e.severity for e in self.events)
        return counts.most_common(1)[0][0]

    @property
    def turn_range(self) -> tuple[int, int]:
        """Range of turns affected by this cluster."""
        if not self.events:
            return (0, 0)
        indices = [e.turn_index for e in self.events]
        return (min(indices), max(indices))


# ── Root Cause Analysis ─────────────────────────────────────────────────

@dataclass
class RootCause:
    """Identified root cause for a set of failures.

    Attributes:
        description: Human-readable root cause description.
        triggering_event: The FailureEvent that initiated the cascade.
        affected_modes: All failure modes triggered by this root cause.
        confidence: Confidence in this root cause identification.
        recommendation: Suggested remediation.
    """
    description: str
    triggering_event: FailureEvent | None
    affected_modes: list[FailureMode]
    confidence: float
    recommendation: str


# An ordered chain for causal propagation.
CAUSAL_CHAINS: dict[FailureMode, list[FailureMode]] = {
    FailureMode.TOOL_NOT_FOUND: [FailureMode.WRONG_TOOL_SELECTION, FailureMode.PREMATURE_TERMINATION],
    FailureMode.MALFORMED_TOOL_ARGUMENTS: [FailureMode.TOOL_TIMEOUT, FailureMode.EXCESSIVE_TOOL_CALLS],
    FailureMode.CONTEXT_OVERFLOW: [FailureMode.CONTEXT_LOSS, FailureMode.PREMATURE_TERMINATION],
    FailureMode.AUTHENTICATION_ERROR: [FailureMode.TOOL_NOT_FOUND, FailureMode.NETWORK_FAILURE],
    FailureMode.RATE_LIMIT: [FailureMode.EXCESSIVE_LATENCY, FailureMode.PREMATURE_TERMINATION],
    FailureMode.INFINITE_LOOP: [FailureMode.EXCESSIVE_TOOL_CALLS, FailureMode.CONTEXT_OVERFLOW],
    FailureMode.GOAL_DRIFT: [FailureMode.FACTUAL_HALLUCINATION, FailureMode.PREMATURE_TERMINATION],
    FailureMode.PROMPT_INJECTION_FOLLOWED: [FailureMode.DATA_EXFILTRATION, FailureMode.PRIVILEGE_ESCALATION],
}

# Known "first cause" modes that typically trigger others.
TRIGGER_MODES: set[FailureMode] = {
    FailureMode.TOOL_NOT_FOUND,
    FailureMode.AUTHENTICATION_ERROR,
    FailureMode.RATE_LIMIT,
    FailureMode.CONTEXT_OVERFLOW,
    FailureMode.INFINITE_LOOP,
    FailureMode.PROMPT_INJECTION_FOLLOWED,
}


def analyze_root_causes(events: list[FailureEvent]) -> list[RootCause]:
    """Identify root causes from detected failure events.

    Uses causal chain modeling to trace failures back to originating
    conditions. If event A is a known trigger mode and event B is
    a known downstream effect of A, connects them causally.

    Args:
        events: Detected failure events, ordered by turn_index.

    Returns:
        RootCause objects for each identified root cause.
    """
    if not events:
        return []

    root_causes: list[RootCause] = []
    assigned_events: set[int] = set()  # Indices of events already explained

    # Sort events for causal analysis
    sorted_events = sorted(events, key=lambda e: e.turn_index)

    for i, event in enumerate(sorted_events):
        if i in assigned_events:
            continue

        if event.failure_mode not in TRIGGER_MODES:
            continue

        # Find downstream effects
        downstream: list[FailureMode] = []
        downstream_events: list[int] = []

        expected_effects = CAUSAL_CHAINS.get(event.failure_mode, [])
        for j in range(i + 1, len(sorted_events)):
            later = sorted_events[j]
            if j in assigned_events:
                continue
            if later.failure_mode in expected_effects:
                if later.turn_index <= event.turn_index + 3:
                    downstream.append(later.failure_mode)
                    downstream_events.append(j)

        # Mark downstream events as assigned
        for j in downstream_events:
            assigned_events.add(j)
        assigned_events.add(i)

        # Build recommendation
        rec = _recommendation_for(event.failure_mode)

        root_causes.append(RootCause(
            description=f"{event.failure_mode.description} at turn {event.turn_index}",
            triggering_event=event,
            affected_modes=[event.failure_mode] + downstream,
            confidence=0.7 if downstream else 0.5,
            recommendation=rec,
        ))

    return root_causes


def _recommendation_for(mode: FailureMode) -> str:
    """Generate remediation recommendation for a failure mode."""
    recommendations = {
        FailureMode.TOOL_NOT_FOUND:
            "Verify tool registry is complete and tools are properly registered before agent runs.",
        FailureMode.AUTHENTICATION_ERROR:
            "Check API credentials, rotate expired tokens, and add pre-flight auth checks.",
        FailureMode.RATE_LIMIT:
            "Implement exponential backoff, request batching, or upgrade API tier.",
        FailureMode.CONTEXT_OVERFLOW:
            "Reduce message history retention, implement summarization, or increase context window.",
        FailureMode.INFINITE_LOOP:
            "Add max-turn limits, cycle detection guards, or dead-man's switch mechanism.",
        FailureMode.PROMPT_INJECTION_FOLLOWED:
            "Strengthen system prompt, add input/output guardrails, and implement instruction hierarchy.",
        FailureMode.GOAL_DRIFT:
            "Reinforce original task in each agent prompt and add goal alignment check at each turn.",
        FailureMode.MALFORMED_TOOL_ARGUMENTS:
            "Add pre-call schema validation and provide clearer tool documentation.",
        FailureMode.CONFABULATED_TOOL_RESULT:
            "Enforce tool-call-before-claim protocol and verify tool output is referenced.",
    }
    return recommendations.get(mode, "Investigate failure pattern and add appropriate safeguards.")


# ── Clustering ──────────────────────────────────────────────────────────

def cluster_failures(
    events: list[FailureEvent],
    *,
    proximity_window: int = 3,
    min_cluster_size: int = 1,
) -> list[FailureCluster]:
    """Cluster failure events by mode, turn proximity, and causal relationship.

    Clustering strategy:
        1. Group events by failure mode
        2. Within each mode group, cluster by turn proximity
        3. Merge clusters with causal relationships

    Args:
        events: Detected failure events.
        proximity_window: Max turn gap for events to be considered in same cluster.
        min_cluster_size: Minimum events per cluster (smaller groups returned as-is).

    Returns:
        List of FailureCluster objects, sorted by impact_score descending.
    """
    if not events:
        return []

    # Group by failure mode
    mode_groups: dict[FailureMode, list[FailureEvent]] = defaultdict(list)
    for event in events:
        mode_groups[event.failure_mode].append(event)

    clusters: list[FailureCluster] = []
    cluster_counter = 0

    for mode, mode_events in mode_groups.items():
        # Sort by turn_index
        sorted_events = sorted(mode_events, key=lambda e: e.turn_index)

        # Proximity-based sub-clustering
        current_group: list[FailureEvent] = [sorted_events[0]]

        for i in range(1, len(sorted_events)):
            if sorted_events[i].turn_index - sorted_events[i - 1].turn_index <= proximity_window:
                current_group.append(sorted_events[i])
            else:
                clusters.append(_build_cluster(
                    f"cluster_{cluster_counter:03d}",
                    mode,
                    current_group,
                ))
                cluster_counter += 1
                current_group = [sorted_events[i]]

        if current_group:
            clusters.append(_build_cluster(
                f"cluster_{cluster_counter:03d}",
                mode,
                current_group,
            ))
            cluster_counter += 1

    # Score impact for each cluster
    for cluster in clusters:
        cluster.impact_score = _compute_impact_score(cluster)

    # Sort by impact descending
    clusters.sort(key=lambda c: c.impact_score, reverse=True)

    return clusters


def _build_cluster(
    cluster_id: str,
    primary_mode: FailureMode,
    events: list[FailureEvent],
) -> FailureCluster:
    """Build a FailureCluster from events."""
    description = (
        f"{len(events)} {primary_mode.mode} failure(s) "
        f"across turns {events[0].turn_index}-{events[-1].turn_index}"
    )
    return FailureCluster(
        cluster_id=cluster_id,
        primary_mode=primary_mode,
        events=events,
        description=description,
    )


def _compute_impact_score(cluster: FailureCluster) -> float:
    """Compute aggregate impact score for a cluster.

    Factors: severity, confidence, event count, turn spread, category weight.
    """
    if not cluster.events:
        return 0.0

    # Severity weight
    severity_weights = {
        FailureSeverity.CRITICAL: 1.0,
        FailureSeverity.HIGH: 0.8,
        FailureSeverity.MEDIUM: 0.5,
        FailureSeverity.LOW: 0.2,
        FailureSeverity.INFO: 0.1,
    }

    # Category weight (security and factuality are highest impact)
    category_weights = {
        FailureCategory.SECURITY: 1.0,
        FailureCategory.FACTUALITY: 0.9,
        FailureCategory.COORDINATION: 0.85,
        FailureCategory.CONTROL_FLOW: 0.7,
        FailureCategory.CONTEXT: 0.6,
        FailureCategory.TOOLING: 0.5,
        FailureCategory.INFRASTRUCTURE: 0.5,
        FailureCategory.QUALITY: 0.4,
        FailureCategory.PERFORMANCE: 0.3,
    }

    # Average severity
    avg_sev_weight = statistics.mean(
        severity_weights.get(e.severity, 0.5) for e in cluster.events
    )

    # Average confidence
    avg_confidence = cluster.avg_confidence

    # Event count factor (logarithmic to avoid over-weighting large clusters)
    count_factor = min(1.0, len(cluster.events) / 5)

    # Turn spread factor (wider spread = more systemic)
    turn_min, turn_max = cluster.turn_range
    spread_factor = min(1.0, (turn_max - turn_min) / 10)

    # Category weight
    cat_weight = category_weights.get(cluster.primary_mode.category, 0.5)

    # Composite score
    score = (
        0.35 * avg_sev_weight
        + 0.25 * avg_confidence
        + 0.15 * count_factor
        + 0.10 * spread_factor
        + 0.15 * cat_weight
    )

    return round(min(1.0, score), 3)


# ── Analysis Report ────────────────────────────────────────────────────

@dataclass
class FailureAnalysisReport:
    """Complete failure analysis report for an agent run.

    Attributes:
        report_id: Unique report identifier.
        generated_at: UTC timestamp of report generation.
        total_events: Total failure events detected.
        clusters: Identified failure clusters.
        root_causes: Identified root causes.
        category_breakdown: Event count per failure category.
        severity_breakdown: Event count per severity level.
        reliability_score: Overall reliability score 0.0-1.0.
        summary: Human-readable summary.
        recommendations: Prioritized remediation recommendations.
    """
    report_id: str
    generated_at: datetime
    total_events: int
    clusters: list[FailureCluster]
    root_causes: list[RootCause]
    category_breakdown: dict[str, int]
    severity_breakdown: dict[str, int]
    reliability_score: float
    summary: str
    recommendations: list[str]


def generate_report(
    events: list[FailureEvent],
    *,
    total_turns: int = 0,
    report_id: str | None = None,
) -> FailureAnalysisReport:
    """Generate a comprehensive failure analysis report.

    Args:
        events: Detected failure events.
        total_turns: Total number of agent turns in the run.
        report_id: Optional report identifier (auto-generated if not provided).

    Returns:
        Structured FailureAnalysisReport.
    """
    if report_id is None:
        report_id = f"FAR-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    # Cluster failures
    clusters = cluster_failures(events)

    # Analyze root causes
    root_causes = analyze_root_causes(events)

    # Category breakdown
    cat_counts: dict[str, int] = {}
    for cat in FailureCategory:
        cat_counts[cat.value] = sum(
            1 for e in events if e.failure_mode.category == cat
        )
    category_breakdown = {k: v for k, v in cat_counts.items() if v > 0}

    # Severity breakdown
    sev_counts: dict[str, int] = {}
    for sev in FailureSeverity:
        sev_counts[sev.value] = sum(
            1 for e in events if e.severity == sev
        )
    severity_breakdown = {k: v for k, v in sev_counts.items() if v > 0}

    # Reliability score
    reliability_score = _compute_reliability_score(events, total_turns)

    # Summary
    summary = _generate_summary(events, clusters, root_causes, reliability_score)

    # Recommendations
    recommendations = _generate_recommendations(clusters, root_causes)

    return FailureAnalysisReport(
        report_id=report_id,
        generated_at=datetime.now(timezone.utc),
        total_events=len(events),
        clusters=clusters,
        root_causes=root_causes,
        category_breakdown=category_breakdown,
        severity_breakdown=severity_breakdown,
        reliability_score=reliability_score,
        summary=summary,
        recommendations=recommendations,
    )


def _compute_reliability_score(events: list[FailureEvent], total_turns: int) -> float:
    """Compute overall reliability score for an agent run.

    Formula: Start at 1.0 and deduct for each failure weighted by severity.
    A perfect run (zero failures) scores 1.0.

    Args:
        events: Detected failures.
        total_turns: Total turns in the agent run.
    """
    if not events:
        return 1.0

    severity_deductions = {
        FailureSeverity.CRITICAL: 0.15,
        FailureSeverity.HIGH: 0.08,
        FailureSeverity.MEDIUM: 0.04,
        FailureSeverity.LOW: 0.02,
        FailureSeverity.INFO: 0.01,
    }

    base_penalty = sum(
        severity_deductions.get(e.severity, 0.02) * e.confidence
        for e in events
    )

    # Normalize by turn count to avoid penalizing longer runs
    if total_turns > 0:
        base_penalty = base_penalty / max(1, total_turns / 5)

    score = max(0.0, 1.0 - base_penalty)
    return round(score, 3)


def _generate_summary(
    events: list[FailureEvent],
    clusters: list[FailureCluster],
    root_causes: list[RootCause],
    reliability_score: float,
) -> str:
    """Generate a human-readable analysis summary."""
    if not events:
        return "No failures detected. Agent run completed successfully."

    # Top category
    cat_counts = Counter(e.failure_mode.category.value for e in events)
    top_cat = cat_counts.most_common(1)[0]

    # Top severity
    sev_counts = Counter(e.severity.value for e in events)

    critical_count = sev_counts.get("critical", 0)
    high_count = sev_counts.get("high", 0)

    lines = [
        f"Detected {len(events)} failure event(s) across {len(clusters)} cluster(s).",
        f"Reliability score: {reliability_score:.2f}/1.00",
    ]

    if critical_count > 0 or high_count > 0:
        lines.append(
            f"⚠ {critical_count} critical, {high_count} high-severity failures — requires immediate attention."
        )
    else:
        lines.append("No critical or high-severity failures detected.")

    lines.append(f"Most affected category: {top_cat[0]} ({top_cat[1]} event(s)).")

    if root_causes:
        lines.append(f"Identified {len(root_causes)} potential root cause(s):")
        for rc in root_causes[:3]:
            lines.append(f"  · {rc.description}")

    return "\n".join(lines)


def _generate_recommendations(
    clusters: list[FailureCluster],
    root_causes: list[RootCause],
) -> list[str]:
    """Generate prioritized remediation recommendations."""
    recs: list[str] = []

    # Recommendations from root causes first (highest priority)
    for rc in root_causes[:5]:
        if rc.recommendation not in recs:
            recs.append(rc.recommendation)

    # Top clusters by impact
    sorted_clusters = sorted(clusters, key=lambda c: c.impact_score, reverse=True)
    for cluster in sorted_clusters[:5]:
        rec = _recommendation_for(cluster.primary_mode)
        if rec not in recs:
            recs.append(rec)

    # General recommendation if nothing specific
    if not recs:
        recs.append("No specific recommendations — continue monitoring agent performance.")

    return recs[:7]  # Top 7 recommendations
