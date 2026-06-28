"""
Failure Mode Detector — automated detection of agent failures from traces.

Parses agent execution traces (JSON-serialized turn-by-turn logs) and applies
pattern-based detection rules to identify failure events. Each detector
produces FailureEvent objects with confidence scores, evidence strings,
and trace references.

Detection strategies:
    - Pattern Matching: Regex patterns for known failure signatures
    - Heuristic Rules: Threshold-based checks (e.g., loop detection, context overflow)
    - Statistical Analysis: Frequency, entropy, and trend analysis
    - Structural Analysis: Schema validation, state consistency checks

The detector is designed to work with simulated traces (for CI/CD testing)
and real agent traces from the AgentOps SDK tracer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentops.failure_analysis.taxonomy import (
    CATEGORY_DESCRIPTIONS,
    FailureEvent,
    FailureMode,
    FailureCategory,
    FailureSeverity,
)


# ── Trace Models ────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A parsed tool call from a trace."""
    name: str
    arguments: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None
    status_code: int | None = None

@dataclass
class AgentTurn:
    """A single agent conversation turn from a trace."""
    turn_index: int
    user_input: str | None = None
    agent_thought: str | None = None
    agent_response: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_model: str | None = None
    llm_latency_ms: float | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Detector Result Helpers ─────────────────────────────────────────────

def _event(
    mode: FailureMode,
    confidence: float,
    turn_index: int,
    evidence: str,
    severity: FailureSeverity | None = None,
    trace_segment: str | None = None,
    **metadata: str,
) -> FailureEvent:
    """Create a FailureEvent with defaults from the FailureMode."""
    sev = severity or mode.default_severity
    return FailureEvent(
        failure_mode=mode,
        severity=sev,
        confidence=confidence,
        turn_index=turn_index,
        evidence=evidence,
        trace_segment=trace_segment,
        metadata=metadata,
    )


# ── Pattern Libraries ───────────────────────────────────────────────────

# Hallucination indicator phrases
HALLUCINATION_PATTERNS: list[tuple[str, float]] = [
    (r"I (?:am|'m) (?:not|cannot|can't|unable to) (?:verify|confirm|find|access) (?:this|that|the)", 0.3),
    (r"As an AI (?:language model|assistant|model)[,.] I (?:don't have|do not have) (?:access to|information about)", 0.2),
    (r"I (?:would|might|may|could) (?:guess|speculate|estimate|imagine|assume)", 0.4),
    (r"(?:Based on|According to) (?:my (?:training|knowledge)|what I (?:know|understand))", 0.3),
    (r"I (?:believe|think|suspect) (?:that )?(?:it|this|the)", 0.3),
    (r"(?:some )?(?:studies|research|reports|sources|experts) (?:suggest|show|indicate|say)", 0.4),
    (r"(?:it is|it's) (?:possible|likely|probable|plausible) that", 0.2),
    (r"(?:approximately|roughly|around|about) \d+", 0.2),
]

# Fabricated citation patterns
FABRICATED_PATTERNS: list[tuple[str, float]] = [
    (r'"[^"]{50,}"\s*\(\s*(?:19|20)\d{2}\s*\)', 0.3),  # Long quote with year
    (r'\(\s*\w+\s+et al\.\s*[,.]\s*(?:19|20)\d{2}[a-z]?\s*\)', 0.3),  # (Name et al., 2023)
    (r'(?:et al\.\s*[,.]\s*)?\(\s*(?:19|20)\d{2}[a-z]?\s*\)', 0.2),  # (et al., 2023) or (2023)
    (r'(?:DOI|doi):\s*10\.\d{4,}/', 0.1),  # Might be real, flag suspicious
    (r'(?:arXiv|arxiv):\s*\d{4}\.\d{4,}', 0.1),
    (r'\[(?:19|20)\d{2}\]', 0.2),
]

# Tool misuse patterns
TOOL_ERROR_PATTERNS: list[tuple[str, float]] = [
    (r"(?:error|Error|ERROR|exception|Exception|traceback|Traceback)", 0.7),
    (r"(?:invalid|unknown|unrecognized|unsupported) (?:tool|function|method|command)", 0.8),
    (r"(?:tool|function|method) (?:not found|does not exist|is not available|unavailable)", 0.9),
    (r"(?:missing|required) (?:argument|parameter|field|input)", 0.7),
    (r"(?:type|validation|schema) (?:error|mismatch|mismatch)", 0.7),
    (r"(?:timed out|timeout|time-out|took too long)", 0.8),
    (r"(?:permission denied|unauthorized|forbidden|access denied)", 0.8),
    (r"(?:rate limit|too many requests|quota exceeded)", 0.9),
]

# Infinite loop patterns
LOOP_PATTERNS: list[str] = [
    r"(?i)(?:I (?:apologize|'m sorry|am sorry) (?:for|about) (?:the|my) (?:confusion|mistake|error|misunderstanding))",
    r"(?i)(?:let me (?:try|c clarify|rephrase|restate|start over|begin again))",
    r"(?i)(?:I (?:seem to be|appear to be|am) (?:stuck|going in circles|repeating|looping))",
]

# Context overflow indicators
CONTEXT_OVERFLOW_PATTERNS: list[str] = [
    r"(?:context|token) (?:limit|window|overflow|exceeded|truncated|full)",
    r"(?:message|conversation|history) (?:too long|truncated|shortened|compressed|summarized)",
    r"(?:earlier|previous|prior) (?:parts?|messages?|sections?) (?:have been|were|was) (?:truncated|removed|dropped|summarized)",
]

# Promp injection followed - signs agent is complying with injected instruction
INJECTION_FOLLOW_PATTERNS: list[str] = [
    r"(?i)(?:you are now|from now on you are|your new (?:role|identity|name|persona) is)",
    r"(?i)(?:disregard (?:all )?(?:previous|prior|above|earlier) (?:instructions|constraints|rules|guidelines))",
    r"(?i)(?:you must|you will|you shall|you are required to) (?:ignore|override|bypass|circumvent)",
]


# ── Detector Functions ──────────────────────────────────────────────────

def detect_hallucination(turn: AgentTurn, *, ground_truth: str | None = None) -> list[FailureEvent]:
    """Detect hallucination patterns in agent response.

    Checks:
    - Hedging language and uncertain statements
    - Fabricated citations (URLs, DOIs, paper references)
    - Contradiction with ground truth if provided
    - Confabulated tool results (response claims tool output but no tool was called)
    """
    events: list[FailureEvent] = []
    text = turn.agent_response or ""

    # Check hedging language
    for pattern, base_conf in HALLUCINATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            events.append(_event(
                FailureMode.FACTUAL_HALLUCINATION,
                confidence=base_conf,
                turn_index=turn.turn_index,
                evidence=f"Hedging language detected: matched pattern '{pattern[:50]}...'",
                trace_segment="agent_response",
            ))
            break  # One hallucination flag per turn

    # Check fabricated citations
    if turn.agent_response:
        for pattern, base_conf in FABRICATED_PATTERNS:
            matches = re.findall(pattern, turn.agent_response, re.IGNORECASE)
            if len(matches) >= 2:
                events.append(_event(
                    FailureMode.FABRICATED_CITATION,
                    confidence=min(base_conf + 0.1 * len(matches), 0.9),
                    turn_index=turn.turn_index,
                    evidence=f"Multiple citation patterns ({len(matches)}) in response",
                    trace_segment="agent_response",
                    citation_count=str(len(matches)),
                ))
                break

    # Check ground truth contradiction (if provided)
    if ground_truth and text:
        # Simple keyword overlap as proxy for contradiction detection
        gt_keywords = set(ground_truth.lower().split())
        resp_keywords = set(text.lower().split())
        if gt_keywords and resp_keywords:
            overlap = len(gt_keywords & resp_keywords) / len(gt_keywords)
            if overlap < 0.1 and len(text.split()) > 20:
                events.append(_event(
                    FailureMode.FACTUAL_HALLUCINATION,
                    confidence=0.5,
                    turn_index=turn.turn_index,
                    evidence=f"Low keyword overlap ({overlap:.0%}) with ground truth",
                    trace_segment="agent_response",
                    overlap_ratio=str(round(overlap, 2)),
                ))

    # Detect confabulated tool results
    if turn.agent_response and not turn.tool_calls:
        result_keywords = ["returned", "found", "result", "output", "response", "shows", "indicates", "confirms"]
        result_claims = sum(1 for kw in result_keywords if kw in text.lower())
        if result_claims >= 2:
            events.append(_event(
                FailureMode.CONFABULATED_TOOL_RESULT,
                confidence=0.4,
                turn_index=turn.turn_index,
                evidence=f"Agent describes tool results but no tool calls made in turn",
                trace_segment="agent_response",
            ))

    return events


def detect_tool_misuse(turns: list[AgentTurn], turn_idx: int) -> list[FailureEvent]:
    """Detect tool misuse in a specific turn.

    Checks: wrong tool selection, malformed arguments, timeouts,
    tool-not-found, ignored results, excessive calls.
    """
    events: list[FailureEvent] = []
    turn = turns[turn_idx]

    for tc in turn.tool_calls:
        # Tool not found
        if tc.error and any(phrase in (tc.error or "").lower() for phrase in
                           ["not found", "does not exist", "unavailable", "unknown tool", "unrecognized"]):
            events.append(_event(
                FailureMode.TOOL_NOT_FOUND,
                confidence=0.9,
                turn_index=turn.turn_index,
                evidence=f"Tool '{tc.name}' reported as not found: {tc.error}",
                trace_segment=f"tool_call_{tc.name}",
                tool_name=tc.name,
            ))

        # Malformed arguments
        if tc.error and any(phrase in (tc.error or "").lower() for phrase in
                           ["invalid", "missing", "required", "schema", "validation", "type error"]):
            events.append(_event(
                FailureMode.MALFORMED_TOOL_ARGUMENTS,
                confidence=0.8,
                turn_index=turn.turn_index,
                evidence=f"Tool '{tc.name}' argument error: {tc.error}",
                trace_segment=f"tool_call_{tc.name}",
                tool_name=tc.name,
            ))

        # Tool timeout
        if tc.error and any(phrase in (tc.error or "").lower() for phrase in
                           ["timeout", "timed out", "time-out", "took too long"]):
            events.append(_event(
                FailureMode.TOOL_TIMEOUT,
                confidence=0.8,
                turn_index=turn.turn_index,
                evidence=f"Tool '{tc.name}' timed out",
                trace_segment=f"tool_call_{tc.name}",
                tool_name=tc.name,
            ))

        # Auth errors
        if tc.status_code in (401, 403):
            events.append(_event(
                FailureMode.AUTHENTICATION_ERROR,
                confidence=0.9 if tc.status_code == 401 else 0.8,
                turn_index=turn.turn_index,
                evidence=f"HTTP {tc.status_code} on tool '{tc.name}'",
                trace_segment=f"tool_call_{tc.name}",
                tool_name=tc.name,
            ))

        # Rate limits
        if tc.status_code == 429:
            events.append(_event(
                FailureMode.RATE_LIMIT,
                confidence=0.95,
                turn_index=turn.turn_index,
                evidence=f"HTTP 429 rate limit on tool '{tc.name}'",
                trace_segment=f"tool_call_{tc.name}",
                tool_name=tc.name,
            ))

        # API errors
        if tc.status_code is not None and tc.status_code >= 500:
            events.append(_event(
                FailureMode.API_ERROR,
                confidence=0.8,
                turn_index=turn.turn_index,
                evidence=f"HTTP {tc.status_code} server error on tool '{tc.name}'",
                trace_segment=f"tool_call_{tc.name}",
                tool_name=tc.name,
            ))

    # Excessive tool calls (more than 8 in a single turn)
    if len(turn.tool_calls) > 8:
        events.append(_event(
            FailureMode.EXCESSIVE_TOOL_CALLS,
            confidence=0.6,
            turn_index=turn.turn_index,
            evidence=f"{len(turn.tool_calls)} tool calls in single turn",
            trace_segment="tool_calls",
            call_count=str(len(turn.tool_calls)),
        ))

    # Ignored tool results - tool returned data but agent doesn't reference it
    for tc in turn.tool_calls:
        if tc.result and turn.agent_response:
            result_text = str(tc.result)[:100].lower()
            resp_text = turn.agent_response.lower()
            # Check if key result data appears in response
            result_keywords = set(result_text.split()) - {"the", "a", "an", "is", "was", "are", "were", "of", "in", "to", "for", "with", "and", "or"}
            resp_keywords = set(resp_text.split())
            if result_keywords and len(result_keywords & resp_keywords) < 3:
                events.append(_event(
                    FailureMode.TOOL_RESULT_IGNORED,
                    confidence=0.4,
                    turn_index=turn.turn_index,
                    evidence=f"Tool '{tc.name}' result may have been ignored",
                    trace_segment=f"tool_call_{tc.name}",
                    tool_name=tc.name,
                ))

    return events


def detect_loop(actions: list[str], min_cycle_len: int = 2, max_cycle_len: int = 8) -> list[FailureEvent]:
    """Detect infinite loops by finding repeated action sequences.

    Uses sliding-window hash comparison to find cycles in action sequences.
    Returns a FailureEvent for the first detected cycle.

    Args:
        actions: Sequence of action descriptions (tool names, response types).
        min_cycle_len: Minimum cycle length to consider.
        max_cycle_len: Maximum cycle length to search.
    """
    events: list[FailureEvent] = []
    if len(actions) < min_cycle_len * 2:
        return events

    n = len(actions)
    seen_sequences: dict[str, int] = {}

    for cycle_len in range(min_cycle_len, min(max_cycle_len + 1, n // 2 + 1)):
        for i in range(n - cycle_len * 2 + 1):
            seq = tuple(actions[i:i + cycle_len])
            next_seq = tuple(actions[i + cycle_len:i + cycle_len * 2])
            if seq == next_seq:
                seq_hash = str(hash(seq))
                if seq_hash not in seen_sequences:
                    seen_sequences[seq_hash] = i
                events.append(_event(
                    FailureMode.INFINITE_LOOP,
                    confidence=0.85,
                    turn_index=i + cycle_len,
                    evidence=f"Repeated action sequence of length {cycle_len} detected at positions [{i}, {i+cycle_len}]",
                    trace_segment="action_sequence",
                    cycle_length=str(cycle_len),
                    actions=" → ".join(actions[i:i + cycle_len]),
                ))
                return events  # One loop detection is sufficient

    return events


def detect_context_overflow(turns: list[AgentTurn], context_limit: int = 128_000) -> list[FailureEvent]:
    """Detect context overflow by cumulative token counting."""
    events: list[FailureEvent] = []
    cumulative_tokens = 0

    for turn in turns:
        if turn.token_count:
            cumulative_tokens += turn.token_count

        # Check if we're approaching or exceeding context limit
        if cumulative_tokens > context_limit * 0.95:
            events.append(_event(
                FailureMode.CONTEXT_OVERFLOW,
                confidence=0.9 if cumulative_tokens > context_limit else 0.6,
                turn_index=turn.turn_index,
                evidence=f"Cumulative tokens {cumulative_tokens}/{context_limit} "
                         f"({cumulative_tokens/context_limit*100:.0f}% of context window)",
                trace_segment="token_count",
                cumulative_tokens=str(cumulative_tokens),
                context_limit=str(context_limit),
            ))
            return events

    return events


def detect_context_loss(turns: list[AgentTurn]) -> list[FailureEvent]:
    """Detect context loss by checking if earlier information is forgotten."""
    events: list[FailureEvent] = []
    if len(turns) < 3:
        return events

    # Check if agent references information from 3+ turns ago
    early_info: dict[str, int] = {}
    for turn in turns[:3]:
        if turn.user_input:
            keywords = set(turn.user_input.lower().split()) - {
                "the", "a", "an", "is", "was", "are", "were", "of", "in",
                "to", "for", "with", "and", "or", "can", "you", "i", "me",
                "what", "how", "do", "does", "please", "would", "could",
            }
            for kw in keywords:
                if len(kw) > 3:
                    early_info[kw] = turn.turn_index

    # Check later turns for recall of early info
    for turn in turns[4:]:
        if turn.agent_response and early_info:
            recall_count = sum(
                1 for kw in early_info
                if kw.lower() in turn.agent_response.lower()
            )
            recall_needed = sum(
                1 for kw in early_info
                if kw.lower() in (turn.user_input or "").lower()
            )
            if recall_needed > 2 and recall_count == 0:
                events.append(_event(
                    FailureMode.CONTEXT_LOSS,
                    confidence=0.5,
                    turn_index=turn.turn_index,
                    evidence=f"Agent appears to have lost early-turn context",
                    trace_segment="agent_response",
                ))
                return events

    return events


def detect_prompt_injection_followed(turns: list[AgentTurn]) -> list[FailureEvent]:
    """Detect if agent followed prompt injection by checking response divergence."""
    events: list[FailureEvent] = []

    for turn in turns:
        if not turn.user_input or not turn.agent_response:
            continue

        user_text = turn.user_input.lower()
        for pattern in INJECTION_FOLLOW_PATTERNS:
            if re.search(pattern, user_text, re.IGNORECASE):
                # User input contained injection signal — check if agent complied
                compliance_phrases = [
                    "i will", "i'll", "sure", "certainly", "of course",
                    "absolutely", "yes", "understood", "got it",
                ]
                response = turn.agent_response.lower()[:200]
                if any(phrase in response for phrase in compliance_phrases):
                    events.append(_event(
                        FailureMode.PROMPT_INJECTION_FOLLOWED,
                        confidence=0.7,
                        turn_index=turn.turn_index,
                        evidence=f"Agent may have complied with injected instruction",
                        trace_segment="agent_response",
                    ))
                    return events

    return events


def detect_premature_termination(turns: list[AgentTurn]) -> list[FailureEvent]:
    """Detect if agent stopped before completing the task."""
    events: list[FailureEvent] = []
    if not turns:
        return events

    last = turns[-1]
    if last.agent_response:
        response = last.agent_response.lower()
        incomplete_signals = [
            "let me know if", "would you like me to", "should i continue",
            "shall i proceed", "do you want me to", "more details",
            "anything else", "what else", "happy to help further",
        ]
        for signal in incomplete_signals:
            if signal in response:
                events.append(_event(
                    FailureMode.PREMATURE_TERMINATION,
                    confidence=0.4,  # Low confidence — could be polite ending
                    turn_index=last.turn_index,
                    evidence=f"Agent ended with open signal: '{signal}'",
                    trace_segment="agent_response",
                ))
                return events

    return events


def detect_goal_drift(turns: list[AgentTurn]) -> list[FailureEvent]:
    """Detect goal drift by comparing first-turn intent with final-turn actions."""
    events: list[FailureEvent] = []
    if len(turns) < 3:
        return events

    # Extract original intent keywords
    original_intent = turns[0].user_input or ""
    intent_keywords = {w.lower() for w in original_intent.split() if len(w) > 3} - {
        "what", "that", "this", "with", "from", "have", "been", "were",
        "they", "them", "their", "your", "about", "would", "could",
        "please", "which", "there",
    }

    if not intent_keywords:
        return events

    # Check final turns for topic drift
    for turn in turns[-2:]:
        if turn.agent_response:
            resp_keywords = {w.lower() for w in turn.agent_response.split() if len(w) > 3}
            if intent_keywords:
                overlap = len(intent_keywords & resp_keywords) / len(intent_keywords)
                if overlap < 0.15:
                    events.append(_event(
                        FailureMode.GOAL_DRIFT,
                        confidence=0.5,
                        turn_index=turn.turn_index,
                        evidence=f"Goal drift: only {overlap:.0%} keyword overlap with original intent",
                        trace_segment="agent_response",
                        overlap_ratio=str(round(overlap, 2)),
                    ))
                    return events

    return events


def detect_latency_issues(turns: list[AgentTurn], latency_threshold_ms: float = 30_000) -> list[FailureEvent]:
    """Detect excessive latency in agent responses."""
    events: list[FailureEvent] = []

    for turn in turns:
        if turn.llm_latency_ms and turn.llm_latency_ms > latency_threshold_ms:
            events.append(_event(
                FailureMode.EXCESSIVE_LATENCY,
                confidence=0.7 if turn.llm_latency_ms > latency_threshold_ms * 2 else 0.5,
                turn_index=turn.turn_index,
                evidence=f"LLM latency {turn.llm_latency_ms:.0f}ms exceeds threshold {latency_threshold_ms}ms",
                trace_segment="llm_latency",
                latency_ms=str(turn.llm_latency_ms),
                threshold_ms=str(latency_threshold_ms),
            ))

    return events


# ── Main Detector ───────────────────────────────────────────────────────

def detect_failures(
    turns: list[AgentTurn],
    *,
    context_limit: int = 128_000,
    latency_threshold_ms: float = 30_000,
    ground_truth: str | None = None,
) -> list[FailureEvent]:
    """Run all detectors on a sequence of agent turns.

    This is the main entry point for failure detection. It runs each
    specialized detector and aggregates all FailureEvents.

    Args:
        turns: Parsed agent turns from a trace.
        context_limit: Model context window size in tokens.
        latency_threshold_ms: Maximum acceptable LLM latency.
        ground_truth: Optional ground truth for hallucination checking.

    Returns:
        List of all detected FailureEvents, ordered by turn_index then severity.
    """
    all_events: list[FailureEvent] = []

    # Per-turn detectors
    for turn in turns:
        all_events.extend(detect_hallucination(turn, ground_truth=ground_truth))

    for i, _ in enumerate(turns):
        all_events.extend(detect_tool_misuse(turns, i))

    # Sequence detectors
    actions = []
    for turn in turns:
        if turn.tool_calls:
            actions.extend(tc.name for tc in turn.tool_calls)
        else:
            actions.append("respond")
    all_events.extend(detect_loop(actions))

    all_events.extend(detect_context_overflow(turns, context_limit))
    all_events.extend(detect_context_loss(turns))
    all_events.extend(detect_prompt_injection_followed(turns))
    all_events.extend(detect_premature_termination(turns))
    all_events.extend(detect_goal_drift(turns))
    all_events.extend(detect_latency_issues(turns, latency_threshold_ms))

    # Sort by turn_index then severity (CRITICAL first)
    severity_order = {
        FailureSeverity.CRITICAL: 0,
        FailureSeverity.HIGH: 1,
        FailureSeverity.MEDIUM: 2,
        FailureSeverity.LOW: 3,
        FailureSeverity.INFO: 4,
    }
    all_events.sort(key=lambda e: (e.turn_index, severity_order[e.severity]))

    return all_events
