"""
Data models for streaming verification.

Streaming verification operates at the claim level within a stream of LLM
output. It extracts claims from partial text, verifies each against evidence,
and can abort the stream when hallucination exceeds configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class VerificationStrategy(str, Enum):
    """How streaming verification evaluates claims."""

    STRICT = "strict"
    """Every claim must be grounded. Abort on first ungrounded claim."""

    THRESHOLD = "threshold"
    """Abort when ungrounded rate exceeds abort_threshold."""

    LENIENT = "lenient"
    """Only flag ungrounded claims; never abort."""

    ACCUMULATING = "accumulating"
    """Wait for more context before deciding on borderline claims."""


class AbortReason(str, Enum):
    """Why the stream was aborted."""

    UNGROUNDED_CLAIM = "ungrounded_claim"
    """A single claim could not be verified against evidence."""

    THRESHOLD_EXCEEDED = "threshold_exceeded"
    """The ungrounded claim rate exceeded the configured threshold."""

    CONTRADICTION = "contradiction"
    """A claim directly contradicted the evidence."""

    HALLUCINATED_ENTITY = "hallucinated_entity"
    """An entity was referenced that doesn't exist in evidence."""

    TIMEOUT = "timeout"
    """Verification took too long."""

    MANUAL = "manual"
    """Externally triggered abort."""

    NONE = "none"
    """No abort — stream completed normally."""


@dataclass
class StreamingConfig:
    """Configuration for streaming verification.

    Attributes:
        strategy: How to evaluate claims (strict, threshold, lenient, accumulating).
        abort_threshold: Ungrounded rate at which to abort (0.0–1.0). Used by
            THRESHOLD and ACCUMULATING strategies.
        chunk_overlap: Characters of overlap between chunks for context continuity.
        min_claim_length: Minimum characters for text to be considered a claim.
        max_buffer_chars: Maximum characters to buffer before forcing extraction.
        verification_timeout_ms: Max milliseconds per claim verification.
        abort_on_contradiction: Whether contradiction triggers immediate abort.
        track_entities: Whether to track and verify named entities.
        evidence_window: Max evidence chunks to consider per verification.
    """

    strategy: VerificationStrategy = VerificationStrategy.THRESHOLD
    abort_threshold: float = 0.30
    chunk_overlap: int = 50
    min_claim_length: int = 20
    max_buffer_chars: int = 500
    verification_timeout_ms: int = 2000
    abort_on_contradiction: bool = True
    track_entities: bool = True
    evidence_window: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.abort_threshold <= 1.0:
            raise ValueError(f"abort_threshold must be 0–1, got {self.abort_threshold}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.min_claim_length < 5:
            raise ValueError(f"min_claim_length must be >= 5, got {self.min_claim_length}")


@dataclass
class StreamChunk:
    """A chunk of streaming output from the LLM.

    Attributes:
        index: Zero-based chunk index in the stream.
        text: The raw text chunk from the LLM.
        timestamp_ms: When this chunk was received (relative to stream start).
        is_final: Whether this is the last chunk in the stream.
    """

    index: int
    text: str
    timestamp_ms: float = 0.0
    is_final: bool = False


@dataclass
class StreamingClaim:
    """A factual claim extracted from streaming text.

    Attributes:
        text: The claim text.
        chunk_indices: Which chunks contributed to this claim.
        confidence: How confident the extractor is that this is a factual claim.
        entities: Named entities found in the claim.
        is_partial: Whether this claim might span into future chunks.
        start_char: Character offset in the accumulated text.
        end_char: Character offset in the accumulated text.
    """

    text: str
    chunk_indices: list[int] = field(default_factory=list)
    confidence: float = 1.0
    entities: list[str] = field(default_factory=list)
    is_partial: bool = False
    start_char: int = 0
    end_char: int = 0


@dataclass
class StreamingVerificationResult:
    """Result of verifying a single claim during streaming.

    Attributes:
        claim: The claim that was verified.
        grounded: Whether the claim is supported by evidence.
        evidence_chunks: Evidence that supports the claim.
        contradictory_evidence: Evidence that contradicts the claim.
        score: How well the claim matches evidence (0.0–1.0).
        latency_ms: Time spent verifying this claim.
        error: Any error during verification.
    """

    claim: StreamingClaim
    grounded: bool = False
    evidence_chunks: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    score: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class StreamingMetrics:
    """Aggregate metrics for a streaming verification run.

    Attributes:
        run_id: Unique identifier for this run.
        total_chunks: Total chunks received.
        total_claims: Total claims extracted and verified.
        grounded_claims: Claims that passed verification.
        ungrounded_claims: Claims that failed verification.
        contradicted_claims: Claims that contradicted evidence.
        hallucinated_entities: Entities found that aren't in evidence.
        aborted: Whether the stream was aborted.
        abort_reason: Why the stream was aborted.
        abort_at_chunk: Which chunk triggered the abort.
        groundedness: grounded_claims / total_claims (1.0 if no claims).
        time_to_first_abort_ms: Time from stream start to abort.
        total_latency_ms: Total time spent in verification.
        avg_verification_latency_ms: Average per-claim verification time.
        evidence_hits: Number of times evidence was found for a claim.
        evidence_misses: Number of times evidence was not found.
    """

    run_id: str = ""
    total_chunks: int = 0
    total_claims: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    contradicted_claims: int = 0
    hallucinated_entities: int = 0
    aborted: bool = False
    abort_reason: AbortReason = AbortReason.NONE
    abort_at_chunk: int = -1
    groundedness: float = 1.0
    time_to_first_abort_ms: float = 0.0
    total_latency_ms: float = 0.0
    avg_verification_latency_ms: float = 0.0
    evidence_hits: int = 0
    evidence_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_chunks": self.total_chunks,
            "total_claims": self.total_claims,
            "grounded_claims": self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "contradicted_claims": self.contradicted_claims,
            "hallucinated_entities": self.hallucinated_entities,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason.value,
            "abort_at_chunk": self.abort_at_chunk,
            "groundedness": round(self.groundedness, 4),
            "time_to_first_abort_ms": round(self.time_to_first_abort_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_verification_latency_ms": round(self.avg_verification_latency_ms, 2),
            "evidence_hits": self.evidence_hits,
            "evidence_misses": self.evidence_misses,
        }


@dataclass
class StreamingRun:
    """Full state for a streaming verification run.

    Tracks accumulated text, claims, verification results, and abort state.
    """

    run_id: str
    task: str
    config: StreamingConfig = field(default_factory=StreamingConfig)
    evidence_store: dict[str, list[str]] = field(default_factory=dict)

    # Accumulated state
    accumulated_text: str = ""
    chunks: list[StreamChunk] = field(default_factory=list)
    claims: list[StreamingClaim] = field(default_factory=list)
    results: list[StreamingVerificationResult] = field(default_factory=list)

    # Abort state
    aborted: bool = False
    abort_reason: AbortReason = AbortReason.NONE
    abort_at_chunk: int = -1

    def ungrounded_rate(self) -> float:
        """Current rate of ungrounded claims."""
        if not self.results:
            return 0.0
        failed = sum(1 for r in self.results if not r.grounded)
        return failed / len(self.results)

    def should_abort(self) -> tuple[bool, AbortReason | None]:
        """Check if the stream should be aborted based on strategy and state."""
        if self.aborted:
            return True, self.abort_reason

        strategy = self.config.strategy

        if strategy == VerificationStrategy.STRICT:
            for r in self.results:
                if not r.grounded:
                    if r.contradictory_evidence and self.config.abort_on_contradiction:
                        return True, AbortReason.CONTRADICTION
                    return True, AbortReason.UNGROUNDED_CLAIM

        elif strategy in (VerificationStrategy.THRESHOLD, VerificationStrategy.ACCUMULATING):
            if self.results and self.ungrounded_rate() > self.config.abort_threshold:
                # Check for contradiction
                for r in self.results:
                    if r.contradictory_evidence and self.config.abort_on_contradiction:
                        return True, AbortReason.CONTRADICTION
                return True, AbortReason.THRESHOLD_EXCEEDED

        # LENIENT never aborts; ACCUMULATING only aborts on threshold
        return False, None
