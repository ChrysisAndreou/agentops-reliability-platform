"""
Reliability evaluation metrics for agent outputs.

Measures groundedness, citation accuracy, verification outcomes,
tool correctness, and run-level quality indicators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def groundedness(grounded_claims: list[str], ungrounded_claims: list[str]) -> float:
    """Fraction of claims that are grounded in retrieved evidence."""
    total = len(grounded_claims) + len(ungrounded_claims)
    if total == 0:
        return 1.0
    return len(grounded_claims) / total


def citation_precision(citations_used: list[str], retrieved_chunks_count: int) -> float:
    """Citations as a fraction of available chunks (correctness proxy)."""
    if retrieved_chunks_count == 0:
        return 0.0 if citations_used else 1.0
    return min(len(citations_used) / retrieved_chunks_count, 1.0)


def verification_pass_rate(passed: bool) -> float:
    """Binary verification outcome."""
    return 1.0 if passed else 0.0


def tool_success_rate(tool_results: list[dict[str, Any]]) -> float:
    """Fraction of tool calls that succeeded."""
    if not tool_results:
        return 1.0
    successes = sum(1 for t in tool_results if not t.get("error"))
    return successes / len(tool_results)


def answer_completeness(final_answer: str, key_terms: list[str]) -> float:
    """Fraction of expected key terms present in the answer."""
    if not key_terms:
        return 1.0
    answer_lower = final_answer.lower()
    found = sum(1 for term in key_terms if term.lower() in answer_lower)
    return found / len(key_terms)


def latency_score_ms(latency_ms: float, max_acceptable_ms: float = 120_000) -> float:
    """Latency score: 1.0 at 0ms, decays linearly to 0 at max."""
    return max(0.0, 1.0 - latency_ms / max_acceptable_ms)


@dataclass
class ReliabilityMetrics:
    """Aggregate reliability metrics for a single agent run."""

    run_id: str
    task_id: str
    
    # Core reliability
    groundedness: float = 0.0
    citation_precision: float = 0.0
    verification_passed: bool = False
    verification_pass_rate: float = 0.0
    
    # Tool quality
    tool_success_rate: float = 1.0
    tool_calls_count: int = 0
    
    # Answer quality
    answer_completeness: float = 0.0
    key_terms_found: int = 0
    key_terms_total: int = 0
    
    # Performance
    latency_ms: float = 0.0
    latency_score: float = 0.0
    
    # Composite
    composite: float = 0.0
    
    # Metadata
    grounded_claims_count: int = 0
    ungrounded_claims_count: int = 0
    citations_used_count: int = 0
    retrieved_chunks_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "groundedness": round(self.groundedness, 3),
            "citation_precision": round(self.citation_precision, 3),
            "verification_passed": self.verification_passed,
            "tool_success_rate": round(self.tool_success_rate, 3),
            "tool_calls_count": self.tool_calls_count,
            "answer_completeness": round(self.answer_completeness, 3),
            "latency_ms": round(self.latency_ms, 1),
            "latency_score": round(self.latency_score, 3),
            "composite": round(self.composite, 3),
        }


def compute_metrics(result: Any, key_terms: list[str] | None = None) -> ReliabilityMetrics:
    """Compute reliability metrics from an AgentRunResult."""
    grounded = result.grounded_claims if hasattr(result, "grounded_claims") else []
    ungrounded = result.ungrounded_claims if hasattr(result, "ungrounded_claims") else []
    citations = result.citations_used if hasattr(result, "citations_used") else []
    retrieved = result.retrieved_chunks_count if hasattr(result, "retrieved_chunks_count") else 0
    latency = result.total_latency_ms if hasattr(result, "total_latency_ms") else 0
    verified = result.verification_passed if hasattr(result, "verification_passed") else False
    tool_count = result.tool_calls_count if hasattr(result, "tool_calls_count") else 0
    answer = result.final_answer if hasattr(result, "final_answer") else ""
    task_id = result.task_id if hasattr(result, "task_id") else ""

    g = groundedness(grounded, ungrounded)
    cp = citation_precision(citations, retrieved)
    vp = verification_pass_rate(verified)
    ts = 1.0  # default — tool success tracked separately
    ls = latency_score_ms(latency)
    ac = answer_completeness(answer, key_terms or [])
    kt_found = sum(1 for t in (key_terms or []) if t.lower() in answer.lower())

    # Composite: heavy weight on groundedness and verification
    composite = (
        0.30 * g
        + 0.20 * cp
        + 0.25 * vp
        + 0.10 * ts
        + 0.10 * ls
        + 0.05 * ac
    )

    return ReliabilityMetrics(
        run_id=task_id,
        task_id=task_id,
        groundedness=g,
        citation_precision=cp,
        verification_passed=verified,
        verification_pass_rate=vp,
        tool_success_rate=ts,
        tool_calls_count=tool_count,
        answer_completeness=ac,
        key_terms_found=kt_found,
        key_terms_total=len(key_terms or []),
        latency_ms=latency,
        latency_score=ls,
        composite=composite,
        grounded_claims_count=len(grounded),
        ungrounded_claims_count=len(ungrounded),
        citations_used_count=len(citations),
        retrieved_chunks_count=retrieved,
    )
