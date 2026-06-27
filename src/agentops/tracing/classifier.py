"""
Failure pattern classification for agent runs.

Analyses trace data to identify common failure modes:
hallucination, retrieval gaps, tool errors, verification failures,
timeouts, etc. Produces a failure taxonomy report useful for
reliability improvement and evaluation reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FailurePattern:
    """A classified failure pattern found in traces."""
    pattern_name: str
    description: str
    count: int
    example_task_ids: list[str]
    severity: str  # "critical", "high", "medium", "low"


class FailureClassifier:
    """Classifies agent run failures into patterns.

    Usage:
        classifier = FailureClassifier()
        patterns = classifier.classify(traces)
        for p in patterns:
            print(f"{p.pattern_name}: {p.count} occurrences")
    """

    PATTERNS = [
        {
            "name": "hallucination",
            "description": "Agent produced claims not grounded in retrieved evidence",
            "check": lambda r: (
                getattr(r, "ungrounded_claims", None)
                and len(r.ungrounded_claims) > 0
            ),
            "severity": "critical",
        },
        {
            "name": "verification_failure",
            "description": "Verification step explicitly rejected the output",
            "check": lambda r: (
                not getattr(r, "verification_passed", True)
                and not getattr(r, "error", None)
            ),
            "severity": "high",
        },
        {
            "name": "retrieval_gap",
            "description": "No relevant chunks retrieved for the query",
            "check": lambda r: getattr(r, "retrieved_chunks_count", 1) == 0,
            "severity": "high",
        },
        {
            "name": "tool_error",
            "description": "A tool call failed during execution",
            "check": lambda r: (
                getattr(r, "reliability_trace", None)
                and any(
                    step.get("step_type") == "tool_call" and step.get("error")
                    for step in r.reliability_trace
                )
            ),
            "severity": "medium",
        },
        {
            "name": "timeout_or_abort",
            "description": "Agent run didn't complete successfully due to error",
            "check": lambda r: (
                not getattr(r, "success", True) and getattr(r, "error", None)
            ),
            "severity": "high",
        },
        {
            "name": "planning_failure",
            "description": "Agent produced an empty or nonsensical plan",
            "check": lambda r: (
                getattr(r, "plan", None) is not None
                and (len(getattr(r, "plan", [])) == 0
                     or all(len(str(s).strip()) < 5 for s in getattr(r, "plan", [])))
            ),
            "severity": "medium",
        },
        {
            "name": "no_citations",
            "description": "Response provided but no evidence citations used",
            "check": lambda r: (
                getattr(r, "final_answer", "")
                and len(getattr(r, "citations_used", [])) == 0
            ),
            "severity": "medium",
        },
        {
            "name": "low_retrieval_quality",
            "description": "Chunks retrieved but relevance was poor",
            "check": lambda r: (
                getattr(r, "retrieved_chunks_count", 0) > 0
                and not getattr(r, "verification_passed", True)
                and getattr(r, "verification_notes", "")
                and "ungrounded" in getattr(r, "verification_notes", "").lower()
            ),
            "severity": "medium",
        },
    ]

    def classify(self, traces: list[Any]) -> list[FailurePattern]:
        """Classify a list of trace objects into failure patterns.

        Each trace is checked against all known patterns. Multiple
        patterns may match a single trace.
        """
        patterns = []
        for pattern_def in self.PATTERNS:
            matches = []
            for trace in traces:
                try:
                    if pattern_def["check"](trace):
                        task_id = getattr(trace, "task_id", "unknown")
                        matches.append(str(task_id))
                except Exception:
                    continue

            if matches:
                patterns.append(FailurePattern(
                    pattern_name=pattern_def["name"],
                    description=pattern_def["description"],
                    count=len(matches),
                    example_task_ids=matches[:5],
                    severity=pattern_def["severity"],
                ))

        patterns.sort(key=lambda p: p.count, reverse=True)
        return patterns

    def summary_report(self, traces: list[Any], total_runs: int | None = None) -> str:
        """Generate a human-readable failure analysis report."""
        if total_runs is None:
            total_runs = len(traces)

        patterns = self.classify(traces)
        failed = sum(1 for t in traces if not getattr(t, "success", True)
                     or not getattr(t, "verification_passed", True))

        lines = []
        lines.append("=" * 65)
        lines.append("FAILURE ANALYSIS REPORT")
        lines.append("=" * 65)
        lines.append(f"Total runs analysed: {total_runs}")
        lines.append(f"Runs with failures: {failed}")
        lines.append(f"Failure rate: {failed/max(total_runs,1)*100:.1f}%")
        lines.append("")

        if not patterns:
            lines.append("No failure patterns detected.")
            return "\n".join(lines)

        lines.append(f"Detected {len(patterns)} failure pattern(s):")
        lines.append("")

        for pattern in patterns:
            severity_marker = {
                "critical": "!!!",
                "high": "!! ",
                "medium": "!  ",
                "low": "   ",
            }.get(pattern.severity, "   ")

            lines.append(f"  [{severity_marker}] {pattern.pattern_name.upper()} ({pattern.count} occurrences)")
            lines.append(f"       {pattern.description}")
            if pattern.example_task_ids:
                lines.append(f"       Examples: {', '.join(pattern.example_task_ids[:3])}")
            lines.append("")

        lines.append("-" * 65)
        lines.append("Severity legend: !!! = critical, !! = high, ! = medium")
        lines.append("=" * 65)

        return "\n".join(lines)

    def to_dict(self, traces: list[Any]) -> dict[str, Any]:
        """Return failure analysis as a structured dictionary."""
        patterns = self.classify(traces)
        return {
            "total_traces": len(traces),
            "failure_patterns": [
                {
                    "name": p.pattern_name,
                    "description": p.description,
                    "count": p.count,
                    "severity": p.severity,
                    "examples": p.example_task_ids[:3],
                }
                for p in patterns
            ],
        }
