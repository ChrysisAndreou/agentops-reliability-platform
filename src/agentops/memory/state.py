"""
Memory evaluation state models — typed containers for multi-turn
conversation memory testing.

Models three memory types:
- Episodic: remembering specific events from earlier turns
- Semantic: recalling facts presented during the conversation
- Working: maintaining task context across execution steps

Includes configurable memory degradation profiles for realistic
simulation of production agent memory behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Types of memory tested in the evaluation framework."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"


@dataclass
class MemoryEntry:
    """A single memory — a fact or event injected into the agent's context.

    During simulation, these are injected at specific conversation turns
    and recalled later to measure memory quality.
    """
    id: str
    content: str
    memory_type: MemoryType
    injected_at_turn: int  # conversation turn when this memory was introduced
    importance: float = 1.0  # 0.0–1.0, affects simulated recall probability
    related_to: list[str] = field(default_factory=list)  # IDs of related memories
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStore:
    """Accumulates memories injected during a multi-turn conversation.

    Used by the simulated agent to track what it "should" remember,
    enabling precise recall measurement.
    """
    entries: list[MemoryEntry] = field(default_factory=list)

    def inject(self, entry: MemoryEntry) -> None:
        """Add a memory to the store at a specific conversation turn."""
        self.entries.append(entry)

    def get_by_turn(self, turn: int) -> list[MemoryEntry]:
        """Get all memories injected at or before a given turn."""
        return [e for e in self.entries if e.injected_at_turn <= turn]

    def get_by_type(self, memory_type: MemoryType) -> list[MemoryEntry]:
        """Get all memories of a specific type."""
        return [e for e in self.entries if e.memory_type == memory_type]

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Look up a memory by ID."""
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.memory_type.value] = counts.get(e.memory_type.value, 0) + 1
        return counts


@dataclass
class MemoryContext:
    """The active memory context for a single conversation turn.

    Simulates what the agent "has in mind" during a given turn,
    with configurable decay and confusion.
    """
    turn: int
    active_memories: list[str]  # IDs of memories currently accessible
    recalled_correctly: list[str] = field(default_factory=list)  # correctly recalled
    recalled_incorrectly: list[str] = field(default_factory=list)  # confused/hallucinated
    forgotten: list[str] = field(default_factory=list)  # should recall but didn't
    hallucinated: list[str] = field(default_factory=list)  # fabricated memories


@dataclass
class MemoryRecallResult:
    """Result of a single memory recall test within a conversation."""
    task_id: str
    turn: int
    target_memory_id: str
    memory_type: MemoryType
    target_content: str
    recalled_content: str  # what the agent actually produced
    correct: bool
    partial: bool  # partially correct (some details right, some wrong)
    confidence: float  # 0.0–1.0, agent's self-reported confidence
    latency_ms: float
    error: str | None = None


@dataclass
class MemoryMetrics:
    """Aggregate metrics for a memory evaluation run."""
    total_tests: int = 0
    correct_recalls: int = 0
    partial_recalls: int = 0
    incorrect_recalls: int = 0
    hallucinations: int = 0  # fabricated memories

    # Recall precision: fraction of recalled memories that are correct
    recall_precision: float = 0.0
    # Recall rate: fraction of target memories successfully recalled
    recall_rate: float = 0.0
    # Memory F1
    memory_f1: float = 0.0

    # Per-type breakdown
    episodic_precision: float = 0.0
    semantic_precision: float = 0.0
    working_precision: float = 0.0

    # Degradation
    early_turn_accuracy: float = 0.0  # turns 0–5
    late_turn_accuracy: float = 0.0  # turns 15+
    degradation_rate: float = 0.0  # accuracy drop per turn

    # Latency
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0

    @property
    def composite_score(self) -> float:
        """Weighted composite: F1 + penalize hallucinations."""
        return max(0.0, self.memory_f1 - (0.2 * self.hallucinations / max(self.total_tests, 1)))


@dataclass
class MemoryReport:
    """Complete memory evaluation report for presentation and export."""
    title: str
    profile_name: str
    metrics: MemoryMetrics
    results: list[MemoryRecallResult]
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "profile_name": self.profile_name,
            "metrics": {
                "total_tests": self.metrics.total_tests,
                "correct_recalls": self.metrics.correct_recalls,
                "partial_recalls": self.metrics.partial_recalls,
                "incorrect_recalls": self.metrics.incorrect_recalls,
                "hallucinations": self.metrics.hallucinations,
                "recall_precision": self.metrics.recall_precision,
                "recall_rate": self.metrics.recall_rate,
                "memory_f1": self.metrics.memory_f1,
                "episodic_precision": self.metrics.episodic_precision,
                "semantic_precision": self.metrics.semantic_precision,
                "working_precision": self.metrics.working_precision,
                "early_turn_accuracy": self.metrics.early_turn_accuracy,
                "late_turn_accuracy": self.metrics.late_turn_accuracy,
                "degradation_rate": self.metrics.degradation_rate,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "avg_confidence": self.metrics.avg_confidence,
                "composite_score": self.metrics.composite_score,
            },
            "results": [r.__dict__ for r in self.results],
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        m = self.metrics
        lines = [
            f"# {self.title}",
            "",
            f"**Profile:** {self.profile_name}",
            "",
            "## Memory Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {m.total_tests} |",
            f"| Correct Recalls | {m.correct_recalls} |",
            f"| Partial Recalls | {m.partial_recalls} |",
            f"| Incorrect Recalls | {m.incorrect_recalls} |",
            f"| Hallucinations | {m.hallucinations} |",
            f"| Recall Precision | {m.recall_precision:.3f} |",
            f"| Recall Rate | {m.recall_rate:.3f} |",
            f"| Memory F1 | {m.memory_f1:.3f} |",
            f"| Composite Score | {m.composite_score:.3f} |",
            "",
            "## Per-Type Breakdown",
            "",
            "| Memory Type | Precision |",
            "|-------------|-----------|",
            f"| Episodic | {m.episodic_precision:.3f} |",
            f"| Semantic | {m.semantic_precision:.3f} |",
            f"| Working | {m.working_precision:.3f} |",
            "",
            "## Degradation Analysis",
            "",
            "| Phase | Accuracy |",
            "|-------|----------|",
            f"| Early Turns (0-5) | {m.early_turn_accuracy:.3f} |",
            f"| Late Turns (15+) | {m.late_turn_accuracy:.3f} |",
            f"| Degradation Rate | {m.degradation_rate:.3f}/turn |",
            "",
            f"**Avg Latency:** {m.avg_latency_ms:.1f}ms | **Avg Confidence:** {m.avg_confidence:.3f}",
            "",
        ]
        if self.summary:
            lines.extend(["## Summary", "", self.summary, ""])
        return "\n".join(lines)


# ── Memory degradation profiles ──────────────────────────────────────

@dataclass
class MemoryProfile:
    """Controls how the simulated agent's memory degrades over turns.

    Higher values = better memory. All ranges 0.0–1.0.
    """
    name: str
    description: str

    # Base recall probability (turn 0)
    base_recall_prob: float = 0.95

    # Per-turn decay factor (multiplied each turn)
    decay_per_turn: float = 0.02

    # Probability of confusing two similar memories
    confusion_rate: float = 0.03

    # Probability of fabricating a memory (hallucination)
    hallucination_rate: float = 0.02

    # How much importance weights recall (0 = ignore importance)
    importance_weight: float = 0.3

    # Minimum recall probability (floor)
    min_recall_prob: float = 0.20

    # Latency parameters
    base_recall_latency_ms: float = 500
    latency_per_turn_ms: float = 50


PERFECT_MEMORY = MemoryProfile(
    name="perfect",
    description="Ideal memory: perfect recall, zero decay, no hallucinations",
    base_recall_prob=1.0,
    decay_per_turn=0.0,
    confusion_rate=0.0,
    hallucination_rate=0.0,
    importance_weight=0.0,
    min_recall_prob=1.0,
    base_recall_latency_ms=300,
    latency_per_turn_ms=5,
)

PRODUCTION_MEMORY = MemoryProfile(
    name="production",
    description="Realistic production agent: strong recall with gradual decay",
    base_recall_prob=0.95,
    decay_per_turn=0.015,
    confusion_rate=0.03,
    hallucination_rate=0.02,
    importance_weight=0.25,
    min_recall_prob=0.30,
    base_recall_latency_ms=500,
    latency_per_turn_ms=30,
)

DEVELOPMENT_MEMORY = MemoryProfile(
    name="development",
    description="Agent under development: moderate recall with noticeable decay",
    base_recall_prob=0.85,
    decay_per_turn=0.03,
    confusion_rate=0.06,
    hallucination_rate=0.05,
    importance_weight=0.20,
    min_recall_prob=0.15,
    base_recall_latency_ms=700,
    latency_per_turn_ms=50,
)

DEGRADED_MEMORY = MemoryProfile(
    name="degraded",
    description="Context-overloaded agent: poor recall, frequent confusion and hallucinations",
    base_recall_prob=0.65,
    decay_per_turn=0.05,
    confusion_rate=0.12,
    hallucination_rate=0.10,
    importance_weight=0.10,
    min_recall_prob=0.05,
    base_recall_latency_ms=1000,
    latency_per_turn_ms=80,
)

MEMORY_PROFILES = [PERFECT_MEMORY, PRODUCTION_MEMORY, DEVELOPMENT_MEMORY, DEGRADED_MEMORY]


def get_memory_profile(name: str) -> MemoryProfile | None:
    """Look up a memory profile by name."""
    for p in MEMORY_PROFILES:
        if p.name == name:
            return p
    return None
