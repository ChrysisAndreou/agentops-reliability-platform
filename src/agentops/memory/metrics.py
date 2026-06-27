"""
Memory evaluation metrics — systematic measurement of agent memory quality.

Computes precision, recall, F1, degradation curves, and per-type breakdowns
from a set of MemoryRecallResults. Generates structured MemoryMetrics and
MemoryReport objects for integration with the broader evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import (
    MemoryEntry,
    MemoryMetrics,
    MemoryRecallResult,
    MemoryReport,
    MemoryType,
)


@dataclass
class MemoryBenchmarkTask:
    """A single memory evaluation task in the benchmark suite."""
    id: str
    name: str
    description: str
    turns: list[dict[str, Any]]  # conversation turns with inject/recall


class MemoryEvaluator:
    """Computes memory metrics from recall results and generates reports.

    Usage:
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        report = evaluator.generate_report(metrics, results, "production")
        print(report.to_markdown())
    """

    def compute_metrics(self, results: list[MemoryRecallResult]) -> MemoryMetrics:
        """Compute aggregate memory metrics from recall results."""
        if not results:
            return MemoryMetrics()

        total = len(results)
        correct = sum(1 for r in results if r.correct)
        partial = sum(1 for r in results if r.partial)
        incorrect = sum(1 for r in results if not r.correct and not r.partial)
        hallucinations = sum(
            1 for r in results
            if not r.correct and not r.partial and len(r.recalled_content) > 0
        )

        # Precision: fraction of recall ATTEMPTS that were correct
        attempted = sum(1 for r in results if len(r.recalled_content) > 0)
        recall_precision = correct / max(attempted, 1)

        # Recall rate: fraction of TARGETS successfully recalled
        recall_rate = (correct + 0.5 * partial) / max(total, 1)

        # F1
        memory_f1 = (
            2 * recall_precision * recall_rate / max(recall_precision + recall_rate, 0.001)
        )

        # Per-type breakdown (compute directly to avoid recursion)
        def _type_precision(mt: MemoryType) -> float:
            subset = [r for r in results if r.memory_type == mt]
            if not subset:
                return 0.0
            c = sum(1 for r in subset if r.correct)
            a = sum(1 for r in subset if len(r.recalled_content) > 0)
            return c / max(a, 1)

        episodic_precision = _type_precision(MemoryType.EPISODIC)
        semantic_precision = _type_precision(MemoryType.SEMANTIC)
        working_precision = _type_precision(MemoryType.WORKING)

        # Degradation analysis
        early = [r for r in results if r.turn <= 5]
        late = [r for r in results if r.turn >= 15]
        early_acc = sum(1 for r in early if r.correct) / max(len(early), 1)
        late_acc = sum(1 for r in late if r.correct) / max(len(late), 1)
        turn_range = max(1, max(r.turn for r in results) - min(r.turn for r in results))
        degradation_rate = (early_acc - late_acc) / max(turn_range, 1) if late else 0.0

        # Latency and confidence
        avg_latency = sum(r.latency_ms for r in results) / total
        avg_confidence = sum(r.confidence for r in results) / total

        m = MemoryMetrics(
            total_tests=total,
            correct_recalls=correct,
            partial_recalls=partial,
            incorrect_recalls=incorrect,
            hallucinations=hallucinations,
            recall_precision=recall_precision,
            recall_rate=recall_rate,
            memory_f1=memory_f1,
            episodic_precision=episodic_precision,
            semantic_precision=semantic_precision,
            working_precision=working_precision,
            early_turn_accuracy=early_acc,
            late_turn_accuracy=late_acc,
            degradation_rate=degradation_rate,
            avg_latency_ms=avg_latency,
            avg_confidence=avg_confidence,
        )
        return m

    def generate_report(
        self,
        metrics: MemoryMetrics,
        results: list[MemoryRecallResult],
        profile_name: str,
        title: str = "Agent Memory Evaluation Report",
    ) -> MemoryReport:
        """Generate a structured report from metrics and results."""
        total = metrics.total_tests
        if total == 0:
            return MemoryReport(
                title=title,
                profile_name=profile_name,
                metrics=metrics,
                results=results,
                summary="No recall tests performed.",
            )

        # Build summary
        if metrics.memory_f1 >= 0.90:
            quality = "excellent"
        elif metrics.memory_f1 >= 0.75:
            quality = "good"
        elif metrics.memory_f1 >= 0.50:
            quality = "moderate"
        else:
            quality = "poor"

        summary = (
            f"The agent demonstrates {quality} memory performance "
            f"(F1={metrics.memory_f1:.3f}) across {total} recall tests. "
            f"{metrics.correct_recalls}/{total} correct, "
            f"{metrics.partial_recalls} partial, "
            f"{metrics.hallucinations} hallucinations. "
        )

        if metrics.degradation_rate > 0.01:
            summary += (
                f"Memory degrades at {metrics.degradation_rate:.3f}/turn "
                f"(early: {metrics.early_turn_accuracy:.1%}, "
                f"late: {metrics.late_turn_accuracy:.1%}). "
            )
        else:
            summary += "Memory shows minimal degradation over turns. "

        return MemoryReport(
            title=title,
            profile_name=profile_name,
            metrics=metrics,
            results=results,
            summary=summary,
        )


# ── Built-in memory benchmarks ────────────────────────────────────────


def _build_episodic_task() -> MemoryBenchmarkTask:
    """Test episodic memory: recalling specific events from earlier turns."""
    turns = []
    # Turn 0-2: inject events
    events = [
        MemoryEntry(id="ev1", content="System experienced a disk failure at 14:32 UTC",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=0, importance=0.9),
        MemoryEntry(id="ev2", content="Database migration completed successfully at 18:00",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=0, importance=0.8),
        MemoryEntry(id="ev3", content="Load balancer auto-scaled from 3 to 5 nodes",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=1, importance=0.7),
        MemoryEntry(id="ev4", content="Security patch CVE-2024-1234 applied to all nodes",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=2, importance=0.95),
    ]
    turns.append({"inject": [events[0], events[1]], "recall": []})
    turns.append({"inject": [events[2]], "recall": ["ev1"]})  # recall after 1 turn
    turns.append({"inject": [events[3]], "recall": []})
    # Later turns: test recall
    for i in range(3, 10):
        recall_targets = [e.id for e in events if e.injected_at_turn <= i - 2]
        turns.append({"inject": [], "recall": recall_targets[:2]})
    return MemoryBenchmarkTask(
        id="episodic-memory",
        name="Episodic Memory",
        description="Tests ability to recall specific events (disk failure, migration, scaling, patching) from earlier conversation turns",
        turns=turns,
    )


def _build_semantic_task() -> MemoryBenchmarkTask:
    """Test semantic memory: recalling facts presented during conversation."""
    facts = [
        MemoryEntry(id="f1", content="The API rate limit is 1000 requests per minute per tenant",
                     memory_type=MemoryType.SEMANTIC, injected_at_turn=0, importance=0.9),
        MemoryEntry(id="f2", content="Authentication tokens expire after 24 hours",
                     memory_type=MemoryType.SEMANTIC, injected_at_turn=1, importance=0.85),
        MemoryEntry(id="f3", content="The primary database runs PostgreSQL 16 with connection pooling",
                     memory_type=MemoryType.SEMANTIC, injected_at_turn=2, importance=0.8),
        MemoryEntry(id="f4", content="Error codes 5xx trigger automatic retry with exponential backoff",
                     memory_type=MemoryType.SEMANTIC, injected_at_turn=3, importance=0.75),
        MemoryEntry(id="f5", content="Webhook payloads are limited to 256KB",
                     memory_type=MemoryType.SEMANTIC, injected_at_turn=4, importance=0.7),
    ]
    turns = []
    for i, fact in enumerate(facts):
        turns.append({"inject": [fact], "recall": []})
    # Test recall at turns 6-15
    for i in range(5, 18):
        recall = [f"f{j+1}" for j in range(min(i - 3, len(facts)))]
        turns.append({"inject": [], "recall": recall[:3]})
    return MemoryBenchmarkTask(
        id="semantic-memory",
        name="Semantic Memory",
        description="Tests retention of factual information (API limits, auth config, DB version, error handling) across a long conversation",
        turns=turns,
    )


def _build_working_task() -> MemoryBenchmarkTask:
    """Test working memory: maintaining task context across steps."""
    turns = []
    # Step 1: define a multi-step task
    turns.append({"inject": [
        MemoryEntry(id="w1", content="Task: deploy new microservice 'payment-processor' to staging",
                     memory_type=MemoryType.WORKING, injected_at_turn=0, importance=1.0),
    ], "recall": []})
    # Step 2: sub-step context
    turns.append({"inject": [
        MemoryEntry(id="w2", content="Step 1: Build Docker image from commit abc123",
                     memory_type=MemoryType.WORKING, injected_at_turn=1, importance=0.9),
    ], "recall": ["w1"]})
    # Step 3: more sub-steps
    turns.append({"inject": [
        MemoryEntry(id="w3", content="Step 2: Push to ECR and update ECS task definition",
                     memory_type=MemoryType.WORKING, injected_at_turn=2, importance=0.9),
    ], "recall": ["w1", "w2"]})
    turns.append({"inject": [
        MemoryEntry(id="w4", content="Step 3: Run integration tests against staging endpoint",
                     memory_type=MemoryType.WORKING, injected_at_turn=3, importance=0.85),
    ], "recall": ["w1"]})
    # Later turns: verify full task context is preserved
    for i in range(4, 12):
        turns.append({"inject": [], "recall": ["w1", "w2", "w3", "w4"][:min(4, i)]})
    return MemoryBenchmarkTask(
        id="working-memory",
        name="Working Memory",
        description="Tests maintenance of multi-step task context (deployment pipeline) across execution steps",
        turns=turns,
    )


def _build_cross_conversation_task() -> MemoryBenchmarkTask:
    """Test cross-conversation memory: recalling info from prior conversations."""
    turns = []
    # "Session 1": turns 0-3
    turns.append({"inject": [
        MemoryEntry(id="cc1", content="User reported intermittent 503 errors on /api/v2/orders",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=0, importance=0.85),
        MemoryEntry(id="cc2", content="Investigation revealed memcached connection pool exhaustion",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=1, importance=0.9),
    ], "recall": []})
    turns.append({"inject": [], "recall": ["cc1"]})
    turns.append({"inject": [
        MemoryEntry(id="cc3", content="Fix applied: increased pool size from 50 to 200 connections",
                     memory_type=MemoryType.EPISODIC, injected_at_turn=2, importance=0.95),
    ], "recall": []})
    # Turn 3: end of session 1
    turns.append({"inject": [], "recall": ["cc1", "cc2", "cc3"]})

    # "Session 2": turns 4-10 (simulated gap)
    for i in range(4, 8):
        turns.append({"inject": [
            MemoryEntry(id=f"distractor_{i}", content=f"New task context for turn {i}",
                         memory_type=MemoryType.SEMANTIC, injected_at_turn=i, importance=0.3),
        ], "recall": []})
    # Test cross-session recall
    for i in range(8, 15):
        turns.append({"inject": [], "recall": ["cc1", "cc2", "cc3"][:2]})
    return MemoryBenchmarkTask(
        id="cross-conversation",
        name="Cross-Conversation Memory",
        description="Tests ability to recall information from a prior conversation session after intervening unrelated context",
        turns=turns,
    )


def _build_degradation_task() -> MemoryBenchmarkTask:
    """Test memory degradation: recall quality over very long conversations."""
    turns = []
    # Inject 20 facts across 20 turns
    for i in range(20):
        turns.append({"inject": [
            MemoryEntry(
                id=f"dg{i}",
                content=f"Configuration parameter config_{i} = value_{i}",
                memory_type=MemoryType.SEMANTIC,
                injected_at_turn=i,
                importance=0.5 + (0.5 / 20) * i,  # later facts more important
            ),
        ], "recall": []})
    # Test recall of all facts at turns 8, 15, 20, 25
    for test_turn in [8, 15, 20, 25]:
        targets = [f"dg{i}" for i in range(min(test_turn, 20))]
        turns.append({"inject": [], "recall": targets})
    return MemoryBenchmarkTask(
        id="memory-degradation",
        name="Memory Degradation Over Time",
        description="Tests how recall quality degrades as conversation length grows (20 facts over 25 turns)",
        turns=turns,
    )


ALL_MEMORY_BENCHMARKS = [
    _build_episodic_task(),
    _build_semantic_task(),
    _build_working_task(),
    _build_cross_conversation_task(),
    _build_degradation_task(),
]
