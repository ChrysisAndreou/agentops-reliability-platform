"""
Multi-Agent System Evaluation Framework.

Evaluates the quality of multi-agent coordination, not just individual agent
performance. Metrics cover:

1. Decomposition Quality — did the supervisor decompose correctly?
2. Routing Accuracy — were subtasks assigned to the right workers?
3. Coordination Efficiency — how much inter-agent communication was needed?
4. Conflict Resolution — were contradictory outputs reconciled correctly?
5. Synthesis Quality — did the aggregated answer capture all worker insights?
6. Scalability — how does performance change with worker count?
7. Topology Fitness — which topology best fits the task?

Each metric is normalized to [0.0, 1.0] for composability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agentops.multi_agent.benchmarks import (
    BenchmarkDifficulty,
    CoordinationPattern,
    MultiAgentBenchmark,
    MultiAgentBenchmarkTask,
    MULTI_AGENT_BENCHMARK,
)
from agentops.multi_agent.state import (
    WorkerRole,
    WorkerAssignment,
    WorkerResult,
    InterAgentMessage,
)
from agentops.multi_agent.coordinator import MultiAgentRunResult


# ═══════════════════════════════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════════════════════════════

def decomposition_quality(
    actual_decomposition_count: int,
    expected_count: int,
    actual_roles: List[str],
    expected_roles: List[str],
) -> float:
    """How well the supervisor decomposed the task.

    Combines:
    - Count accuracy: did we create the right number of subtasks?
    - Role coverage: did we assign the right worker types?

    Returns:
        Score 0.0-1.0. 1.0 = perfect decomposition.
    """
    if expected_count == 0:
        return 1.0

    # Count accuracy — exponential decay for over/under decomposition
    count_diff = abs(actual_decomposition_count - expected_count)
    count_score = max(0.0, 1.0 - count_diff / max(expected_count, 1))

    # Role coverage — Jaccard similarity between expected and actual roles
    expected_set = set(expected_roles)
    actual_set = set(actual_roles)
    if not expected_set:
        role_score = 1.0
    else:
        intersection = expected_set & actual_set
        union = expected_set | actual_set
        role_score = len(intersection) / len(union) if union else 0.0

    # Weighted combination: count matters more than exact role match
    return 0.6 * count_score + 0.4 * role_score


def routing_accuracy(
    assignments: List[WorkerAssignment],
    expected_roles: List[str],
) -> float:
    """How accurately subtasks were routed to appropriate workers.

    Each assignment gets a score based on whether the assigned worker
    role appears in the expected role list.

    Returns:
        0.0-1.0. 1.0 = all assignments routed correctly.
    """
    if not assignments:
        return 0.0

    expected_set = set(expected_roles)
    correct = sum(
        1 for a in assignments
        if a.get("worker_role", "") in expected_set
    )
    return correct / len(assignments)


def coordination_efficiency(
    inter_agent_messages: List[InterAgentMessage],
    worker_count: int,
    coordination_pattern: CoordinationPattern,
) -> float:
    """How efficiently agents coordinated — fewer messages is better,
    but some patterns naturally require more communication.

    Measures:
    - Message-to-worker ratio (lower = more efficient)
    - Pattern awareness (does message volume match pattern expectations?)

    Returns:
        0.0-1.0. 1.0 = optimally efficient for the pattern.
    """
    if worker_count <= 1:
        return 1.0

    msg_count = len(inter_agent_messages)

    # Expected message ranges per pattern (per worker)
    expected_ranges: dict[CoordinationPattern, Tuple[float, float]] = {
        CoordinationPattern.INDEPENDENT: (1.0, 2.0),    # assignment + result
        CoordinationPattern.SEQUENTIAL: (2.0, 3.0),     # assignment, handoff, result
        CoordinationPattern.CONSENSUS: (2.5, 5.0),      # assignment, negotiation, result
        CoordinationPattern.PIPELINE: (2.0, 3.5),       # handoffs between stages
        CoordinationPattern.FAN_OUT: (1.5, 3.0),        # distribute + collect
        CoordinationPattern.FAN_IN: (1.5, 3.0),         # collect + aggregate
    }

    low, high = expected_ranges.get(coordination_pattern, (1.0, 3.0))
    msgs_per_worker = msg_count / worker_count

    if low <= msgs_per_worker <= high:
        return 1.0
    elif msgs_per_worker < low:
        # Too few messages — possible incomplete coordination
        return max(0.5, msgs_per_worker / low)
    else:
        # Too many messages — inefficient
        penalty = (msgs_per_worker - high) / high
        return max(0.0, 1.0 - penalty)


def conflict_resolution_quality(
    has_contradictions: bool,
    verification_passed: bool,
    worker_verification_results: List[bool],
    ungrounded_claims: List[str],
) -> float:
    """How well the system resolved contradictory information.

    For tasks without contradictions, score defaults to 1.0.
    For tasks with contradictions:
    - Did the system detect the conflict? (high ungrounded claims = detected)
    - Did it resolve correctly? (verification_passed = success)

    Returns:
        0.0-1.0.
    """
    if not has_contradictions:
        return 1.0

    # For contradictory tasks, we need the system to:
    # 1. Detect the contradiction (reflected in ungrounded claims)
    # 2. Resolve it (reflected in verification outcome)

    conflict_detected = len(ungrounded_claims) > 0
    conflict_resolved = verification_passed and all(worker_verification_results) if worker_verification_results else verification_passed

    if conflict_detected and conflict_resolved:
        return 1.0
    elif conflict_detected and not conflict_resolved:
        return 0.5  # Detected but failed to resolve
    elif not conflict_detected and conflict_resolved:
        return 0.3  # Missed the conflict but got lucky
    else:
        return 0.0  # Neither detected nor resolved


def synthesis_quality(
    aggregated_answer: str,
    key_facts: List[str],
    worker_count: int,
    workers_contributed: int,
) -> float:
    """How well the aggregated answer synthesizes all worker contributions.

    Measured by:
    - Key fact coverage: what fraction of expected facts appear?
    - Worker contribution utilization: did all workers' outputs inform the answer?

    Returns:
        0.0-1.0.
    """
    if not key_facts:
        return 1.0

    answer_lower = aggregated_answer.lower()

    # Key fact coverage with partial matching
    found = 0
    for fact in key_facts:
        # Split multi-word facts and check if most words appear
        fact_words = fact.lower().split()
        words_found = sum(1 for w in fact_words if w in answer_lower)
        if words_found >= len(fact_words) * 0.6:  # 60% word match = fact covered
            found += 1

    fact_score = found / len(key_facts)

    # Worker utilization
    if worker_count <= 1:
        util_score = 1.0
    else:
        util_score = min(1.0, workers_contributed / worker_count)

    return 0.7 * fact_score + 0.3 * util_score


def load_balance_score(
    worker_latencies_ms: List[float],
) -> float:
    """How evenly work was distributed across workers.

    Uses the Jain's fairness index: (Σx)² / (n * Σx²)
    1.0 = perfectly balanced, 1/n = maximally unbalanced.

    Returns:
        0.0-1.0.
    """
    if len(worker_latencies_ms) <= 1:
        return 1.0

    n = len(worker_latencies_ms)
    total = sum(worker_latencies_ms)
    if total == 0:
        return 0.0

    sum_sq = sum(x * x for x in worker_latencies_ms)
    jain = (total * total) / (n * sum_sq) if sum_sq > 0 else 0.0
    return jain


def scalability_score(
    easy_success_rate: float,
    medium_success_rate: float,
    hard_success_rate: float,
    expert_success_rate: float,
) -> float:
    """How well the system scales with task difficulty.

    A good system maintains success across difficulty tiers.
    Steep drop-offs indicate scalability problems.

    Returns:
        0.0-1.0. 1.0 = consistent across all difficulties.
    """
    rates = [easy_success_rate, medium_success_rate, hard_success_rate, expert_success_rate]
    if not rates:
        return 0.0

    # Compute weighted average with heavier weight on harder tasks
    weights = [0.1, 0.2, 0.3, 0.4]
    weighted_avg = sum(r * w for r, w in zip(rates, weights)) / sum(weights)

    # Penalize sharp drop-offs
    max_rate = max(rates)
    min_rate = min(rates)
    drop_penalty = (max_rate - min_rate) * 0.5

    return max(0.0, weighted_avg - drop_penalty)


# ═══════════════════════════════════════════════════════════════════
# Evaluation Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CoordinationMetrics:
    """Per-task coordination quality metrics."""
    task_id: str
    task_name: str
    difficulty: BenchmarkDifficulty
    coordination_pattern: CoordinationPattern
    # Individual scores
    decomposition: float = 0.0
    routing: float = 0.0
    efficiency: float = 0.0
    conflict_resolution: float = 0.0
    synthesis: float = 0.0
    load_balance: float = 0.0
    # Aggregate
    coordination_score: float = 0.0  # Weighted composite 0-100
    passed: bool = False
    # Raw data for transparency
    actual_decomposition_count: int = 0
    actual_worker_count: int = 0
    message_count: int = 0
    worker_latencies: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class MultiAgentEvalReport:
    """Complete multi-agent evaluation report."""
    benchmark_name: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    overall_score: float  # 0-100
    # Per-difficulty breakdowns
    easy_score: float = 0.0
    medium_score: float = 0.0
    hard_score: float = 0.0
    expert_score: float = 0.0
    # Per-pattern breakdowns
    pattern_scores: Dict[str, float] = field(default_factory=dict)
    # Per-metric averages
    avg_decomposition: float = 0.0
    avg_routing: float = 0.0
    avg_efficiency: float = 0.0
    avg_conflict_resolution: float = 0.0
    avg_synthesis: float = 0.0
    avg_load_balance: float = 0.0
    scalability: float = 0.0
    # Detailed results
    results: List[CoordinationMetrics] = field(default_factory=list)
    # Metadata
    worker_topology: str = "supervisor-worker"
    total_workers_deployed: int = 0
    total_messages_exchanged: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    def summary(self) -> str:
        return (
            f"Multi-Agent Eval: {self.passed_tasks}/{self.total_tasks} passed "
            f"({self.pass_rate:.0%}), score={self.overall_score:.0f}/100"
        )


# ═══════════════════════════════════════════════════════════════════
# Multi-Agent Evaluator
# ═══════════════════════════════════════════════════════════════════

class MultiAgentEvaluator:
    """Evaluates multi-agent coordination quality across benchmark tasks.

    The evaluator takes benchmark tasks and multi-agent run results,
    computes coordination metrics, and produces structured reports.

    Usage:
        >>> evaluator = MultiAgentEvaluator()
        >>> metrics = evaluator.evaluate_task(task, run_result)
        >>> print(metrics.coordination_score)
        >>> report = evaluator.evaluate_benchmark(MULTI_AGENT_BENCHMARK, all_results)
    """

    # Weights for composite scoring (sum = 1.0)
    METRIC_WEIGHTS = {
        "decomposition": 0.20,
        "routing": 0.15,
        "efficiency": 0.10,
        "conflict_resolution": 0.20,
        "synthesis": 0.25,
        "load_balance": 0.10,
    }

    # Pass threshold per difficulty
    PASS_THRESHOLDS: dict[BenchmarkDifficulty, float] = {
        BenchmarkDifficulty.EASY: 60.0,
        BenchmarkDifficulty.MEDIUM: 55.0,
        BenchmarkDifficulty.HARD: 50.0,
        BenchmarkDifficulty.EXPERT: 45.0,
    }

    def __init__(self, topology: str = "supervisor-worker"):
        self.topology = topology

    def evaluate_task(
        self,
        task: MultiAgentBenchmarkTask,
        run_result: MultiAgentRunResult,
    ) -> CoordinationMetrics:
        """Evaluate a single benchmark task against its run result.

        Args:
            task: The benchmark task definition
            run_result: The multi-agent run result to evaluate

        Returns:
            CoordinationMetrics with all scores computed
        """
        notes: List[str] = []

        # Extract data from run result
        actual_roles = list(set(
            a.get("worker_role", "") for a in run_result.assignments
        ))
        worker_count = run_result.worker_count
        messages = run_result.inter_agent_messages
        worker_verifications = [
            r.get("verification_passed", False)
            for r in run_result.worker_results
        ]

        # 1. Decomposition quality
        decomp = decomposition_quality(
            actual_decomposition_count=len(run_result.subtasks),
            expected_count=task.expected_decomposition_count,
            actual_roles=actual_roles,
            expected_roles=task.expected_worker_roles,
        )

        # 2. Routing accuracy
        routing = routing_accuracy(
            assignments=run_result.assignments,
            expected_roles=task.expected_worker_roles,
        )

        # 3. Coordination efficiency
        efficiency = coordination_efficiency(
            inter_agent_messages=messages,
            worker_count=max(worker_count, 1),
            coordination_pattern=task.coordination_pattern,
        )

        # 4. Conflict resolution
        conflict = conflict_resolution_quality(
            has_contradictions=task.has_contradictions,
            verification_passed=run_result.verification_passed,
            worker_verification_results=worker_verifications,
            ungrounded_claims=run_result.ungrounded_claims,
        )

        # 5. Synthesis quality
        workers_contributed = len([
            r for r in run_result.worker_results
            if r.get("answer", "") and not r.get("error")
        ])
        synthesis = synthesis_quality(
            aggregated_answer=run_result.aggregated_answer,
            key_facts=task.key_facts,
            worker_count=max(worker_count, 1),
            workers_contributed=workers_contributed,
        )

        # 6. Load balance
        latencies = [
            r.get("latency_ms", 0.0)
            for r in run_result.worker_results
        ]
        load_bal = load_balance_score(latencies)

        # Compute composite coordination score
        raw_score = (
            self.METRIC_WEIGHTS["decomposition"] * decomp +
            self.METRIC_WEIGHTS["routing"] * routing +
            self.METRIC_WEIGHTS["efficiency"] * efficiency +
            self.METRIC_WEIGHTS["conflict_resolution"] * conflict +
            self.METRIC_WEIGHTS["synthesis"] * synthesis +
            self.METRIC_WEIGHTS["load_balance"] * load_bal
        )
        coordination_score = raw_score * 100.0

        # Determine pass/fail
        threshold = self.PASS_THRESHOLDS.get(task.difficulty, 50.0)
        passed = coordination_score >= threshold

        # Add contextual notes
        if decomp < 0.5:
            notes.append(
                f"Poor decomposition: expected {task.expected_decomposition_count} "
                f"subtasks, got {len(run_result.subtasks)}"
            )
        if routing < 0.5:
            notes.append(
                f"Routing issues: only {routing:.0%} of assignments matched expected roles"
            )
        if efficiency < 0.5:
            notes.append(
                f"Inefficient coordination: {len(messages)} messages for {worker_count} workers"
            )
        if conflict < 0.7 and task.has_contradictions:
            notes.append("Failed to properly resolve contradictory information")
        if synthesis < 0.4:
            notes.append(
                f"Poor synthesis: only partial key fact coverage in aggregated answer"
            )

        return CoordinationMetrics(
            task_id=task.id,
            task_name=task.name,
            difficulty=task.difficulty,
            coordination_pattern=task.coordination_pattern,
            decomposition=decomp,
            routing=routing,
            efficiency=efficiency,
            conflict_resolution=conflict,
            synthesis=synthesis,
            load_balance=load_bal,
            coordination_score=coordination_score,
            passed=passed,
            actual_decomposition_count=len(run_result.subtasks),
            actual_worker_count=worker_count,
            message_count=len(messages),
            worker_latencies=latencies,
            notes=notes,
        )

    def evaluate_benchmark(
        self,
        benchmark: MultiAgentBenchmark,
        run_results: Dict[str, MultiAgentRunResult],
    ) -> MultiAgentEvalReport:
        """Evaluate all tasks in a benchmark against their run results.

        Args:
            benchmark: The benchmark to evaluate
            run_results: Dict mapping task_id → MultiAgentRunResult

        Returns:
            MultiAgentEvalReport with aggregate scores and breakdowns
        """
        results: List[CoordinationMetrics] = []

        for task in benchmark.tasks:
            run_result = run_results.get(task.id)
            if run_result is None:
                # Task wasn't run — create a zero-score placeholder
                results.append(CoordinationMetrics(
                    task_id=task.id,
                    task_name=task.name,
                    difficulty=task.difficulty,
                    coordination_pattern=task.coordination_pattern,
                    notes=["Not executed — no run result provided"],
                ))
                continue
            metrics = self.evaluate_task(task, run_result)
            results.append(metrics)

        # Compute aggregates
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        overall = sum(r.coordination_score for r in results) / max(total, 1)

        # Per-difficulty breakdown
        def avg_score(metric_list, attr):
            scores = [getattr(r, attr) for r in metric_list]
            return sum(scores) / max(len(scores), 1) * 100.0 if scores else 0.0

        easy_results = [r for r in results if r.difficulty == BenchmarkDifficulty.EASY]
        medium_results = [r for r in results if r.difficulty == BenchmarkDifficulty.MEDIUM]
        hard_results = [r for r in results if r.difficulty == BenchmarkDifficulty.HARD]
        expert_results = [r for r in results if r.difficulty == BenchmarkDifficulty.EXPERT]

        # Per-pattern breakdown
        pattern_scores: Dict[str, float] = {}
        pattern_results: Dict[str, List[CoordinationMetrics]] = {}
        for r in results:
            pattern = r.coordination_pattern.value
            if pattern not in pattern_results:
                pattern_results[pattern] = []
            pattern_results[pattern].append(r)
        for pattern, pr in pattern_results.items():
            pattern_scores[pattern] = sum(r.coordination_score for r in pr) / len(pr) if pr else 0.0

        # Scalability: how does success rate change with difficulty?
        easy_sr = sum(1 for r in easy_results if r.passed) / max(len(easy_results), 1)
        medium_sr = sum(1 for r in medium_results if r.passed) / max(len(medium_results), 1)
        hard_sr = sum(1 for r in hard_results if r.passed) / max(len(hard_results), 1)
        expert_sr = sum(1 for r in expert_results if r.passed) / max(len(expert_results), 1)
        scalability = scalability_score(easy_sr, medium_sr, hard_sr, expert_sr) * 100.0

        return MultiAgentEvalReport(
            benchmark_name=benchmark.name,
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=failed,
            overall_score=overall,
            easy_score=avg_score(easy_results, "coordination_score"),
            medium_score=avg_score(medium_results, "coordination_score"),
            hard_score=avg_score(hard_results, "coordination_score"),
            expert_score=avg_score(expert_results, "coordination_score"),
            pattern_scores=pattern_scores,
            avg_decomposition=sum(r.decomposition for r in results) / max(total, 1),
            avg_routing=sum(r.routing for r in results) / max(total, 1),
            avg_efficiency=sum(r.efficiency for r in results) / max(total, 1),
            avg_conflict_resolution=sum(r.conflict_resolution for r in results) / max(total, 1),
            avg_synthesis=sum(r.synthesis for r in results) / max(total, 1),
            avg_load_balance=sum(r.load_balance for r in results) / max(total, 1),
            scalability=scalability,
            results=results,
            worker_topology=self.topology,
            total_workers_deployed=sum(r.actual_worker_count for r in results),
            total_messages_exchanged=sum(r.message_count for r in results),
        )

    def format_report(self, report: MultiAgentEvalReport) -> str:
        """Format a MultiAgentEvalReport as readable Markdown."""
        lines = [
            "# Multi-Agent Coordination Evaluation Report",
            "",
            f"**Benchmark:** {report.benchmark_name}",
            f"**Topology:** {report.worker_topology}",
            "",
            f"## Overall: {report.overall_score:.0f}/100",
            "",
            f"- **Passed:** {report.passed_tasks}/{report.total_tasks} ({report.pass_rate:.0%})",
            f"- **Failed:** {report.failed_tasks}/{report.total_tasks}",
            f"- **Workers Deployed:** {report.total_workers_deployed}",
            f"- **Messages Exchanged:** {report.total_messages_exchanged}",
            f"- **Scalability Score:** {report.scalability:.0f}/100",
            "",
            "## Scores by Difficulty",
            "",
            f"| Difficulty | Score | Tasks |",
            f"|------------|-------|-------|",
            f"| Easy | {report.easy_score:.0f}/100 | {len([r for r in report.results if r.difficulty == BenchmarkDifficulty.EASY])} |",
            f"| Medium | {report.medium_score:.0f}/100 | {len([r for r in report.results if r.difficulty == BenchmarkDifficulty.MEDIUM])} |",
            f"| Hard | {report.hard_score:.0f}/100 | {len([r for r in report.results if r.difficulty == BenchmarkDifficulty.HARD])} |",
            f"| Expert | {report.expert_score:.0f}/100 | {len([r for r in report.results if r.difficulty == BenchmarkDifficulty.EXPERT])} |",
            "",
            "## Metric Averages",
            "",
            f"| Metric | Average | Weight |",
            f"|--------|---------|--------|",
            f"| Decomposition Quality | {report.avg_decomposition:.2f} | 20% |",
            f"| Routing Accuracy | {report.avg_routing:.2f} | 15% |",
            f"| Coordination Efficiency | {report.avg_efficiency:.2f} | 10% |",
            f"| Conflict Resolution | {report.avg_conflict_resolution:.2f} | 20% |",
            f"| Synthesis Quality | {report.avg_synthesis:.2f} | 25% |",
            f"| Load Balance | {report.avg_load_balance:.2f} | 10% |",
            "",
        ]

        if report.pattern_scores:
            lines.extend([
                "## Scores by Coordination Pattern",
                "",
                "| Pattern | Score |",
                "|---------|-------|",
            ])
            for pattern, score in sorted(report.pattern_scores.items()):
                lines.append(f"| {pattern} | {score:.0f}/100 |")
            lines.append("")

        # Detailed task results
        lines.extend([
            "## Detailed Results",
            "",
            "| # | Task | Difficulty | Pattern | Decomp | Route | Effic | Conflict | Synth | Balance | Score | Pass |",
            "|---|------|------------|---------|--------|-------|-------|----------|-------|---------|-------|------|",
        ])

        for i, r in enumerate(report.results, 1):
            icon = "✓" if r.passed else "✗"
            lines.append(
                f"| {i} | {r.task_name} | {r.difficulty.value} | {r.coordination_pattern.value} | "
                f"{r.decomposition:.2f} | {r.routing:.2f} | {r.efficiency:.2f} | "
                f"{r.conflict_resolution:.2f} | {r.synthesis:.2f} | {r.load_balance:.2f} | "
                f"{r.coordination_score:.0f} | {icon} |"
            )

        # Notes from failures
        failing = [r for r in report.results if not r.passed and r.notes]
        if failing:
            lines.extend([
                "",
                "## Failing Task Notes",
                "",
            ])
            for r in failing:
                lines.append(f"### {r.task_id} — {r.task_name} (score: {r.coordination_score:.0f})")
                for note in r.notes:
                    lines.append(f"- {note}")
                lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════

def evaluate_multi_agent_system(
    run_results: Dict[str, MultiAgentRunResult],
    benchmark: Optional[MultiAgentBenchmark] = None,
    topology: str = "supervisor-worker",
) -> MultiAgentEvalReport:
    """Evaluate a multi-agent system against the standard benchmark.

    Args:
        run_results: Dict mapping task_id → MultiAgentRunResult
        benchmark: Benchmark to use (defaults to MULTI_AGENT_BENCHMARK)
        topology: Topology name for reporting

    Returns:
        MultiAgentEvalReport
    """
    evaluator = MultiAgentEvaluator(topology=topology)
    return evaluator.evaluate_benchmark(
        benchmark or MULTI_AGENT_BENCHMARK,
        run_results,
    )


def format_multi_agent_eval_report(report: MultiAgentEvalReport) -> str:
    """Format a multi-agent evaluation report as Markdown."""
    evaluator = MultiAgentEvaluator()
    return evaluator.format_report(report)
