"""
Tests for AgentOps v0.29: Multi-Agent System Evaluation Framework.

Covers: benchmark task definitions, coordination metrics, evaluator,
report formatting, and edge cases.
"""

import pytest
from agentops.multi_agent.benchmarks import (
    CoordinationPattern,
    BenchmarkDifficulty,
    MultiAgentBenchmarkTask,
    MultiAgentBenchmark,
    MULTI_AGENT_BENCHMARK,
    get_multi_agent_benchmark,
    get_tasks_by_difficulty,
    get_tasks_by_pattern,
)
from agentops.multi_agent.eval import (
    CoordinationMetrics,
    MultiAgentEvalReport,
    MultiAgentEvaluator,
    evaluate_multi_agent_system,
    format_multi_agent_eval_report,
    decomposition_quality,
    routing_accuracy,
    coordination_efficiency,
    conflict_resolution_quality,
    synthesis_quality,
    load_balance_score,
    scalability_score,
)
from agentops.multi_agent.state import (
    WorkerRole,
    WorkerAssignment,
    WorkerResult,
    InterAgentMessage,
)
from agentops.multi_agent.coordinator import MultiAgentRunResult


# ═══════════════════════════════════════════════════════════════════
# Helpers — create realistic run results
# ═══════════════════════════════════════════════════════════════════

def _make_worker_result(
    worker_role="retrieval_specialist",
    answer="Test answer",
    verification_passed=True,
    latency_ms=100.0,
    grounded_claims=None,
    ungrounded_claims=None,
    error=None,
):
    return {
        "assignment_id": "asgn-001",
        "worker_role": worker_role,
        "subtask": "Test subtask",
        "answer": answer,
        "grounded_claims": grounded_claims or ["claim 1", "claim 2"],
        "ungrounded_claims": ungrounded_claims or [],
        "citations_used": ["doc-1"],
        "verification_passed": verification_passed,
        "tool_calls_count": 0,
        "retrieved_chunks_count": 2,
        "latency_ms": latency_ms,
        "error": error,
    }


def _make_assignment(worker_role="retrieval_specialist"):
    return {
        "assignment_id": "asgn-001",
        "worker_role": worker_role,
        "subtask": "Test subtask",
        "context": "",
        "priority": 0,
    }


def _make_message(
    from_agent="supervisor",
    to_agent="retrieval_specialist",
    msg_type="assignment",
    content="Test",
):
    return {
        "msg_id": "msg-001",
        "from_agent": from_agent,
        "to_agent": to_agent,
        "msg_type": msg_type,
        "content": content,
        "timestamp_ms": 1000.0,
        "metadata": {},
    }


def _make_run_result(
    task="test task",
    worker_count=2,
    subtasks=None,
    assignments=None,
    worker_results=None,
    messages=None,
    aggregated_answer="test answer",
    verification_passed=True,
    grounded_claims=None,
    ungrounded_claims=None,
    success=True,
):
    return MultiAgentRunResult(
        run_id="run-001",
        task=task,
        subtasks=subtasks or ["subtask 1", "subtask 2"],
        assignments=assignments or [_make_assignment(), _make_assignment("verifier")],
        worker_results=worker_results or [
            _make_worker_result("retrieval_specialist"),
            _make_worker_result("verifier"),
        ],
        inter_agent_messages=messages or [_make_message(), _make_message("verifier", "supervisor", "result")],
        aggregated_answer=aggregated_answer,
        final_answer=aggregated_answer,
        verification_passed=verification_passed,
        grounded_claims=grounded_claims or ["claim 1", "claim 2"],
        ungrounded_claims=ungrounded_claims or [],
        citations_used=["doc-1"],
        coordination_trace=[],
        error=None,
        total_latency_ms=250.0,
        worker_count=worker_count,
        success=success,
    )


# ═══════════════════════════════════════════════════════════════════
# Benchmark Task Tests
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkTasks:
    """Tests for multi-agent benchmark task definitions."""

    def test_benchmark_has_10_tasks(self):
        assert MULTI_AGENT_BENCHMARK.total_tasks == 10

    def test_benchmark_covers_all_difficulties(self):
        difficulties = set(t.difficulty for t in MULTI_AGENT_BENCHMARK.tasks)
        assert BenchmarkDifficulty.EASY in difficulties
        assert BenchmarkDifficulty.MEDIUM in difficulties
        assert BenchmarkDifficulty.HARD in difficulties
        assert BenchmarkDifficulty.EXPERT in difficulties

    def test_benchmark_covers_all_patterns(self):
        patterns = set(t.coordination_pattern for t in MULTI_AGENT_BENCHMARK.tasks)
        assert CoordinationPattern.INDEPENDENT in patterns
        assert CoordinationPattern.SEQUENTIAL in patterns
        assert CoordinationPattern.CONSENSUS in patterns
        assert CoordinationPattern.FAN_IN in patterns

    def test_easy_tasks_have_min_2_workers(self):
        for t in get_tasks_by_difficulty(BenchmarkDifficulty.EASY):
            assert t.min_workers >= 2

    def test_expert_tasks_have_min_4_workers(self):
        for t in get_tasks_by_difficulty(BenchmarkDifficulty.EXPERT):
            assert t.min_workers >= 4

    def test_consensus_tasks_have_contradictions(self):
        for t in get_tasks_by_pattern(CoordinationPattern.CONSENSUS):
            assert t.has_contradictions is True

    def test_all_tasks_have_key_facts(self):
        for t in MULTI_AGENT_BENCHMARK.tasks:
            assert len(t.key_facts) > 0, f"Task {t.id} has no key facts"

    def test_all_tasks_have_expected_roles(self):
        for t in MULTI_AGENT_BENCHMARK.tasks:
            assert len(t.expected_worker_roles) > 0, f"Task {t.id} has no expected roles"

    def test_filter_by_difficulty(self):
        easy = get_tasks_by_difficulty(BenchmarkDifficulty.EASY)
        assert len(easy) == 2
        for t in easy:
            assert t.difficulty == BenchmarkDifficulty.EASY

    def test_filter_by_pattern(self):
        consensus = get_tasks_by_pattern(CoordinationPattern.CONSENSUS)
        assert len(consensus) >= 2
        for t in consensus:
            assert t.coordination_pattern == CoordinationPattern.CONSENSUS

    def test_get_benchmark_factory(self):
        bm = get_multi_agent_benchmark()
        assert bm is MULTI_AGENT_BENCHMARK

    def test_benchmark_task_attributes(self):
        t = MULTI_AGENT_BENCHMARK.tasks[0]
        assert t.id.startswith("ma-")
        assert len(t.name) > 0
        assert len(t.task) > 50
        assert len(t.domain) > 0

    def test_benchmark_by_domain(self):
        security = MULTI_AGENT_BENCHMARK.by_domain("security")
        assert len(security) >= 2

    def test_hard_tasks_require_tool_use(self):
        hard = get_tasks_by_difficulty(BenchmarkDifficulty.HARD)
        for t in hard:
            assert t.requires_tool_use is True

    def test_expert_task_ma010_has_all_patterns_covered(self):
        t = MULTI_AGENT_BENCHMARK.tasks[9]  # ma-010
        assert t.id == "ma-010"
        assert t.difficulty == BenchmarkDifficulty.EXPERT
        assert "47-minute" in t.task
        assert "root cause" in t.task


# ═══════════════════════════════════════════════════════════════════
# Decomposition Quality Tests
# ═══════════════════════════════════════════════════════════════════

class TestDecompositionQuality:
    """Tests for decomposition_quality metric."""

    def test_exact_match(self):
        score = decomposition_quality(3, 3, ["a", "b", "c"], ["a", "b", "c"])
        assert score == pytest.approx(1.0)

    def test_over_decomposition(self):
        score = decomposition_quality(5, 3, ["a", "b", "c", "d", "e"], ["a", "b", "c"])
        assert 0.4 < score < 0.9  # Penalized but not zero

    def test_under_decomposition(self):
        score = decomposition_quality(1, 3, ["a"], ["a", "b", "c"])
        assert 0.2 < score < 0.7

    def test_no_expected_count(self):
        score = decomposition_quality(5, 0, ["a"], ["a"])
        assert score == pytest.approx(1.0)

    def test_role_mismatch(self):
        score = decomposition_quality(3, 3, ["x", "y", "z"], ["a", "b", "c"])
        assert score < 0.7  # Good count but bad roles

    def test_empty_expected_roles(self):
        score = decomposition_quality(2, 2, ["a"], [])
        assert score == pytest.approx(1.0)  # No expected roles = no penalty

    def test_both_empty(self):
        score = decomposition_quality(2, 2, [], [])
        assert score == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# Routing Accuracy Tests
# ═══════════════════════════════════════════════════════════════════

class TestRoutingAccuracy:
    """Tests for routing_accuracy metric."""

    def test_perfect_routing(self):
        assignments = [
            _make_assignment("retrieval_specialist"),
            _make_assignment("verifier"),
        ]
        score = routing_accuracy(assignments, ["retrieval_specialist", "verifier"])
        assert score == pytest.approx(1.0)

    def test_partial_routing(self):
        assignments = [
            _make_assignment("retrieval_specialist"),
            _make_assignment("wrong_role"),
        ]
        score = routing_accuracy(assignments, ["retrieval_specialist", "verifier"])
        assert score == pytest.approx(0.5)

    def test_all_wrong_routing(self):
        assignments = [
            _make_assignment("wrong_1"),
            _make_assignment("wrong_2"),
        ]
        score = routing_accuracy(assignments, ["retrieval_specialist"])
        assert score == pytest.approx(0.0)

    def test_empty_assignments(self):
        score = routing_accuracy([], ["retrieval_specialist"])
        assert score == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# Coordination Efficiency Tests
# ═══════════════════════════════════════════════════════════════════

class TestCoordinationEfficiency:
    """Tests for coordination_efficiency metric."""

    def test_optimal_independent_efficiency(self):
        messages = [_make_message() for _ in range(3)]  # 3 msgs, 2 workers = 1.5 ratio
        score = coordination_efficiency(messages, 2, CoordinationPattern.INDEPENDENT)
        assert score == pytest.approx(1.0)

    def test_too_many_messages(self):
        messages = [_make_message() for _ in range(20)]
        score = coordination_efficiency(messages, 2, CoordinationPattern.INDEPENDENT)
        assert score < 0.5

    def test_single_worker_efficiency(self):
        score = coordination_efficiency([], 1, CoordinationPattern.INDEPENDENT)
        assert score == pytest.approx(1.0)

    def test_consensus_allows_more_messages(self):
        # Consensus pattern: 4 workers, 15 messages = 3.75 per worker (within 2.5-5.0)
        messages = [_make_message() for _ in range(15)]
        score = coordination_efficiency(messages, 4, CoordinationPattern.CONSENSUS)
        assert score == pytest.approx(1.0)

    def test_sequential_pattern_range(self):
        # Sequential: 2 workers, 5 messages = 2.5 per worker (within 2.0-3.0)
        messages = [_make_message() for _ in range(5)]
        score = coordination_efficiency(messages, 2, CoordinationPattern.SEQUENTIAL)
        assert score == pytest.approx(1.0)

    def test_too_few_messages(self):
        # 0 messages for 4 workers is suspiciously low
        score = coordination_efficiency([], 4, CoordinationPattern.INDEPENDENT)
        assert 0.4 < score < 0.8


# ═══════════════════════════════════════════════════════════════════
# Conflict Resolution Quality Tests
# ═══════════════════════════════════════════════════════════════════

class TestConflictResolutionQuality:
    """Tests for conflict_resolution_quality metric."""

    def test_no_contradictions_perfect(self):
        score = conflict_resolution_quality(
            has_contradictions=False,
            verification_passed=True,
            worker_verification_results=[True, True],
            ungrounded_claims=[],
        )
        assert score == pytest.approx(1.0)

    def test_contradiction_resolved(self):
        score = conflict_resolution_quality(
            has_contradictions=True,
            verification_passed=True,
            worker_verification_results=[True, True],
            ungrounded_claims=["conflict detected"],
        )
        assert score == pytest.approx(1.0)

    def test_contradiction_detected_but_unresolved(self):
        score = conflict_resolution_quality(
            has_contradictions=True,
            verification_passed=False,
            worker_verification_results=[True, False],
            ungrounded_claims=["conflict detected"],
        )
        assert score == pytest.approx(0.5)

    def test_contradiction_missed(self):
        score = conflict_resolution_quality(
            has_contradictions=True,
            verification_passed=True,
            worker_verification_results=[True, True],
            ungrounded_claims=[],
        )
        assert score == pytest.approx(0.3)

    def test_contradiction_complete_failure(self):
        score = conflict_resolution_quality(
            has_contradictions=True,
            verification_passed=False,
            worker_verification_results=[False, False],
            ungrounded_claims=[],
        )
        assert score == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# Synthesis Quality Tests
# ═══════════════════════════════════════════════════════════════════

class TestSynthesisQuality:
    """Tests for synthesis_quality metric."""

    def test_perfect_synthesis(self):
        score = synthesis_quality(
            aggregated_answer="The system uses TLS 1.3 and AES-256 encryption for data at rest and in transit.",
            key_facts=["TLS 1.3", "AES-256", "encryption"],
            worker_count=2,
            workers_contributed=2,
        )
        assert score > 0.8

    def test_partial_synthesis(self):
        score = synthesis_quality(
            aggregated_answer="The system uses encryption for data.",
            key_facts=["TLS 1.3", "AES-256", "encryption", "SOC 2"],
            worker_count=4,
            workers_contributed=2,
        )
        assert 0.2 < score < 0.7

    def test_no_key_facts(self):
        score = synthesis_quality(
            aggregated_answer="anything",
            key_facts=[],
            worker_count=4,
            workers_contributed=4,
        )
        assert score == pytest.approx(1.0)

    def test_empty_answer(self):
        score = synthesis_quality(
            aggregated_answer="",
            key_facts=["TLS 1.3", "AES-256"],
            worker_count=2,
            workers_contributed=2,
        )
        # Empty answer gets 0 fact coverage, but non-zero from worker utilization
        assert score < 0.4

    def test_all_workers_contributed(self):
        score = synthesis_quality(
            aggregated_answer="TLS 1.3 AES-256 SOC 2 encryption audit compliance",
            key_facts=["TLS 1.3", "AES-256", "SOC 2"],
            worker_count=4,
            workers_contributed=4,
        )
        assert score > 0.8

    def test_single_worker(self):
        score = synthesis_quality(
            aggregated_answer="TLS 1.3 encryption",
            key_facts=["TLS 1.3", "encryption"],
            worker_count=1,
            workers_contributed=1,
        )
        assert score > 0.8


# ═══════════════════════════════════════════════════════════════════
# Load Balance Tests
# ═══════════════════════════════════════════════════════════════════

class TestLoadBalance:
    """Tests for load_balance_score metric."""

    def test_perfect_balance(self):
        score = load_balance_score([100.0, 100.0, 100.0])
        assert score == pytest.approx(1.0)

    def test_imbalanced(self):
        score = load_balance_score([10.0, 100.0, 500.0])
        assert score < 0.6

    def test_single_worker(self):
        score = load_balance_score([100.0])
        assert score == pytest.approx(1.0)

    def test_all_zero_latency(self):
        score = load_balance_score([0.0, 0.0, 0.0])
        assert score == pytest.approx(0.0)

    def test_near_balance(self):
        score = load_balance_score([95.0, 100.0, 105.0])
        assert score > 0.99


# ═══════════════════════════════════════════════════════════════════
# Scalability Score Tests
# ═══════════════════════════════════════════════════════════════════

class TestScalabilityScore:
    """Tests for scalability_score metric."""

    def test_perfect_scaling(self):
        score = scalability_score(1.0, 1.0, 1.0, 1.0)
        assert score == pytest.approx(1.0)

    def test_degrading_scaling(self):
        score = scalability_score(1.0, 0.8, 0.5, 0.2)
        assert score < 0.5  # Steep drop-offs are penalized heavily

    def test_sharp_dropoff(self):
        score = scalability_score(1.0, 1.0, 0.0, 0.0)
        assert score < 0.5

    def test_all_failure(self):
        score = scalability_score(0.0, 0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# MultiAgentEvaluator Tests
# ═══════════════════════════════════════════════════════════════════

class TestMultiAgentEvaluator:
    """Tests for the MultiAgentEvaluator class."""

    def test_evaluate_perfect_task(self):
        evaluator = MultiAgentEvaluator()
        task = MULTI_AGENT_BENCHMARK.tasks[0]  # ma-001: easy, independent

        result = _make_run_result(
            task=task.task,
            worker_count=2,
            subtasks=["Audit security", "Verify compliance"],
            assignments=[
                _make_assignment("retrieval_specialist"),
                _make_assignment("verifier"),
            ],
            worker_results=[
                _make_worker_result("retrieval_specialist", "TLS 1.3 and AES-256 encryption standards found"),
                _make_worker_result("verifier", "SOC 2 documentation verified, encryption meets standards"),
            ],
            aggregated_answer="Platform uses TLS 1.3 and AES-256 encryption. SOC 2 compliance verified.",
            verification_passed=True,
        )

        metrics = evaluator.evaluate_task(task, result)
        assert metrics.coordination_score > 60.0
        assert metrics.passed is True

    def test_evaluate_poor_task(self):
        evaluator = MultiAgentEvaluator()
        task = MULTI_AGENT_BENCHMARK.tasks[0]

        result = _make_run_result(
            task=task.task,
            worker_count=1,
            subtasks=["wrong decomposition"],
            assignments=[_make_assignment("wrong_role")],
            worker_results=[
                _make_worker_result("wrong_role", "incomplete", verification_passed=False),
            ],
            aggregated_answer="",
            verification_passed=False,
            grounded_claims=[],
            success=False,
        )

        metrics = evaluator.evaluate_task(task, result)
        assert metrics.coordination_score < 60.0, f"Expected <60, got {metrics.coordination_score}"

    def test_evaluate_task_with_contradictions(self):
        evaluator = MultiAgentEvaluator()
        task = [t for t in MULTI_AGENT_BENCHMARK.tasks if t.has_contradictions][0]

        result = _make_run_result(
            task=task.task,
            worker_count=4,
            subtasks=["s1", "s2", "s3", "s4"],
            assignments=[
                _make_assignment("retrieval_specialist"),
                _make_assignment("tool_executor"),
                _make_assignment("code_analyst"),
                _make_assignment("verifier"),
            ],
            worker_results=[
                _make_worker_result("retrieval_specialist", "HIPAA features documented", latency_ms=100),
                _make_worker_result("tool_executor", "Computed migration cost", latency_ms=100),
                _make_worker_result("code_analyst", "Encryption meets requirements", latency_ms=100),
                _make_worker_result("verifier", "PHI storage issues reconciled", latency_ms=100),
            ],
            aggregated_answer="HIPAA compliant migration ready. PHI storage requires additional safeguards.",
            verification_passed=True,
            ungrounded_claims=["PHI storage conflict detected"],
        )

        metrics = evaluator.evaluate_task(task, result)
        assert metrics.conflict_resolution == pytest.approx(1.0)  # Detected + resolved
        assert metrics.coordination_score > 50.0

    def test_evaluate_benchmark_full(self):
        evaluator = MultiAgentEvaluator()
        run_results = {}

        for task in MULTI_AGENT_BENCHMARK.tasks:
            worker_count = task.min_workers
            roles = task.expected_worker_roles[:worker_count]

            run_results[task.id] = _make_run_result(
                task=task.task,
                worker_count=worker_count,
                subtasks=[f"subtask {i}" for i in range(task.expected_decomposition_count)],
                assignments=[_make_assignment(r) for r in roles],
                worker_results=[
                    _make_worker_result(
                        r,
                        f"{r} completed analysis with findings about {' '.join(task.key_facts[:3])}",
                        latency_ms=100.0 + i * 10,
                    )
                    for i, r in enumerate(roles)
                ],
                aggregated_answer=f"Results: {' '.join(task.key_facts[:5])}",
                verification_passed=True,
                ungrounded_claims=["conflict detected"] if task.has_contradictions else [],
            )

        report = evaluator.evaluate_benchmark(MULTI_AGENT_BENCHMARK, run_results)
        assert report.total_tasks == 10
        assert report.pass_rate >= 0.5  # At least half should pass with reasonable data
        assert 0.0 <= report.overall_score <= 100.0

    def test_report_formatting(self):
        evaluator = MultiAgentEvaluator()
        run_results = {}

        for task in MULTI_AGENT_BENCHMARK.tasks[:3]:
            run_results[task.id] = _make_run_result(
                task=task.task,
                worker_count=task.min_workers,
                subtasks=[f"subtask {i}" for i in range(task.expected_decomposition_count)],
                assignments=[_make_assignment(r) for r in task.expected_worker_roles[:task.min_workers]],
                worker_results=[
                    _make_worker_result(r, f"Result from {r}")
                    for r in task.expected_worker_roles[:task.min_workers]
                ],
                aggregated_answer=f"Analysis: {' '.join(task.key_facts[:3])}",
                verification_passed=True,
            )

        # Partial benchmark evaluation
        partial_benchmark = MultiAgentBenchmark(
            name="test-benchmark",
            description="Test",
            tasks=MULTI_AGENT_BENCHMARK.tasks[:3],
        )
        report = evaluator.evaluate_benchmark(partial_benchmark, run_results)
        formatted = evaluator.format_report(report)

        assert "Multi-Agent Coordination Evaluation Report" in formatted
        assert "test-benchmark" in formatted
        assert "Overall" in formatted
        assert "Passed:" in formatted
        assert "Metric Averages" in formatted
        assert "Detailed Results" in formatted
        assert "Decomposition Quality" in formatted
        assert "Scores by Difficulty" in formatted

    def test_convenience_functions(self):
        run_results = {
            MULTI_AGENT_BENCHMARK.tasks[0].id: _make_run_result(
                task=MULTI_AGENT_BENCHMARK.tasks[0].task,
                worker_count=2,
                subtasks=["s1", "s2"],
                assignments=[
                    _make_assignment("retrieval_specialist"),
                    _make_assignment("verifier"),
                ],
                worker_results=[
                    _make_worker_result("retrieval_specialist", "TLS 1.3 AES-256 found"),
                    _make_worker_result("verifier", "SOC 2 verified"),
                ],
                aggregated_answer="TLS 1.3 AES-256 SOC 2 encryption audit",
                verification_passed=True,
            ),
        }

        report = evaluate_multi_agent_system(run_results, topology="supervisor-worker")
        assert isinstance(report, MultiAgentEvalReport)

        formatted = format_multi_agent_eval_report(report)
        assert "Multi-Agent Coordination Evaluation Report" in formatted


class TestCoordinationMetrics:
    """Tests for CoordinationMetrics dataclass."""

    def test_default_metrics(self):
        m = CoordinationMetrics(
            task_id="test-1",
            task_name="Test Task",
            difficulty=BenchmarkDifficulty.EASY,
            coordination_pattern=CoordinationPattern.INDEPENDENT,
        )
        assert m.coordination_score == 0.0
        assert m.passed is False
        assert m.message_count == 0

    def test_high_scoring_metrics(self):
        m = CoordinationMetrics(
            task_id="test-1",
            task_name="Test",
            difficulty=BenchmarkDifficulty.EASY,
            coordination_pattern=CoordinationPattern.INDEPENDENT,
            decomposition=1.0,
            routing=1.0,
            efficiency=1.0,
            conflict_resolution=1.0,
            synthesis=1.0,
            load_balance=1.0,
            coordination_score=100.0,
            passed=True,
            message_count=4,
        )
        assert m.coordination_score == 100.0
        assert m.passed is True


class TestMultiAgentEvalReport:
    """Tests for MultiAgentEvalReport dataclass."""

    def test_pass_rate_calculation(self):
        report = MultiAgentEvalReport(
            benchmark_name="test",
            total_tasks=10,
            passed_tasks=7,
            failed_tasks=3,
            overall_score=72.5,
        )
        assert report.pass_rate == pytest.approx(0.7)

    def test_zero_tasks(self):
        report = MultiAgentEvalReport(
            benchmark_name="test",
            total_tasks=0,
            passed_tasks=0,
            failed_tasks=0,
            overall_score=0.0,
        )
        assert report.pass_rate == pytest.approx(0.0)

    def test_summary_string(self):
        report = MultiAgentEvalReport(
            benchmark_name="test",
            total_tasks=10,
            passed_tasks=8,
            failed_tasks=2,
            overall_score=78.0,
        )
        summary = report.summary()
        assert "8/10" in summary
        assert "78/100" in summary


class TestEdgeCases:
    """Edge case tests for multi-agent evaluation."""

    def test_evaluate_with_missing_result(self):
        evaluator = MultiAgentEvaluator()
        # Only provide results for 2 of 10 tasks
        run_results = {
            MULTI_AGENT_BENCHMARK.tasks[0].id: _make_run_result(
                worker_count=2,
                subtasks=["s1", "s2"],
                assignments=[
                    _make_assignment("retrieval_specialist"),
                    _make_assignment("verifier"),
                ],
                worker_results=[
                    _make_worker_result("retrieval_specialist", "OK"),
                    _make_worker_result("verifier", "OK"),
                ],
                aggregated_answer="OK",
            ),
        }

        report = evaluator.evaluate_benchmark(MULTI_AGENT_BENCHMARK, run_results)
        assert report.total_tasks == 10
        assert report.passed_tasks <= 1  # Only 1 task had results
        # Missing tasks should have zero scores
        missing = [r for r in report.results if "Not executed" in r.notes[0]]
        assert len(missing) == 9

    def test_worker_error_in_result(self):
        evaluator = MultiAgentEvaluator()
        task = MULTI_AGENT_BENCHMARK.tasks[0]

        result = _make_run_result(
            worker_count=2,
            subtasks=["s1", "s2"],
            assignments=[
                _make_assignment("retrieval_specialist"),
                _make_assignment("verifier"),
            ],
            worker_results=[
                _make_worker_result("retrieval_specialist", "OK"),
                _make_worker_result("verifier", "", error="Worker crashed"),
            ],
            aggregated_answer="Partial results",
            verification_passed=False,
            grounded_claims=["claim 1"],
            success=False,
        )

        metrics = evaluator.evaluate_task(task, result)
        # Should still compute metrics even with errors
        assert metrics.synthesis < 0.5  # One worker errored
        assert metrics.coordination_score < 100.0
