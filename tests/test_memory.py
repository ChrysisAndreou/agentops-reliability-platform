"""
Tests for the agent memory evaluation module (v0.12).

Covers:
- State models: MemoryEntry, MemoryStore, MemoryContext, profiles
- SimulatedMemoryAgent: recall, recall_all, run_conversation
- MemoryEvaluator: metrics computation, report generation
- Benchmark tasks: all 5 built-in benchmarks
- Memory profiles: all 4 degradation profiles
"""

import pytest

from agentops.memory.state import (
    MemoryType,
    MemoryEntry,
    MemoryStore,
    MemoryContext,
    MemoryRecallResult,
    MemoryMetrics,
    MemoryReport,
    MemoryProfile,
    PERFECT_MEMORY,
    PRODUCTION_MEMORY,
    DEVELOPMENT_MEMORY,
    DEGRADED_MEMORY,
    MEMORY_PROFILES,
    get_memory_profile,
)
from agentops.memory.simulator import SimulatedMemoryAgent
from agentops.memory.metrics import (
    MemoryEvaluator,
    MemoryBenchmarkTask,
    ALL_MEMORY_BENCHMARKS,
)


# ── State model tests ──────────────────────────────────────────────────

class TestMemoryEntry:
    def test_create_entry(self):
        e = MemoryEntry(
            id="test-1",
            content="User prefers dark mode",
            memory_type=MemoryType.SEMANTIC,
            injected_at_turn=3,
            importance=0.8,
        )
        assert e.id == "test-1"
        assert e.memory_type == MemoryType.SEMANTIC
        assert e.injected_at_turn == 3
        assert e.importance == 0.8
        assert e.related_to == []
        assert e.metadata == {}

    def test_entry_with_relations(self):
        e = MemoryEntry(
            id="test-2",
            content="Related fact",
            memory_type=MemoryType.EPISODIC,
            injected_at_turn=1,
            related_to=["test-1"],
            metadata={"source": "user"},
        )
        assert "test-1" in e.related_to
        assert e.metadata["source"] == "user"


class TestMemoryStore:
    def test_inject_and_retrieve(self):
        store = MemoryStore()
        e = MemoryEntry(id="m1", content="Fact 1", memory_type=MemoryType.SEMANTIC,
                        injected_at_turn=0)
        store.inject(e)
        assert store.total == 1
        assert store.get_by_id("m1") is e
        assert store.get_by_id("nonexistent") is None

    def test_get_by_turn(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="m1", content="A", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=1))
        store.inject(MemoryEntry(id="m2", content="B", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=3))
        store.inject(MemoryEntry(id="m3", content="C", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=5))
        assert len(store.get_by_turn(2)) == 1
        assert len(store.get_by_turn(3)) == 2
        assert len(store.get_by_turn(10)) == 3

    def test_get_by_type(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="m1", content="A", memory_type=MemoryType.EPISODIC,
                                  injected_at_turn=0))
        store.inject(MemoryEntry(id="m2", content="B", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0))
        store.inject(MemoryEntry(id="m3", content="C", memory_type=MemoryType.EPISODIC,
                                  injected_at_turn=0))
        assert len(store.get_by_type(MemoryType.EPISODIC)) == 2
        assert len(store.get_by_type(MemoryType.SEMANTIC)) == 1
        assert len(store.get_by_type(MemoryType.WORKING)) == 0

    def test_by_type_property(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="m1", content="A", memory_type=MemoryType.EPISODIC,
                                  injected_at_turn=0))
        store.inject(MemoryEntry(id="m2", content="B", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0))
        counts = store.by_type
        assert counts["episodic"] == 1
        assert counts["semantic"] == 1


class TestMemoryContext:
    def test_default_context(self):
        ctx = MemoryContext(turn=3, active_memories=["m1", "m2"])
        assert ctx.turn == 3
        assert ctx.active_memories == ["m1", "m2"]
        assert ctx.recalled_correctly == []
        assert ctx.forgotten == []


class TestMemoryProfiles:
    def test_all_profiles_defined(self):
        assert len(MEMORY_PROFILES) == 4
        for p in MEMORY_PROFILES:
            assert 0.0 <= p.base_recall_prob <= 1.0
            assert p.decay_per_turn >= 0.0
            assert p.confusion_rate >= 0.0
            assert p.hallucination_rate >= 0.0

    def test_get_memory_profile(self):
        assert get_memory_profile("perfect") is PERFECT_MEMORY
        assert get_memory_profile("production") is PRODUCTION_MEMORY
        assert get_memory_profile("development") is DEVELOPMENT_MEMORY
        assert get_memory_profile("degraded") is DEGRADED_MEMORY
        assert get_memory_profile("nonexistent") is None

    def test_perfect_profile_values(self):
        assert PERFECT_MEMORY.base_recall_prob == 1.0
        assert PERFECT_MEMORY.decay_per_turn == 0.0
        assert PERFECT_MEMORY.hallucination_rate == 0.0

    def test_production_memory_bounds(self):
        assert PRODUCTION_MEMORY.base_recall_prob > DEVELOPMENT_MEMORY.base_recall_prob
        assert PRODUCTION_MEMORY.decay_per_turn < DEVELOPMENT_MEMORY.decay_per_turn


# ── SimulatedMemoryAgent tests ─────────────────────────────────────────

class TestSimulatedMemoryAgent:
    def test_create_agent(self):
        store = MemoryStore()
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store)
        assert agent.profile is PRODUCTION_MEMORY
        assert agent.store is store

    def test_recall_perfect_memory(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="m1", content="The sky is blue",
                                  memory_type=MemoryType.SEMANTIC, injected_at_turn=0))
        agent = SimulatedMemoryAgent(profile=PERFECT_MEMORY, store=store, seed=42)
        result = agent.recall("m1", turn=5, task_id="test-1")
        assert result.correct is True
        assert result.target_memory_id == "m1"
        assert result.memory_type == MemoryType.SEMANTIC
        assert "sky is blue" in result.recalled_content.lower()

    def test_recall_nonexistent(self):
        store = MemoryStore()
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store)
        result = agent.recall("nonexistent", turn=0, task_id="test-1")
        assert result.correct is False
        assert result.error is not None

    def test_recall_degraded_memory(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="m1", content="Important configuration",
                                  memory_type=MemoryType.SEMANTIC, injected_at_turn=0))
        agent = SimulatedMemoryAgent(profile=DEGRADED_MEMORY, store=store, seed=42)
        # Test at turn 20 (heavily degraded)
        results = []
        for _ in range(10):
            r = agent.recall("m1", turn=20, task_id="test-2")
            results.append(r)
        # At least some failures expected with degraded profile at high turn
        failures = sum(1 for r in results if not r.correct)
        assert failures > 0, "Degraded profile should produce some failures"

    def test_recall_all(self):
        store = MemoryStore()
        for i in range(3):
            store.inject(MemoryEntry(
                id=f"m{i}", content=f"Fact {i}",
                memory_type=MemoryType.SEMANTIC, injected_at_turn=i,
            ))
        agent = SimulatedMemoryAgent(profile=PERFECT_MEMORY, store=store, seed=42)
        results = agent.recall_all(turn=2, task_id="test")
        assert len(results) == 3
        assert all(r.correct for r in results)

    def test_recall_all_by_type(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="e1", content="Event", memory_type=MemoryType.EPISODIC,
                                  injected_at_turn=0))
        store.inject(MemoryEntry(id="s1", content="Fact", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0))
        agent = SimulatedMemoryAgent(profile=PERFECT_MEMORY, store=store, seed=42)
        episodic = agent.recall_all(turn=0, task_id="test", memory_type=MemoryType.EPISODIC)
        assert len(episodic) == 1
        assert episodic[0].target_memory_id == "e1"

    def test_run_conversation(self):
        turns = [
            {"inject": [
                MemoryEntry(id="m1", content="Fact A", memory_type=MemoryType.SEMANTIC, injected_at_turn=0),
            ], "recall": []},
            {"inject": [], "recall": ["m1"]},
        ]
        agent = SimulatedMemoryAgent(profile=PERFECT_MEMORY, seed=42)
        results = agent.run_conversation(turns, task_id="conv-1")
        assert len(results) == 1
        assert results[0].correct is True

    def test_deterministic_seed(self):
        store = MemoryStore()
        store.inject(MemoryEntry(id="m1", content="Test", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0))
        agent1 = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store, seed=42)
        agent2 = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store, seed=42)
        r1 = agent1.recall("m1", turn=5, task_id="test")
        r2 = agent2.recall("m1", turn=5, task_id="test")
        assert r1.correct == r2.correct
        assert r1.confidence == r2.confidence
        assert r1.latency_ms == r2.latency_ms

    def test_reset(self):
        store = MemoryStore()
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store)
        store.inject(MemoryEntry(id="m1", content="Test", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0))
        assert agent.store.total == 1
        agent.reset()
        assert agent.store.total == 0

    def test_importance_affects_recall(self):
        """High-importance memories should be recalled more often than low-importance."""
        store = MemoryStore()
        store.inject(MemoryEntry(id="high", content="Critical info", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0, importance=1.0))
        store.inject(MemoryEntry(id="low", content="Trivial info", memory_type=MemoryType.SEMANTIC,
                                  injected_at_turn=0, importance=0.1))
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store, seed=42)
        high_correct = 0
        low_correct = 0
        for _ in range(50):
            r_high = agent.recall("high", turn=5, task_id="test")
            r_low = agent.recall("low", turn=5, task_id="test")
            if r_high.correct:
                high_correct += 1
            if r_low.correct:
                low_correct += 1
        # High-importance should have >= success rate
        assert high_correct >= low_correct, f"high={high_correct}, low={low_correct}"


# ── MemoryEvaluator tests ──────────────────────────────────────────────

class TestMemoryEvaluator:
    def test_compute_metrics_empty(self):
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics([])
        assert metrics.total_tests == 0
        assert metrics.memory_f1 == 0.0

    def test_compute_metrics_perfect(self):
        results = [
            MemoryRecallResult(
                task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC,
                target_content="Fact", recalled_content="Fact",
                correct=True, partial=False, confidence=0.95, latency_ms=100,
            ),
            MemoryRecallResult(
                task_id="t", turn=1, target_memory_id="m2",
                memory_type=MemoryType.SEMANTIC,
                target_content="Other", recalled_content="Other",
                correct=True, partial=False, confidence=0.90, latency_ms=120,
            ),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.total_tests == 2
        assert metrics.correct_recalls == 2
        assert metrics.recall_precision == 1.0
        assert metrics.recall_rate == 1.0
        assert metrics.memory_f1 == 1.0
        assert metrics.avg_latency_ms == 110.0
        assert metrics.avg_confidence == 0.925

    def test_compute_metrics_failures(self):
        results = [
            MemoryRecallResult(
                task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC,
                target_content="Fact", recalled_content="",
                correct=False, partial=False, confidence=0.1, latency_ms=200,
            ),
            MemoryRecallResult(
                task_id="t", turn=0, target_memory_id="m2",
                memory_type=MemoryType.SEMANTIC,
                target_content="Other", recalled_content="Fabricated",
                correct=False, partial=False, confidence=0.7, latency_ms=200,
            ),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.total_tests == 2
        assert metrics.correct_recalls == 0
        assert metrics.hallucinations == 1  # one had content but was wrong
        assert metrics.recall_precision == 0.0
        assert metrics.memory_f1 == 0.0

    def test_compute_metrics_partial(self):
        results = [
            MemoryRecallResult(
                task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC,
                target_content="Original fact", recalled_content="Original fact",
                correct=True, partial=False, confidence=0.9, latency_ms=100,
            ),
            MemoryRecallResult(
                task_id="t", turn=1, target_memory_id="m2",
                memory_type=MemoryType.SEMANTIC,
                target_content="Another fact", recalled_content="fact Another",
                correct=False, partial=True, confidence=0.6, latency_ms=150,
            ),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.correct_recalls == 1
        assert metrics.partial_recalls == 1
        assert metrics.recall_rate == 0.75  # 1 + 0.5 / 2
        assert 0.5 < metrics.memory_f1 < 0.9

    def test_per_type_breakdown(self):
        results = [
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="e1",
                memory_type=MemoryType.EPISODIC, target_content="Event",
                recalled_content="Event", correct=True, partial=False,
                confidence=0.9, latency_ms=100),
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="s1",
                memory_type=MemoryType.SEMANTIC, target_content="Fact",
                recalled_content="", correct=False, partial=False,
                confidence=0.1, latency_ms=100),
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="w1",
                memory_type=MemoryType.WORKING, target_content="Task",
                recalled_content="Task", correct=True, partial=False,
                confidence=0.9, latency_ms=100),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.episodic_precision == 1.0
        assert metrics.semantic_precision == 0.0
        assert metrics.working_precision == 1.0

    def test_degradation_analysis(self):
        results = []
        for turn in range(20):
            correct = turn <= 8  # early turns correct, late turns wrong
            results.append(MemoryRecallResult(
                task_id="t", turn=turn, target_memory_id=f"m{turn}",
                memory_type=MemoryType.SEMANTIC,
                target_content="Fact", recalled_content="Fact" if correct else "",
                correct=correct, partial=False,
                confidence=0.8, latency_ms=100,
            ))
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.early_turn_accuracy > 0.8
        assert metrics.late_turn_accuracy < 0.5
        assert metrics.degradation_rate > 0

    def test_generate_report(self):
        results = [
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC, target_content="Fact",
                recalled_content="Fact", correct=True, partial=False,
                confidence=0.95, latency_ms=100),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        report = evaluator.generate_report(metrics, results, "production")
        assert report.title == "Agent Memory Evaluation Report"
        assert report.profile_name == "production"
        assert "excellent" in report.summary.lower() or "good" in report.summary.lower()

    def test_report_to_dict(self):
        results = [
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC, target_content="Fact",
                recalled_content="Fact", correct=True, partial=False,
                confidence=0.95, latency_ms=100),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        report = evaluator.generate_report(metrics, results, "production")
        d = report.to_dict()
        assert d["title"] == "Agent Memory Evaluation Report"
        assert "metrics" in d
        assert d["metrics"]["total_tests"] == 1

    def test_report_to_markdown(self):
        results = [
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC, target_content="Fact",
                recalled_content="Fact", correct=True, partial=False,
                confidence=0.95, latency_ms=100),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        report = evaluator.generate_report(metrics, results, "production")
        md = report.to_markdown()
        assert "# Agent Memory Evaluation Report" in md
        assert "## Memory Metrics" in md
        assert "Memory F1" in md


# ── Benchmark tasks tests ──────────────────────────────────────────────

class TestMemoryBenchmarks:
    def test_all_benchmarks_defined(self):
        assert len(ALL_MEMORY_BENCHMARKS) == 5
        benchmark_ids = [b.id for b in ALL_MEMORY_BENCHMARKS]
        assert "episodic-memory" in benchmark_ids
        assert "semantic-memory" in benchmark_ids
        assert "working-memory" in benchmark_ids
        assert "cross-conversation" in benchmark_ids
        assert "memory-degradation" in benchmark_ids

    def test_episodic_benchmark_has_turns(self):
        bench = ALL_MEMORY_BENCHMARKS[0]
        assert bench.id == "episodic-memory"
        assert len(bench.turns) > 5
        # Should inject some memories
        injections = sum(
            len(t.get("inject", [])) for t in bench.turns
        )
        assert injections > 0

    def test_semantic_benchmark_runs(self):
        bench = ALL_MEMORY_BENCHMARKS[1]
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, seed=42)
        results = agent.run_conversation(bench.turns, task_id="semantic-test")
        assert len(results) > 0
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.total_tests > 0

    def test_working_benchmark_runs(self):
        bench = ALL_MEMORY_BENCHMARKS[2]
        agent = SimulatedMemoryAgent(profile=PERFECT_MEMORY, seed=42)
        results = agent.run_conversation(bench.turns, task_id="working-test")
        # With perfect memory, all recalls should be correct
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        assert metrics.correct_recalls == metrics.total_tests

    def test_degradation_benchmark_shows_drop(self):
        bench = ALL_MEMORY_BENCHMARKS[4]
        agent = SimulatedMemoryAgent(profile=DEGRADED_MEMORY, seed=42)
        results = agent.run_conversation(bench.turns, task_id="degradation-test")
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        # Degraded memory should show some failures
        assert metrics.correct_recalls < metrics.total_tests

    def test_all_benchmarks_with_perfect_memory(self):
        """All benchmarks should score 100% with perfect memory."""
        for bench in ALL_MEMORY_BENCHMARKS:
            agent = SimulatedMemoryAgent(profile=PERFECT_MEMORY, seed=42)
            results = agent.run_conversation(bench.turns, task_id=f"perfect-{bench.id}")
            evaluator = MemoryEvaluator()
            metrics = evaluator.compute_metrics(results)
            assert metrics.correct_recalls == metrics.total_tests, \
                f"Benchmark {bench.id}: {metrics.correct_recalls}/{metrics.total_tests} correct"

    def test_cross_profile_comparison(self):
        """All profiles should produce different results on the same benchmark."""
        bench = ALL_MEMORY_BENCHMARKS[1]  # semantic
        scores = {}
        for profile in MEMORY_PROFILES:
            agent = SimulatedMemoryAgent(profile=profile, seed=42)
            results = agent.run_conversation(bench.turns, task_id=f"compare-{profile.name}")
            evaluator = MemoryEvaluator()
            metrics = evaluator.compute_metrics(results)
            scores[profile.name] = metrics.memory_f1
        # Perfect > Production > Development > Degraded
        assert scores["perfect"] > scores["production"] > scores["development"] > scores["degraded"], \
            f"Scores out of order: {scores}"


# ── Integration tests ──────────────────────────────────────────────────

class TestMemoryIntegration:
    def test_full_pipeline(self):
        """End-to-end: store → inject → recall → evaluate → report."""
        store = MemoryStore()
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store, seed=42)
        evaluator = MemoryEvaluator()

        # Inject memories across turns
        for turn in range(5):
            store.inject(MemoryEntry(
                id=f"fact-{turn}",
                content=f"Configuration parameter {turn}",
                memory_type=MemoryType.SEMANTIC,
                injected_at_turn=turn,
                importance=0.8,
            ))

        # Recall at various turns
        results = []
        for turn in range(3, 10):
            results.extend(agent.recall_all(turn=turn, task_id=f"full-{turn}"))

        metrics = evaluator.compute_metrics(results)
        report = evaluator.generate_report(metrics, results, "production")

        assert metrics.total_tests > 0
        assert metrics.memory_f1 > 0
        assert len(report.summary) > 0
        md = report.to_markdown()
        assert "Memory Metrics" in md
        assert "Memory Type" in md or "Per-Type" in md

    def test_memory_metrics_serializable(self):
        """All metric objects should be JSON-serializable via to_dict."""
        results = [
            MemoryRecallResult(task_id="t", turn=0, target_memory_id="m1",
                memory_type=MemoryType.SEMANTIC, target_content="Test",
                recalled_content="Test", correct=True, partial=False,
                confidence=0.9, latency_ms=100),
        ]
        evaluator = MemoryEvaluator()
        metrics = evaluator.compute_metrics(results)
        report = evaluator.generate_report(metrics, results, "production")
        d = report.to_dict()
        # Should not raise
        import json
        json.dumps(d)
