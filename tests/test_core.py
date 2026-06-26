"""
Tests for the AgentOps Reliability Platform.

Run with: pytest tests/ -v
"""

import pytest

from agentops.retrieval.ingest import DocumentIngestor, DocumentChunk
from agentops.retrieval.engine import RetrievalEngine


class TestDocumentIngestor:
    """Test document ingestion and chunking."""

    def test_ingest_simple_text(self):
        ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=20)
        chunks = ingestor.ingest_text(
            "This is a test document. It has multiple sentences. "
            "Each sentence should be preserved during chunking. "
            "The chunker respects paragraph boundaries too."
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.chunk_id
            assert chunk.content
            assert chunk.source

    def test_ingest_directory(self, tmp_path):
        # Create test docs
        (tmp_path / "test.md").write_text("# Test\n\nThis is test content.\n\nMore content here.")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "inner.md").write_text("# Inner\n\nInner document content.")

        ingestor = DocumentIngestor(chunk_size=200, chunk_overlap=30)
        chunks = ingestor.ingest_directory(str(tmp_path))
        assert len(chunks) >= 2

    def test_chunk_metadata(self):
        ingestor = DocumentIngestor(chunk_size=200)
        chunks = ingestor.ingest_text("Test content", source="test.md", source_title="Test Doc")
        assert len(chunks) == 1
        assert chunks[0].source == "test.md"
        assert chunks[0].source_title == "Test Doc"
        assert chunks[0].chunk_index == 0

    def test_chunk_id_stable(self):
        ingestor = DocumentIngestor()
        chunks1 = ingestor.ingest_text("Hello world", source="a.md")
        chunks2 = ingestor.ingest_text("Hello world", source="a.md")
        assert chunks1[0].chunk_id == chunks2[0].chunk_id


class TestRetrievalEngine:
    """Test the retrieval engine."""

    def test_index_and_search_bm25(self):
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest_text(
            "CloudDeploy supports rolling, blue-green, and canary deployment strategies. "
            "Deployment strategy configuration controls how releases are rolled out. "
            "To enable 2FA, go to Settings then Security.",
            source="docs.md"
        )
        engine = RetrievalEngine(use_dense=False)
        engine.index(chunks)
        assert engine.ready
        assert engine.chunk_count == 1

        results = engine.search("deployment strategy", k=3)
        assert len(results) >= 1
        assert results[0].retrieval_method == "bm25"

    def test_search_empty_index(self):
        engine = RetrievalEngine(use_dense=False)
        results = engine.search("test", k=5)
        assert results == []

    def test_clear_engine(self):
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest_text("Test content", source="test.md")
        engine = RetrievalEngine(use_dense=False)
        engine.index(chunks)
        assert engine.ready
        engine.clear()
        assert not engine.ready
        assert engine.chunk_count == 0


class TestToolRegistry:
    """Test the tool registry."""

    def test_register_and_invoke(self):
        from agentops.agent.tool_registry import ToolRegistry, ToolDefinition

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="greet",
            description="Greet someone",
            parameters={"name": {"type": "string", "description": "Name to greet"}},
            required=["name"],
            fn=lambda name: f"Hello, {name}!",
        ))

        result = registry.invoke("greet", {"name": "World"})
        assert result.success
        assert "World" in str(result.output)

    def test_validation_error(self):
        from agentops.agent.tool_registry import ToolRegistry, ToolDefinition, ToolErrorType

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="greet",
            description="Greet someone",
            parameters={"name": {"type": "string"}},
            required=["name"],
            fn=lambda name: f"Hello, {name}!",
        ))

        result = registry.invoke("greet", {})
        assert not result.success
        assert result.error_type == ToolErrorType.VALIDATION

    def test_tool_not_found(self):
        from agentops.agent.tool_registry import ToolRegistry, ToolErrorType

        registry = ToolRegistry()
        result = registry.invoke("nonexistent", {})
        assert not result.success
        assert result.error_type == ToolErrorType.NOT_FOUND

    def test_summary(self):
        from agentops.agent.tool_registry import ToolRegistry, ToolDefinition

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="echo",
            description="Echo",
            parameters={"text": {"type": "string"}},
            required=["text"],
            fn=lambda text: text,
        ))
        registry.invoke("echo", {"text": "hello"})
        registry.invoke("echo", {"text": "world"})

        summary = registry.summary()
        assert summary["total_calls"] == 2
        assert summary["success_rate"] == 1.0


class TestTraceStore:
    """Test the SQLite trace store."""

    def test_save_and_query(self):
        from agentops.tracing.store import TraceStore
        from agentops.agent.implementations import AgentRunResult

        store = TraceStore(":memory:")
        result = AgentRunResult(
            task_id="t1", task="Test", final_answer="Answer",
            success=True, error=None, total_latency_ms=1000,
            verification_passed=True, verification_notes="ok",
            grounded_claims=["c1"], ungrounded_claims=[],
            citations_used=["c:0:abc"], plan=["step 1"],
            tool_calls_count=1, retrieved_chunks_count=2,
            reliability_trace=[{"step_name": "test"}],
        )
        store.save(result)

        traces = store.query(verification_passed=True)
        assert len(traces) == 1
        assert traces[0].task_id == "t1"
        assert traces[0].verification_passed

    def test_stats(self):
        from agentops.tracing.store import TraceStore
        from agentops.agent.implementations import AgentRunResult

        store = TraceStore(":memory:")
        for i in range(3):
            store.save(AgentRunResult(
                task_id=f"t{i}", task="T", final_answer="A",
                success=i < 2, error="err" if i == 2 else None,
                total_latency_ms=100 * (i + 1),
                verification_passed=i < 2, verification_notes="",
                grounded_claims=["c"], ungrounded_claims=[] if i < 2 else ["u"],
                citations_used=[], plan=[], tool_calls_count=0,
                retrieved_chunks_count=1, reliability_trace=[],
            ))

        stats = store.stats()
        assert stats["total_runs"] == 3
        assert stats["verification_pass_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["failure_rate"] == pytest.approx(1 / 3, abs=0.01)


class TestFailureClassifier:
    """Test failure classification."""

    def test_no_failures(self):
        from agentops.tracing.classifier import FailureClassifier
        from agentops.agent.implementations import AgentRunResult

        result = AgentRunResult(
            task_id="t1", task="T", final_answer="A",
            success=True, error=None, total_latency_ms=100,
            verification_passed=True, verification_notes="",
            grounded_claims=["c"], ungrounded_claims=[],
            citations_used=["c:0:a"], plan=["Step 1: Retrieve documentation"],
            tool_calls_count=0, retrieved_chunks_count=1,
            reliability_trace=[],
        )

        classifier = FailureClassifier()
        patterns = classifier.classify([result])
        assert len(patterns) == 0

    def test_hallucination_detected(self):
        from agentops.tracing.classifier import FailureClassifier
        from agentops.agent.implementations import AgentRunResult

        result = AgentRunResult(
            task_id="t1", task="T", final_answer="A",
            success=True, error=None, total_latency_ms=100,
            verification_passed=False, verification_notes="",
            grounded_claims=["c"], ungrounded_claims=["u1", "u2"],
            citations_used=[], plan=["s1"],
            tool_calls_count=0, retrieved_chunks_count=1,
            reliability_trace=[],
        )

        classifier = FailureClassifier()
        patterns = classifier.classify([result])
        names = [p.pattern_name for p in patterns]
        assert "hallucination" in names
        assert "verification_failure" in names


class TestEvalMetrics:
    """Test evaluation metrics."""

    def test_compute_metrics(self):
        from agentops.evals.metrics import compute_metrics
        from agentops.agent.implementations import AgentRunResult

        result = AgentRunResult(
            task_id="t1", task="Enable 2FA", final_answer="Go to Settings, then Security to enable 2FA",
            success=True, error=None, total_latency_ms=2000,
            verification_passed=True, verification_notes="",
            grounded_claims=["Settings", "Security", "2FA"],
            ungrounded_claims=[],
            citations_used=["c:0:a", "c:1:b"],
            plan=["Step 1: Find 2FA documentation"],
            tool_calls_count=0, retrieved_chunks_count=3,
            reliability_trace=[],
        )

        metrics = compute_metrics(result, key_terms=["2FA", "Settings", "Security"])
        assert metrics.groundedness == 1.0
        assert metrics.verification_passed
        assert metrics.citation_precision == 2 / 3
        assert metrics.answer_completeness == 1.0

    def test_compute_metrics_with_hallucination(self):
        from agentops.evals.metrics import compute_metrics
        from agentops.agent.implementations import AgentRunResult

        result = AgentRunResult(
            task_id="t1", task="T", final_answer="Some answer",
            success=True, error=None, total_latency_ms=1000,
            verification_passed=False, verification_notes="",
            grounded_claims=["c1"], ungrounded_claims=["u1", "u2", "u3"],
            citations_used=[],
            plan=[], tool_calls_count=0, retrieved_chunks_count=0,
            reliability_trace=[],
        )

        metrics = compute_metrics(result)
        assert metrics.groundedness == 1 / 4  # 1 grounded out of 4 total
        assert not metrics.verification_passed


class TestBenchmarks:
    """Test benchmark definitions."""

    def test_all_benchmarks_have_tasks(self):
        from agentops.evals.benchmarks import ALL_BENCHMARKS

        for bench in ALL_BENCHMARKS:
            assert len(bench.tasks) > 0, f"{bench.name} has no tasks"
            for task in bench.tasks:
                assert task.id
                assert task.question
                assert task.category

    def test_get_benchmark(self):
        from agentops.evals.benchmarks import get_benchmark

        bench = get_benchmark("support-tickets")
        assert bench is not None
        assert len(bench.tasks) == 5

        assert get_benchmark("nonexistent") is None
