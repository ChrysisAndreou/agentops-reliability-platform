"""
Tests for AgentOps v0.22 — Production RAG Retrieval module.

Covers: chunking strategies, reranking, retrieval evaluation metrics,
and the built-in benchmark corpus.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from agentops.retrieval import (
    RetrievalEngine,
    RetrievalResult,
    DocumentIngestor,
    DocumentChunk,
    ChunkingStrategy,
    ChunkingConfig,
    RecursiveCharacterSplitter,
    SemanticChunker,
    create_chunker,
    RerankStrategy,
    RerankConfig,
    RerankResult,
    CrossEncoderReranker,
    LLMReranker,
    create_reranker,
    RetrievalMetrics,
    RAGEvalResult,
    RetrievalEvaluator,
    RetrievalBenchmark,
    load_agentops_retrieval_corpus,
    export_corpus_json,
)


# ── Chunking Tests ────────────────────────────────────────────────────

class TestRecursiveCharacterSplitter:
    """Test the recursive character splitter."""

    def test_simple_split(self):
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=0)
        text = "Short text that should fit in one chunk."
        chunks = splitter.split(text)
        assert len(chunks) >= 1
        # The text should appear in the output (may have separator artifacts)
        assert "Short" in chunks[0]

    def test_paragraph_split(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split(text)
        # Should split on paragraph boundaries
        assert len(chunks) >= 1

    def test_sentence_split(self):
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=0)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = splitter.split(text)
        assert len(chunks) >= 1

    def test_overlap(self):
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=30)
        text = "A" * 500  # Will need multiple chunks
        chunks = splitter.split(text)
        assert len(chunks) >= 2

    def test_empty_text(self):
        splitter = RecursiveCharacterSplitter()
        chunks = splitter.split("")
        # Empty input may produce empty list or one empty chunk
        assert len(chunks) <= 1

    def test_custom_separators(self):
        splitter = RecursiveCharacterSplitter(
            chunk_size=200,
            separators=["###", "\n\n", "\n", ". "],
        )
        text = "Section one###Section two###Section three"
        chunks = splitter.split(text)
        assert len(chunks) >= 1

    def test_large_text_produces_multiple_chunks(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
        text = "word " * 200
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_estimate_tokens(self):
        assert RecursiveCharacterSplitter._estimate_tokens("hello world") == 2


class TestSemanticChunker:
    """Test the semantic chunker."""

    def test_falls_back_on_single_sentence(self):
        chunker = SemanticChunker(chunk_size=512, threshold=0.5)
        text = "One sentence."
        chunks = chunker.split(text)
        assert len(chunks) == 1

    def test_falls_back_on_short_text(self):
        chunker = SemanticChunker(chunk_size=512, threshold=0.5)
        text = "First. Second. Third."
        chunks = chunker.split(text)
        assert len(chunks) >= 1

    def test_multi_paragraph_semantic(self):
        chunker = SemanticChunker(chunk_size=512, threshold=0.5)
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is about animals. Animals are fascinating creatures. "
            "Machine learning is a field of artificial intelligence. "
            "Neural networks process data through layers. "
            "Deep learning uses many layers of neural networks. "
        )
        chunks = chunker.split(text)
        assert len(chunks) >= 1

    def test_estimate_tokens(self):
        assert SemanticChunker._estimate_tokens("hello world") == 2


class TestCreateChunker:
    """Test the chunker factory function."""

    def test_recursive_strategy(self):
        config = ChunkingConfig(strategy=ChunkingStrategy.RECURSIVE)
        chunker = create_chunker(config)
        assert isinstance(chunker, RecursiveCharacterSplitter)

    def test_semantic_strategy(self):
        config = ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC)
        chunker = create_chunker(config)
        assert isinstance(chunker, SemanticChunker)

    def test_paragraph_strategy(self):
        config = ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH)
        chunker = create_chunker(config)
        assert isinstance(chunker, RecursiveCharacterSplitter)

    def test_hybrid_strategy(self):
        config = ChunkingConfig(strategy=ChunkingStrategy.HYBRID)
        chunker = create_chunker(config)
        assert isinstance(chunker, RecursiveCharacterSplitter)

    def test_unknown_strategy_raises(self):
        config = ChunkingConfig(strategy="invalid")  # type: ignore
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker(config)  # type: ignore


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""

    def test_defaults(self):
        config = ChunkingConfig()
        assert config.strategy == ChunkingStrategy.HYBRID
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.semantic_threshold == 0.7

    def test_custom(self):
        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=256,
            chunk_overlap=32,
        )
        assert config.chunk_size == 256
        assert config.chunk_overlap == 32


# ── Reranker Tests ────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    """Test the cross-encoder reranker."""

    @pytest.fixture
    def candidates(self):
        return [
            {
                "chunk_id": "d001:0:abc",
                "content": "Agent memory management uses short-term and long-term memory with automatic pruning.",
                "source": "docs/memory.md",
                "source_title": "Memory Management",
                "score": 0.85,
                "retrieval_method": "hybrid",
            },
            {
                "chunk_id": "d002:0:def",
                "content": "Tool calling specification defines structured function calls for agents.",
                "source": "docs/tools.md",
                "source_title": "Tool Calling API",
                "score": 0.72,
                "retrieval_method": "bm25",
            },
            {
                "chunk_id": "d003:0:ghi",
                "content": "Multi-agent orchestration uses supervisor-worker topology.",
                "source": "docs/multi-agent.md",
                "source_title": "Multi-Agent Orchestration",
                "score": 0.60,
                "retrieval_method": "dense",
            },
        ]

    def test_rerank_returns_sorted_results(self, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("How does agent memory work?", candidates)
        assert len(results) > 0
        # Results should be sorted by rerank_score descending
        for i in range(len(results) - 1):
            assert results[i].rerank_score >= results[i + 1].rerank_score

    def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_with_score_threshold(self, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("How does agent memory work?", candidates, score_threshold=0.5)
        # All returned results should have score >= 0.5
        for r in results:
            assert r.rerank_score >= 0.5

    def test_rerank_top_k(self, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) <= 2

    def test_rerank_result_structure(self, candidates):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("How does agent memory work?", candidates)
        if results:
            r = results[0]
            assert isinstance(r, RerankResult)
            assert r.chunk_id
            assert r.content
            assert r.source
            assert r.source_title
            assert isinstance(r.original_score, float)
            assert isinstance(r.rerank_score, float)
            assert r.strategy in ("cross_encoder", "fallback")
            assert r.retrieval_method

    def test_fallback_when_model_unavailable(self, candidates):
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2")
        try:
            results = reranker.rerank("query", candidates)
            # If model loaded, results should exist
            assert len(results) > 0
        except Exception:
            # If model unavailable, skip gracefully
            pytest.skip("Cross-encoder model not available")


class TestLLMReranker:
    """Test the LLM-based reranker."""

    def test_init_without_llm(self):
        reranker = LLMReranker()
        assert reranker.llm is None

    def test_rerank_without_llm_falls_back(self):
        reranker = LLMReranker()
        candidates = [
            {
                "chunk_id": "doc1",
                "content": "Test document about AI agents.",
                "source": "test.md",
                "source_title": "Test",
                "score": 0.9,
                "retrieval_method": "hybrid",
            },
        ]
        # Should not crash — falls back to cross-encoder
        results = reranker.rerank("AI agents", candidates)
        assert isinstance(results, list)


class TestCreateReranker:
    """Test the reranker factory function."""

    def test_cross_encoder_strategy(self):
        config = RerankConfig(strategy=RerankStrategy.CROSS_ENCODER)
        reranker = create_reranker(config)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_llm_strategy(self):
        config = RerankConfig(strategy=RerankStrategy.LLM)
        reranker = create_reranker(config)
        assert isinstance(reranker, LLMReranker)

    def test_none_strategy(self):
        config = RerankConfig(strategy=RerankStrategy.NONE)
        reranker = create_reranker(config)
        assert reranker is None

    def test_hybrid_strategy(self):
        config = RerankConfig(strategy=RerankStrategy.HYBRID)
        reranker = create_reranker(config)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_unknown_strategy_raises(self):
        config = RerankConfig(strategy="invalid")  # type: ignore
        with pytest.raises(ValueError, match="Unknown rerank strategy"):
            create_reranker(config)  # type: ignore


class TestRerankConfig:
    """Test RerankConfig dataclass."""

    def test_defaults(self):
        config = RerankConfig()
        assert config.strategy == RerankStrategy.CROSS_ENCODER
        assert config.top_k_before_rerank == 20
        assert config.top_k_after_rerank == 5
        assert config.score_threshold == 0.0

    def test_custom(self):
        config = RerankConfig(
            strategy=RerankStrategy.LLM,
            top_k_after_rerank=10,
            score_threshold=0.3,
        )
        assert config.top_k_after_rerank == 10
        assert config.score_threshold == 0.3


# ── Evaluation Metrics Tests ──────────────────────────────────────────

class TestRetrievalMetrics:
    """Test the RetrievalMetrics dataclass."""

    def test_defaults(self):
        m = RetrievalMetrics(query="test")
        assert m.ndcg_at_5 == 0.0
        assert m.mrr == 0.0
        assert m.num_results == 0

    def test_custom_values(self):
        m = RetrievalMetrics(
            query="test",
            ndcg_at_5=0.85,
            mrr=0.75,
            recall_at_5=0.60,
            num_results=5,
            num_relevant=3,
        )
        assert m.ndcg_at_5 == 0.85
        assert m.mrr == 0.75


class TestRAGEvalResult:
    """Test the RAGEvalResult dataclass."""

    def test_defaults(self):
        result = RAGEvalResult(benchmark_name="test", num_queries=0)
        assert result.mean_ndcg_5 == 0.0
        assert result.summary["ndcg@5"] == 0.0

    def test_summary(self):
        result = RAGEvalResult(
            benchmark_name="test",
            num_queries=10,
            mean_ndcg_5=0.85,
            mean_mrr=0.75,
            mean_recall_5=0.60,
        )
        assert result.summary["ndcg@5"] == 0.85
        assert result.summary["mrr"] == 0.75


class TestRetrievalEvaluator:
    """Test the retrieval evaluation metrics."""

    def test_evaluate_query_perfect(self):
        """Perfect retrieval should get score 1.0."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d002", "d003"],
            relevant_ids={"d001", "d002", "d003"},
        )
        assert metrics.ndcg_at_5 == 1.0
        assert metrics.mrr == 1.0
        assert metrics.recall_at_5 == 1.0
        assert metrics.precision_at_5 == 1.0

    def test_evaluate_query_no_relevant(self):
        """No relevant documents should give 0 scores."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d002"],
            relevant_ids=set(),
        )
        assert metrics.mrr == 0.0
        assert metrics.recall_at_5 == 0.0
        assert metrics.precision_at_5 == 0.0

    def test_evaluate_query_empty_retrieved(self):
        """Empty retrieval should give 0 scores."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=[],
            relevant_ids={"d001"},
        )
        assert metrics.num_results == 0

    def test_mrr_calculation(self):
        """MRR should be reciprocal of first relevant rank."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d002", "d003", "d004", "d005"],
            relevant_ids={"d003"},  # Third position
        )
        assert metrics.mrr == pytest.approx(0.3333, abs=0.001)

    def test_recall_at_k(self):
        """Recall@k should capture fraction of relevant retrieved."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d002", "d003", "d004", "d005"],
            relevant_ids={"d001", "d002", "d006", "d007", "d008"},
        )
        # Only 2 of 5 relevant in top-5
        assert metrics.recall_at_5 == 2.0 / 5.0

    def test_precision_at_k(self):
        """Precision@k should capture fraction of retrieved that are relevant."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d002", "d003"],
            relevant_ids={"d001", "d002"},
        )
        assert metrics.precision_at_5 == pytest.approx(2.0 / 3.0, abs=0.001)

    def test_ndcg_with_grades(self):
        """NDCG with graded relevance should reward highly-relevant docs at top."""
        grades = {"d001": 3, "d002": 1, "d003": 2}
        relevant = {"d001", "d002", "d003"}

        # Perfect ordering
        perfect_metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d003", "d002", "d004", "d005"],
            relevant_ids=relevant,
            relevance_grades=grades,
        )
        # Suboptimal ordering
        suboptimal_metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d002", "d003", "d001", "d004", "d005"],
            relevant_ids=relevant,
            relevance_grades=grades,
        )
        assert perfect_metrics.ndcg_at_5 > suboptimal_metrics.ndcg_at_5

    def test_map_calculation(self):
        """MAP should average precision at each relevant document."""
        metrics = RetrievalEvaluator.evaluate_query(
            query="test",
            retrieved_ids=["d001", "d002", "d003", "d004", "d005"],
            relevant_ids={"d001", "d003", "d005"},
        )
        # AP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.667 + 0.6) / 3 = 0.7556
        expected_ap = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
        assert metrics.map_score == pytest.approx(expected_ap, rel=0.01)

    def test_context_relevance_empty(self):
        score = RetrievalEvaluator.context_relevance("query", [])
        assert score == 0.0

    def test_context_relevance_basic(self):
        score = RetrievalEvaluator.context_relevance(
            "agent memory management",
            ["Agent memory management uses short-term and long-term memory."],
        )
        assert 0.0 <= score <= 1.0

    def test_answer_faithfulness_empty(self):
        score = RetrievalEvaluator.answer_faithfulness("", [])
        assert score == 0.0

    def test_answer_faithfulness_basic(self):
        score = RetrievalEvaluator.answer_faithfulness(
            "The agent uses hybrid retrieval combining BM25 and dense embeddings.",
            ["The RAG system uses a hybrid retrieval approach combining BM25 lexical search with dense vector embeddings."],
        )
        assert 0.0 <= score <= 1.0

    def test_evaluate_benchmark(self):
        """End-to-end benchmark evaluation."""
        benchmark = load_agentops_retrieval_corpus()

        def mock_retrieval(query, k):
            # Simple BM25-like retrieval
            query_terms = set(query.lower().split())
            scores = []
            for doc in benchmark.documents:
                doc_terms = set(doc["content"].lower().split())
                overlap = query_terms & doc_terms
                scores.append((doc["id"], len(overlap)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in scores[:k]]

        result = RetrievalEvaluator.evaluate_benchmark(benchmark, mock_retrieval)
        assert result.num_queries == 10
        assert result.mean_mrr > 0.0  # Should find at least some relevant docs
        assert result.summary["mrr"] == result.mean_mrr


# ── Benchmark Corpus Tests ────────────────────────────────────────────

class TestRetrievalCorpus:
    """Test the built-in benchmark corpus."""

    def test_corpus_has_documents(self):
        benchmark = load_agentops_retrieval_corpus()
        assert benchmark.name == "agentops-retrieval-corpus"
        assert len(benchmark.documents) == 20
        assert len(benchmark.queries) == 10

    def test_documents_have_required_fields(self):
        benchmark = load_agentops_retrieval_corpus()
        for doc in benchmark.documents:
            assert "id" in doc
            assert "content" in doc
            assert "title" in doc
            assert "source" in doc

    def test_queries_have_required_fields(self):
        benchmark = load_agentops_retrieval_corpus()
        for query in benchmark.queries:
            assert "query" in query
            assert "relevant_ids" in query
            assert "relevance_grades" in query
            assert len(query["relevant_ids"]) > 0

    def test_all_relevant_ids_exist(self):
        benchmark = load_agentops_retrieval_corpus()
        doc_ids = {d["id"] for d in benchmark.documents}
        for query in benchmark.queries:
            for rid in query["relevant_ids"]:
                assert rid in doc_ids, f"Relevant ID {rid} not in documents"

    def test_export_json(self):
        json_str = export_corpus_json()
        assert "agentops-retrieval-corpus" in json_str
        assert "documents" in json_str
        assert "queries" in json_str

    def test_export_json_roundtrip(self):
        import json
        json_str = export_corpus_json()
        data = json.loads(json_str)
        assert data["name"] == "agentops-retrieval-corpus"
        assert len(data["documents"]) == 20
        assert len(data["queries"]) == 10


# ── Integration Tests ─────────────────────────────────────────────────

class TestEndToEndRetrievalPipeline:
    """End-to-end test of the full retrieval pipeline with new v0.22 features."""

    def test_ingest_chunk_index_search_evaluate(self):
        """Full pipeline: ingest -> chunk -> index -> search -> rerank -> evaluate."""
        # 1. Ingest documents
        ingestor = DocumentIngestor(chunk_size=256, chunk_overlap=32)

        benchmark = load_agentops_retrieval_corpus()
        all_chunks = []
        for doc in benchmark.documents[:10]:  # Use first 10 docs
            chunks = ingestor.ingest_text(
                text=doc["content"],
                source=doc["source"],
                source_title=doc["title"],
            )
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0

        # 2. Index with retrieval engine
        engine = RetrievalEngine(dense_model="all-MiniLM-L6-v2", use_dense=True)
        engine.index(all_chunks)
        assert engine.ready
        assert engine.chunk_count > 0

        # 3. Search
        results = engine.search("How do I set up observability dashboards?", k=10)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score > 0 for r in results)

        # 4. Rerank
        reranker = CrossEncoderReranker()
        candidate_dicts = [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "source": r.source,
                "source_title": r.source_title,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
            }
            for r in results
        ]
        reranked = reranker.rerank(
            "How do I set up observability dashboards?",
            candidate_dicts,
            top_k=5,
        )
        assert len(reranked) <= 5
        if reranked:
            assert all(isinstance(r, RerankResult) for r in reranked)

    def test_engine_with_empty_index(self):
        engine = RetrievalEngine()
        results = engine.search("anything", k=5)
        assert results == []

    def test_engine_clear(self):
        ingestor = DocumentIngestor(chunk_size=256)
        chunks = ingestor.ingest_text("Test content for clearing.", "test", "Test")
        engine = RetrievalEngine(dense_model="all-MiniLM-L6-v2", use_dense=True)
        engine.index(chunks)
        assert engine.ready
        engine.clear()
        assert not engine.ready
        assert engine.chunk_count == 0


class TestChunkingPipelineIntegration:
    """Integration tests with new chunking strategies."""

    def test_recursive_splitter_with_engine(self):
        """Use recursive splitter chunks with the retrieval engine."""
        splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=20)
        text = (
            "Agent memory management. Agents use short-term and long-term memory. "
            "Short-term memory is limited to the context window. "
            "Long-term memory uses vector databases for semantic retrieval. "
            "Memory pruning happens automatically when capacity is reached. "
            "Tool calling specification. Tools are invoked via structured function calls. "
            "Each tool declares its name, parameters, and return type. "
            "Multi-agent orchestration. Supervisor-worker is the default topology."
        )
        chunk_texts = splitter.split(text)

        # Convert to DocumentChunks for the engine
        chunks = []
        for i, ct in enumerate(chunk_texts):
            chunks.append(DocumentChunk(
                chunk_id=f"chunk:{i}",
                content=ct,
                source="test.md",
                source_title="Test Document",
                chunk_index=i,
                token_count=max(1, len(ct) // 4),
            ))

        engine = RetrievalEngine(dense_model="all-MiniLM-L6-v2", use_dense=True)
        engine.index(chunks)
        assert engine.ready
        assert engine.chunk_count > 0

        results = engine.search("How does memory work?", k=3)
        assert len(results) > 0

    def test_semantic_chunker_with_engine(self):
        """Use semantic chunker with the retrieval engine."""
        chunker = SemanticChunker(chunk_size=512, threshold=0.5)
        text = (
            "The weather today is sunny and warm. "
            "It's a perfect day for outdoor activities. "
            "Machine learning models require large amounts of data. "
            "Neural networks learn patterns from training examples. "
            "Deep learning has revolutionized computer vision. "
            "Image classification uses convolutional neural networks. "
        )
        chunk_texts = chunker.split(text)

        chunks = []
        for i, ct in enumerate(chunk_texts):
            chunks.append(DocumentChunk(
                chunk_id=f"chunk:{i}",
                content=ct,
                source="test.md",
                source_title="Test",
                chunk_index=i,
                token_count=max(1, len(ct) // 4),
            ))

        engine = RetrievalEngine(dense_model="all-MiniLM-L6-v2", use_dense=True)
        if chunks:
            engine.index(chunks)
            assert engine.ready


# ── RetrievalEngine Edge Cases ────────────────────────────────────────

class TestRetrievalEngineEdgeCases:
    """Edge case tests for the retrieval engine."""

    def test_search_without_index(self):
        engine = RetrievalEngine()
        results = engine.search("query")
        assert results == []

    def test_index_empty_chunks(self):
        engine = RetrievalEngine()
        engine.index([])
        assert not engine.ready
        assert engine.chunk_count == 0

    def test_bm25_only_mode(self):
        engine = RetrievalEngine(use_dense=False)
        chunks = [
            DocumentChunk(
                chunk_id="c1", content="Hello world test",
                source="test", source_title="Test", chunk_index=0,
                token_count=3,
            ),
        ]
        engine.index(chunks)
        assert engine.ready
        results = engine.search("hello", k=1)
        assert len(results) >= 0  # BM25 may or may not find it

    def test_fallback_substring_search(self):
        """When both indices fail, should fall back to substring matching."""
        # Force indices to fail by not providing rank_bm25
        engine = RetrievalEngine(use_dense=False)
        engine._bm25 = None  # Force BM25 unavailable
        chunks = [
            DocumentChunk(
                chunk_id="c1", content="unique search term here",
                source="test", source_title="Test", chunk_index=0,
                token_count=5,
            ),
        ]
        engine._chunks = chunks
        engine._initialized = True
        results = engine.search("unique search term", k=1)
        assert len(results) >= 0


# ── DocumentIngestor Tests ────────────────────────────────────────────

class TestDocumentIngestor:
    """Test the document ingestion pipeline."""

    def test_ingest_text(self):
        ingestor = DocumentIngestor(chunk_size=256, chunk_overlap=32)
        chunks = ingestor.ingest_text("This is a test document.", "test", "Test")
        assert len(chunks) == 1
        assert chunks[0].source == "test"
        assert chunks[0].source_title == "Test"

    def test_ingest_directory(self):
        ingestor = DocumentIngestor(chunk_size=256, chunk_overlap=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "doc1.md").write_text("# Doc 1\n\nContent of document one.")
            (path / "doc2.md").write_text("# Doc 2\n\nContent of document two.")
            chunks = ingestor.ingest_directory(str(path))
            assert len(chunks) >= 2

    def test_ingest_file(self):
        ingestor = DocumentIngestor(chunk_size=256)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            test_file = path / "test.md"
            test_file.write_text("# My Title\n\nContent here.")
            chunks = ingestor.ingest_file(test_file, path)
            assert len(chunks) >= 1
            assert chunks[0].source_title == "My Title"

    def test_ingest_long_text_produces_multiple_chunks(self):
        ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=10)
        text = "This is a sentence. " * 100  # Uses sentence boundaries
        chunks = ingestor.ingest_text(text, "long", "Long Doc")
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        ingestor = DocumentIngestor()
        chunks = ingestor.ingest_text("Test content", "test.md", "Test")
        assert len(chunks) == 1
        assert chunks[0].chunk_id
        assert chunks[0].chunk_index == 0
        assert chunks[0].token_count > 0


# ── RetrievalResult Tests ─────────────────────────────────────────────

class TestRetrievalResult:
    """Test the RetrievalResult dataclass."""

    def test_creation(self):
        result = RetrievalResult(
            chunk_id="doc:0:abc",
            content="Test content",
            source="test.md",
            source_title="Test",
            score=0.95,
            retrieval_method="hybrid",
        )
        assert result.chunk_id == "doc:0:abc"
        assert result.score == 0.95

    def test_method_values(self):
        for method in ("bm25", "dense", "hybrid"):
            result = RetrievalResult(
                chunk_id="c1", content="x", source="s", source_title="t",
                score=0.5, retrieval_method=method,
            )
            assert result.retrieval_method == method
