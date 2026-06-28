"""
Hybrid retrieval engine with BM25 + dense vector search, citation tracking,
advanced chunking strategies, cross-encoder/LLM reranking, and BEIR-style
evaluation metrics.

v0.22 adds:
- Multiple chunking strategies (recursive, semantic, paragraph, hybrid)
- Cross-encoder and LLM-based reranking
- NDCG, MRR, Recall, Precision, MAP evaluation metrics
- RAG-specific context relevance and answer faithfulness scoring
- Built-in retrieval benchmark corpus (20 docs, 10 queries)
"""

from .engine import RetrievalEngine, RetrievalResult
from .ingest import DocumentChunk, DocumentIngestor
from .chunking import (
    ChunkingStrategy,
    ChunkingConfig,
    RecursiveCharacterSplitter,
    SemanticChunker,
    create_chunker,
)
from .reranker import (
    RerankStrategy,
    RerankConfig,
    RerankResult,
    CrossEncoderReranker,
    LLMReranker,
    create_reranker,
)
from .eval import (
    RetrievalMetrics,
    RAGEvalResult,
    RetrievalEvaluator,
    RetrievalBenchmark,
    load_agentops_retrieval_corpus,
    export_corpus_json,
)

__all__ = [
    # Engine
    "RetrievalEngine",
    "RetrievalResult",
    "DocumentIngestor",
    "DocumentChunk",
    # Chunking
    "ChunkingStrategy",
    "ChunkingConfig",
    "RecursiveCharacterSplitter",
    "SemanticChunker",
    "create_chunker",
    # Reranking
    "RerankStrategy",
    "RerankConfig",
    "RerankResult",
    "CrossEncoderReranker",
    "LLMReranker",
    "create_reranker",
    # Evaluation
    "RetrievalMetrics",
    "RAGEvalResult",
    "RetrievalEvaluator",
    "RetrievalBenchmark",
    "load_agentops_retrieval_corpus",
    "export_corpus_json",
]
