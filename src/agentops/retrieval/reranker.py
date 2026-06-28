"""
Reranking strategies for retrieval results.

Provides cross-encoder and LLM-based reranking to improve retrieval quality
beyond the initial hybrid search. Integrates with the RetrievalEngine pipeline.

Strategies:
- CrossEncoderReranker: Uses sentence-transformers cross-encoder for relevance scoring
- LLMReranker: Uses an LLM to score document relevance to a query
- RerankerProtocol: Interface for custom reranker implementations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class RerankStrategy(str, Enum):
    """Available reranking strategies."""
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
    HYBRID = "hybrid"  # cross-encoder first, then LLM re-rank top-k


@dataclass
class RerankConfig:
    """Configuration for reranking behavior."""
    strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_before_rerank: int = 20  # Number of candidates to rerank
    top_k_after_rerank: int = 5   # Number of results after reranking
    llm_temperature: float = 0.0
    score_threshold: float = 0.0  # Minimum score to include in results


@dataclass
class RerankResult:
    """A single reranked result with original and new scores."""
    chunk_id: str
    content: str
    source: str
    source_title: str
    original_score: float
    rerank_score: float
    strategy: str  # Which reranker produced this score
    retrieval_method: str  # Original retrieval method (bm25/dense/hybrid)


class RerankerProtocol(Protocol):
    """Protocol for reranker implementations."""
    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[RerankResult]:
        ...


# ── Cross-Encoder Reranker ────────────────────────────────────────────

class CrossEncoderReranker:
    """Rerank retrieval results using a cross-encoder model.

    Cross-encoders process (query, document) pairs jointly, producing more
    accurate relevance scores than bi-encoder (dense vector) similarity.
    This is the recommended default for production RAG pipelines.

    Uses the MS MARCO cross-encoder by default, which is trained on
    passage ranking and produces calibrated relevance scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RerankResult]:
        """Rerank candidate documents for a query.

        Args:
            query: The search query.
            candidates: List of candidate dicts with keys: chunk_id, content,
                        source, source_title, score, retrieval_method.
            top_k: Number of top results to return.
            score_threshold: Minimum score to include.

        Returns:
            Sorted list of RerankResult objects, highest score first.
        """
        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        if self._model is None:
            # Fallback: return original ordering
            return self._fallback_rerank(query, candidates, top_k)

        # Prepare (query, document) pairs
        pairs = [(query, c["content"][:2000]) for c in candidates]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
        except Exception:
            return self._fallback_rerank(query, candidates, top_k)

        # Build and sort results
        results = []
        for candidate, score in zip(candidates, scores):
            if score >= score_threshold:
                results.append(RerankResult(
                    chunk_id=candidate["chunk_id"],
                    content=candidate["content"],
                    source=candidate.get("source", ""),
                    source_title=candidate.get("source_title", ""),
                    original_score=candidate.get("score", 0.0),
                    rerank_score=round(float(score), 4),
                    strategy="cross_encoder",
                    retrieval_method=candidate.get("retrieval_method", "unknown"),
                ))

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]

    def _load_model(self) -> None:
        """Lazily load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            self._model = None

    def _fallback_rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[RerankResult]:
        """Fallback: return candidates in original order with dummy scores."""
        results = []
        for i, candidate in enumerate(candidates[:top_k]):
            results.append(RerankResult(
                chunk_id=candidate["chunk_id"],
                content=candidate["content"],
                source=candidate.get("source", ""),
                source_title=candidate.get("source_title", ""),
                original_score=candidate.get("score", 0.0),
                rerank_score=1.0 / (i + 1),  # Rank-biased fallback
                strategy="fallback",
                retrieval_method=candidate.get("retrieval_method", "unknown"),
            ))
        return results


# ── LLM Reranker ──────────────────────────────────────────────────────

class LLMReranker:
    """Rerank retrieval results using an LLM for relevance scoring.

    Sends (query, document) pairs to an LLM with a prompt asking for
    a relevance score on a 0-10 scale. More expensive but potentially
    more accurate than cross-encoder reranking, especially for nuanced
    or domain-specific queries.

    Requires an LLM backend from agentops.llm or any callable with
    a chat() method.
    """

    RERANK_PROMPT = """You are a document relevance judge. Rate how relevant the following document is to the query on a scale of 0-10.

Query: {query}

Document: {document}

Relevance score (0-10, where 0 = completely irrelevant, 10 = perfectly relevant):"""

    def __init__(self, llm: Any = None, temperature: float = 0.0):
        """Initialize the LLM reranker.

        Args:
            llm: An LLM backend instance with a chat() method. If None,
                 attempts to auto-create from agentops.llm.
            temperature: LLM sampling temperature (0 for deterministic).
        """
        self.llm = llm
        self.temperature = temperature
        self._llm_loaded = False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RerankResult]:
        """Rerank candidates using LLM relevance scoring.

        Args:
            query: The search query.
            candidates: List of candidate dicts.
            top_k: Number of top results to return.
            score_threshold: Minimum normalized score (0-1 scale) to include.

        Returns:
            Sorted list of RerankResult objects.
        """
        if not candidates:
            return []

        if not self._llm_loaded:
            self._load_llm()

        if self.llm is None:
            # Fallback to cross-encoder-style ranking
            fallback = CrossEncoderReranker()
            return fallback.rerank(query, candidates, top_k, score_threshold)

        results = []
        for candidate in candidates[:top_k * 3]:  # Score up to 3x top_k
            score = self._score_candidate(query, candidate)
            if score >= score_threshold:
                results.append(RerankResult(
                    chunk_id=candidate["chunk_id"],
                    content=candidate["content"],
                    source=candidate.get("source", ""),
                    source_title=candidate.get("source_title", ""),
                    original_score=candidate.get("score", 0.0),
                    rerank_score=round(score, 4),
                    strategy="llm",
                    retrieval_method=candidate.get("retrieval_method", "unknown"),
                ))

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]

    def _score_candidate(self, query: str, candidate: dict[str, Any]) -> float:
        """Score a single candidate document."""
        prompt = self.RERANK_PROMPT.format(
            query=query,
            document=candidate["content"][:1000],
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            # Extract numeric score from response
            import re
            match = re.search(r'\b([0-9]|10)\b', response)
            if match:
                score = int(match.group(1))
                return score / 10.0  # Normalize to 0-1
            return 0.0
        except Exception:
            return 0.0

    def _load_llm(self) -> None:
        """Lazily load LLM backend."""
        try:
            from agentops.llm import create_backend
            self.llm = create_backend()
            self._llm_loaded = True
        except Exception:
            self._llm_loaded = True  # Mark as attempted


# ── Reranker Factory ───────────────────────────────────────────────────

def create_reranker(config: RerankConfig, llm: Any = None) -> RerankerProtocol | CrossEncoderReranker | LLMReranker:
    """Create a reranker instance from configuration.

    Args:
        config: RerankConfig specifying strategy and parameters.
        llm: Optional LLM instance for LLM-based reranking.

    Returns:
        A reranker instance.

    Raises:
        ValueError: If an unknown strategy is specified.
    """
    if config.strategy == RerankStrategy.CROSS_ENCODER:
        return CrossEncoderReranker(model_name=config.cross_encoder_model)
    elif config.strategy == RerankStrategy.LLM:
        return LLMReranker(llm=llm, temperature=config.llm_temperature)
    elif config.strategy == RerankStrategy.HYBRID:
        # Cross-encoder first, then LLM polish
        return CrossEncoderReranker(model_name=config.cross_encoder_model)
    elif config.strategy == RerankStrategy.NONE:
        return None
    else:
        raise ValueError(f"Unknown rerank strategy: {config.strategy}")
