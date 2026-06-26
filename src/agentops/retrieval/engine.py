"""
Hybrid retrieval engine: BM25 (lexical) + dense vector (semantic) search
with reciprocal rank fusion and citation-aware results.

Supports:
- BM25 via rank_bm25 for keyword matching
- Dense embeddings via sentence-transformers
- Hybrid fusion with configurable weights
- Citation metadata on every result
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .ingest import DocumentChunk


@dataclass
class RetrievalResult:
    """A single retrieval result with citation metadata."""
    chunk_id: str
    content: str
    source: str
    source_title: str
    score: float
    retrieval_method: Literal["bm25", "dense", "hybrid"]


class RetrievalEngine:
    """Hybrid retrieval engine for document search.

    Usage:
        engine = RetrievalEngine()
        engine.index(chunks)  # from DocumentIngestor

        results = engine.search("How do I reset my password?", k=5)
        for r in results:
            print(f"[{r.chunk_id}] ({r.source}) score={r.score:.3f}")
            print(f"  {r.content[:200]}...")
    """

    def __init__(
        self,
        dense_model: str = "all-MiniLM-L6-v2",
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        use_dense: bool = True,
    ):
        self.dense_model_name = dense_model
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.use_dense = use_dense

        self._chunks: list[DocumentChunk] = []
        self._bm25 = None
        self._dense_model = None
        self._dense_embeddings: np.ndarray | None = None
        self._initialized = False

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Index document chunks for retrieval."""
        if not chunks:
            return

        self._chunks = chunks
        self._build_bm25()
        if self.use_dense:
            self._build_dense()
        self._initialized = True

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Search across indexed chunks using hybrid retrieval."""
        if not self._initialized or not self._chunks:
            return []

        t0 = time.time()

        bm25_results = self._search_bm25(query, k * 2) if self._bm25 else []
        dense_results = self._search_dense(query, k * 2) if self.use_dense and self._dense_embeddings is not None else []

        if bm25_results and dense_results:
            results = self._fuse_results(bm25_results, dense_results, k)
        elif bm25_results:
            results = [(r[0], r[1], "bm25") for r in bm25_results[:k]]
        elif dense_results:
            results = [(r[0], r[1], "dense") for r in dense_results[:k]]
        else:
            # Fallback to substring matching
            results = self._fallback_search(query, k)

        return [
            RetrievalResult(
                chunk_id=self._chunks[idx].chunk_id,
                content=self._chunks[idx].content,
                source=self._chunks[idx].source,
                source_title=self._chunks[idx].source_title,
                score=round(score, 4),
                retrieval_method=method,
            )
            for idx, score, method in results
        ]

    def _build_bm25(self) -> None:
        """Build BM25 index from chunk texts."""
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [self._tokenize(c.content) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

    def _build_dense(self) -> None:
        """Build dense embeddings for all chunks."""
        try:
            from sentence_transformers import SentenceTransformer

            self._dense_model = SentenceTransformer(self.dense_model_name)
            texts = [c.content[:2000] for c in self._chunks]
            self._dense_embeddings = self._dense_model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
        except Exception:
            self._dense_model = None
            self._dense_embeddings = None

    def _search_bm25(self, query: str, k: int) -> list[tuple[int, float, str]]:
        """BM25 lexical search."""
        if not self._bm25:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i]), "bm25") for i in top_indices if scores[i] > 0]

    def _search_dense(self, query: str, k: int) -> list[tuple[int, float, str]]:
        """Dense vector similarity search."""
        if self._dense_model is None or self._dense_embeddings is None:
            return []
        query_vec = self._dense_model.encode([query[:2000]], show_progress_bar=False, convert_to_numpy=True)
        scores = np.dot(self._dense_embeddings, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i]), "dense") for i in top_indices if scores[i] > 0]

    def _fuse_results(
        self,
        bm25: list[tuple[int, float, str]],
        dense: list[tuple[int, float, str]],
        k: int,
    ) -> list[tuple[int, float, str]]:
        """Reciprocal rank fusion of BM25 and dense results."""
        # Normalize scores
        if bm25:
            max_bm25 = max(s[1] for s in bm25)
        else:
            max_bm25 = 1.0
        if dense:
            max_dense = max(s[1] for s in dense)
        else:
            max_dense = 1.0

        fused: dict[int, float] = {}
        for idx, score, _ in bm25:
            fused[idx] = fused.get(idx, 0) + self.bm25_weight * (score / max(max_bm25, 1e-9))
        for idx, score, _ in dense:
            fused[idx] = fused.get(idx, 0) + self.dense_weight * (score / max(max_dense, 1e-9))

        sorted_items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(idx, score, "hybrid") for idx, score in sorted_items if score > 0]

    def _fallback_search(self, query: str, k: int) -> list[tuple[int, float, str]]:
        """Simple substring fallback when indices are unavailable."""
        query_lower = query.lower()
        results = []
        for idx, chunk in enumerate(self._chunks):
            content_lower = chunk.content.lower()
            score = 0.0
            # Word overlap score
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = query_words & content_words
            if overlap:
                score = len(overlap) / len(query_words) if query_words else 0
                # Boost for exact substring matches
                if query_lower in content_lower:
                    score += 0.3
                results.append((idx, score, "bm25"))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def clear(self) -> None:
        """Clear all indices."""
        self._chunks = []
        self._bm25 = None
        self._dense_model = None
        self._dense_embeddings = None
        self._initialized = False

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def ready(self) -> bool:
        return self._initialized
