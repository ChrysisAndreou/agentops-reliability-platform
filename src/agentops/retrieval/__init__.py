"""
Hybrid retrieval engine with BM25 + dense vector search and citation tracking.

Provides document ingestion, chunking, dual-index retrieval (BM25 lexical
+ sentence-transformer dense), score fusion, and citation-aware result
formatting for the reliability agent's evidence grounding workflow.
"""

from .engine import RetrievalEngine, RetrievalResult
from .ingest import DocumentIngestor, DocumentChunk

__all__ = [
    "RetrievalEngine",
    "RetrievalResult",
    "DocumentIngestor",
    "DocumentChunk",
]
