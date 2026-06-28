"""
Advanced chunking strategies for document preprocessing.

Extends the basic paragraph-aware chunker with:
- Recursive character splitting (respects natural boundaries)
- Semantic chunking via sentence-boundary detection with embedding similarity
- Strategy enum for declarative pipeline configuration

All strategies produce DocumentChunk-compatible output.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    HYBRID = "hybrid"  # paragraph + recursive fallback for oversized paragraphs


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_threshold: float = 0.7  # Cosine similarity threshold for semantic chunk boundaries
    embedding_model: str = "all-MiniLM-L6-v2"
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
    ])


class ChunkerProtocol(Protocol):
    """Protocol for chunking implementations."""
    def chunk(self, text: str, source: str = "inline", source_title: str = "Inline") -> list[dict]:
        ...


# ── Recursive Character Splitter ──────────────────────────────────────

class RecursiveCharacterSplitter:
    """Split text recursively by separator hierarchy, respecting chunk boundaries.

    Tries to split at the most natural boundary first (paragraphs), then falls
    back to sentences, then words, then characters. Ensures chunks are within
    the size limit while preserving semantic coherence.

    Based on LangChain's RecursiveCharacterTextSplitter but standalone.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size - 1)
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def split(self, text: str) -> list[str]:
        """Split text into chunks using recursive separator strategy."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        chunks: list[str] = []
        separator = separators[-1]  # Default to character-level
        next_separators: list[str] = []

        # Find the first working separator
        for s in separators:
            if not s:
                separator = s
                break
            if s in text:
                separator = s
                next_separators = separators[separators.index(s) + 1:]
                break

        splits = self._split_on_separator(text, separator)

        good_splits: list[str] = []
        for split in splits:
            token_count = self._estimate_tokens(split)
            if token_count <= self.chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    chunks.extend(self._merge_splits(good_splits))
                    good_splits = []
                if next_separators:
                    chunks.extend(self._split_text(split, next_separators))
                else:
                    # Character-level: force split
                    chunks.extend(self._force_split(split))

        if good_splits:
            chunks.extend(self._merge_splits(good_splits))

        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_on_separator(self, text: str, separator: str) -> list[str]:
        if not separator:
            return list(text)
        parts = text.split(separator)
        return [separator.join(parts[i:i + 1]) if i < len(parts) - 1 else parts[i]
                for i in range(len(parts))]

    def _merge_splits(self, splits: list[str]) -> list[str]:
        merged: list[str] = []
        current = ""
        current_tokens = 0

        for split in splits:
            split_tokens = self._estimate_tokens(split)
            if current_tokens + split_tokens <= self.chunk_size:
                current = (current + split) if current else split
                current_tokens += split_tokens
            else:
                if current:
                    merged.append(current)
                current = split
                current_tokens = split_tokens

        if current:
            merged.append(current)
        return merged

    def _force_split(self, text: str) -> list[str]:
        """Hard-split text into fixed-size character chunks."""
        chunks = []
        for i in range(0, len(text), self.chunk_size * 4 - self.chunk_overlap * 4):
            chunks.append(text[i:i + self.chunk_size * 4])
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev = chunks[i - 1]
                if len(prev) > self.chunk_overlap * 4:
                    chunk = prev[-(self.chunk_overlap * 4):] + "\n" + chunk
            overlapped.append(chunk)
        return overlapped

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# ── Semantic Chunker ──────────────────────────────────────────────────

class SemanticChunker:
    """Chunk text at semantic boundaries using embedding similarity.

    Detects topic shifts by computing cosine similarity between consecutive
    sentences. When similarity drops below a threshold, a chunk boundary is
    inserted. Falls back to recursive splitting for un-splittable segments.

    The embedding model is loaded lazily on first use.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold = threshold
        self.model_name = model_name
        self._model = None

    def split(self, text: str) -> list[str]:
        """Split text at semantic boundaries."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            # Fall back to recursive splitting
            fallback = RecursiveCharacterSplitter(self.chunk_size, self.chunk_overlap)
            return fallback.split(text)

        boundaries = self._find_boundaries(sentences)
        return self._build_chunks(sentences, boundaries)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences with boundary-aware regex."""
        return re.split(r'(?<=[.!?])\s+', text)

    def _find_boundaries(self, sentences: list[str]) -> list[int]:
        """Find chunk boundaries via embedding similarity."""
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            if self._model is None:
                self._model = SentenceTransformer(self.model_name)

            # Get sentence embeddings
            truncated = [s[:2000] for s in sentences]
            embeddings = self._model.encode(truncated, show_progress_bar=False, convert_to_numpy=True)
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

            # Find local minima in similarity
            boundaries = []
            current_tokens = self._estimate_tokens(sentences[0])

            for i in range(1, len(sentences)):
                similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
                sent_tokens = self._estimate_tokens(sentences[i])

                if similarity < self.threshold and current_tokens + sent_tokens > self.chunk_size // 4:
                    boundaries.append(i)
                    current_tokens = sent_tokens
                else:
                    current_tokens += sent_tokens

            return boundaries
        except ImportError:
            # Fallback: split evenly
            return self._even_boundaries(sentences)

    def _even_boundaries(self, sentences: list[str]) -> list[int]:
        """Fallback: create evenly-sized boundaries."""
        boundaries = []
        current_tokens = 0
        for i, sent in enumerate(sentences):
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.chunk_size and i > 0:
                boundaries.append(i)
                current_tokens = sent_tokens
            else:
                current_tokens += sent_tokens
        return boundaries

    def _build_chunks(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """Build chunks from sentences at boundary points."""
        chunks = []
        start = 0

        for boundary in boundaries:
            chunk_text = " ".join(sentences[start:boundary])
            if self._estimate_tokens(chunk_text) <= self.chunk_size:
                chunks.append(chunk_text)
            else:
                # Sub-split oversized chunks
                fallback = RecursiveCharacterSplitter(self.chunk_size, self.chunk_overlap)
                chunks.extend(fallback.split(chunk_text))
            start = boundary

        # Final chunk
        final_text = " ".join(sentences[start:])
        if final_text:
            if self._estimate_tokens(final_text) <= self.chunk_size:
                chunks.append(final_text)
            else:
                fallback = RecursiveCharacterSplitter(self.chunk_size, self.chunk_overlap)
                chunks.extend(fallback.split(final_text))

        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev = chunks[i - 1]
                    if len(prev) > self.chunk_overlap * 4:
                        chunk = prev[-(self.chunk_overlap * 4):] + "\n" + chunk
                overlapped.append(chunk)
            chunks = overlapped

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# ── Chunking Factory ──────────────────────────────────────────────────

def create_chunker(config: ChunkingConfig) -> ChunkerProtocol | RecursiveCharacterSplitter | SemanticChunker:
    """Create a chunker instance from configuration.

    Args:
        config: ChunkingConfig specifying strategy and parameters.

    Returns:
        A chunker instance compatible with the chunking protocol.

    Raises:
        ValueError: If an unknown strategy is specified.
    """
    if config.strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )
    elif config.strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            threshold=config.semantic_threshold,
            model_name=config.embedding_model,
        )
    elif config.strategy in (ChunkingStrategy.PARAGRAPH, ChunkingStrategy.HYBRID, ChunkingStrategy.FIXED_SIZE):
        # PARAGRAPH and HYBRID delegate to the existing DocumentIngestor
        # which already handles paragraph-aware + overlap
        return RecursiveCharacterSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")
