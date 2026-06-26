"""
Document ingestion and chunking pipeline.

Transforms raw documents (Markdown, text) into queryable chunks
with metadata for citation tracking.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class DocumentChunk:
    """A single chunk of a document, ready for indexing."""
    chunk_id: str
    content: str
    source: str          # e.g., "product-docs/onboarding.md"
    source_title: str    # e.g., "Onboarding Guide"
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)

    def to_retrieval_result(self, score: float = 0.0, method: str = "unknown") -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "source_title": self.source_title,
            "score": score,
            "retrieval_method": method,
        }


class DocumentIngestor:
    """Ingests documents from a directory and produces chunked output.

    Usage:
        ingestor = DocumentIngestor(chunk_size=512, chunk_overlap=64)
        chunks = ingestor.ingest_directory("sample_data/docs/")
        for chunk in chunks:
            print(f"{chunk.chunk_id}: {chunk.content[:80]}...")
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_directory(self, directory: str | Path) -> list[DocumentChunk]:
        """Ingest all .md and .txt files from a directory tree."""
        directory = Path(directory)
        all_chunks = []
        for file_path in sorted(directory.rglob("*.md")):
            chunks = self.ingest_file(file_path, directory)
            all_chunks.extend(chunks)
        for file_path in sorted(directory.rglob("*.txt")):
            chunks = self.ingest_file(file_path, directory)
            all_chunks.extend(chunks)
        return all_chunks

    def ingest_file(self, file_path: Path, root_dir: Path | None = None) -> list[DocumentChunk]:
        """Ingest a single file into chunks."""
        content = file_path.read_text(encoding="utf-8", errors="replace")
        source = str(file_path.relative_to(root_dir)) if root_dir else file_path.name
        source_title = self._extract_title(content, file_path.stem)
        chunks = self._chunk_text(content)

        results = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = self._make_chunk_id(source, i, chunk_text)
            results.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                source=source,
                source_title=source_title,
                chunk_index=i,
                token_count=self._estimate_tokens(chunk_text),
                metadata={"file_path": str(file_path)},
            ))
        return results

    def ingest_text(self, text: str, source: str = "inline", source_title: str = "Inline Document") -> list[DocumentChunk]:
        """Ingest raw text into chunks."""
        chunks = self._chunk_text(text)
        results = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = self._make_chunk_id(source, i, chunk_text)
            results.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                source=source,
                source_title=source_title,
                chunk_index=i,
                token_count=self._estimate_tokens(chunk_text),
                metadata={},
            ))
        return results

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks, respecting paragraph boundaries."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens <= self.chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
                current_tokens += para_tokens
            else:
                if current:
                    chunks.append(current)
                # If a single paragraph exceeds chunk_size, split it
                if para_tokens > self.chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sub_chunk = ""
                    sub_tokens = 0
                    for sent in sentences:
                        sent_tokens = self._estimate_tokens(sent)
                        if sub_tokens + sent_tokens <= self.chunk_size:
                            sub_chunk = (sub_chunk + " " + sent).strip() if sub_chunk else sent
                            sub_tokens += sent_tokens
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = sent
                            sub_tokens = sent_tokens
                    current = sub_chunk
                    current_tokens = sub_tokens
                else:
                    current = para
                    current_tokens = para_tokens

        if current:
            chunks.append(current)

        # Create overlapping chunks for the last chunk_boundary
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev_end = chunks[i-1][-self.chunk_overlap:]
                    chunk = prev_end + "\n" + chunk
                overlapped.append(chunk)
            chunks = overlapped

        return chunks

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract a title from markdown content."""
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return fallback.replace("_", " ").title()

    def _make_chunk_id(self, source: str, index: int, text: str) -> str:
        """Create a stable chunk ID."""
        h = hashlib.md5(f"{source}:{index}:{text[:100]}".encode()).hexdigest()[:12]
        return f"{source}:{index}:{h}"

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token for English)."""
        return max(1, len(text) // 4)
