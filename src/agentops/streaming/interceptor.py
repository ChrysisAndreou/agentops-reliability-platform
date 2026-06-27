"""
Streaming interceptor — wraps LLM streaming output for real-time verification.

Intercepts streaming output from LLMs (SSE events, async generators, or
sync generators), extracts claims as they emerge, verifies each claim
against evidence, and can abort the stream mid-generation when hallucination
is detected.

Supports:
- Sync generators: Standard Python generators yielding text chunks.
- Async generators: Async generators (common in LLM SDKs).
- Callback mode: Push chunks manually via process_chunk().
- Simulated streams: For testing without a real LLM.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from typing import Any

from .claim_extractor import ClaimExtractor
from .state import (
    AbortReason,
    StreamChunk,
    StreamingConfig,
    StreamingRun,
)
from .verifier import StreamingVerifier


@dataclass
class StreamingInterceptor:
    """Intercepts and verifies streaming LLM output in real time.

    Wraps any streaming text source (sync/async generator, SSE callback)
    and provides verified output with optional early abort on hallucination.

    Usage:
        # Async mode (most common with LLM SDKs)
        interceptor = StreamingInterceptor(evidence=evidence_store)
        interceptor.start("run-1", "Explain deployment process")

        async for chunk in interceptor.wrap_async(llm_stream):
            if interceptor.is_aborted():
                print("Hallucination detected — stream aborted!")
                break
            print(chunk, end="")

        metrics = interceptor.get_metrics()

        # Sync mode
        for chunk in interceptor.wrap_sync(llm_stream):
            ...

        # Callback/direct mode
        interceptor.process_chunk("The system uses ")
        interceptor.process_chunk("Kubernetes for orchestration.")

    Attributes:
        config: Streaming verification configuration.
        evidence: Ground-truth evidence store.
        extractor: Claim extractor for parsing streaming text.
        verifier: Claim verifier for checking against evidence.
    """

    config: StreamingConfig = field(default_factory=StreamingConfig)
    evidence: dict[str, str] = field(default_factory=dict)

    extractor: ClaimExtractor | None = None
    verifier: StreamingVerifier | None = None

    # Run state
    _run: StreamingRun | None = None
    _aborted: bool = False
    _abort_reason: AbortReason = AbortReason.NONE
    _chunks_processed: int = 0
    _start_time: float = 0.0
    _accumulated_output: str = ""

    def __post_init__(self) -> None:
        self.extractor = ClaimExtractor(config=self.config)
        self.verifier = StreamingVerifier(config=self.config, evidence=self.evidence)

    # ── Public API ───────────────────────────────────────────────────

    def start(self, run_id: str | None = None, task: str = "") -> StreamingRun:
        """Start a new streaming verification run.

        Args:
            run_id: Unique run ID (auto-generated if not provided).
            task: Task description for context.

        Returns:
            The new StreamingRun.
        """
        if run_id is None:
            run_id = f"stream-{uuid.uuid4().hex[:12]}"

        self._start_time = time.time()
        self._chunks_processed = 0
        self._aborted = False
        self._abort_reason = AbortReason.NONE
        self._accumulated_output = ""

        # Reset extractor
        self.extractor = ClaimExtractor(config=self.config)

        # Create verifier with evidence
        if self.verifier is None:
            self.verifier = StreamingVerifier(config=self.config, evidence=self.evidence)
        self.verifier.set_evidence(self.evidence)
        self._run = self.verifier.start_run(run_id, task)

        return self._run

    def process_chunk(self, text: str, is_final: bool = False) -> str:
        """Process a single text chunk through verification.

        Extracts claims, verifies them, and checks abort conditions.
        Returns the chunk text unchanged (transparent passthrough) but
        records verification results.

        Args:
            text: Raw text chunk.
            is_final: Whether this is the last chunk.

        Returns:
            The same text chunk (transparent passthrough).

        Raises:
            RuntimeError: If start() hasn't been called.
        """
        if self._run is None:
            raise RuntimeError("StreamingInterceptor.start() must be called first")

        if self._aborted:
            return text

        self._chunks_processed += 1
        self._accumulated_output += text

        # Record chunk
        chunk = StreamChunk(
            index=self._chunks_processed - 1,
            text=text,
            timestamp_ms=(time.time() - self._start_time) * 1000,
            is_final=is_final,
        )
        if self._run is not None:
            self._run.chunks.append(chunk)

        # Extract claims from this chunk
        if self.extractor is not None:
            claims = self.extractor.process_chunk(text)

            if claims and self.verifier is not None:
                # Verify extracted claims
                for claim in claims:
                    result = self.verifier.verify_claim(claim)

                    # Check abort after each claim (for STRICT strategy)
                    should_abort, reason = self.verifier.check_abort()
                    if should_abort:
                        self._aborted = True
                        self._abort_reason = reason or AbortReason.UNGROUNDED_CLAIM
                        break

        # Handle final chunk — flush remaining buffer
        if is_final and self.extractor is not None:
            final_claims = self.extractor.flush()
            if final_claims and self.verifier is not None and not self._aborted:
                for claim in final_claims:
                    self.verifier.verify_claim(claim)

        return text

    def is_aborted(self) -> bool:
        """Check if the stream has been aborted.

        Returns:
            True if the stream was aborted due to hallucination.
        """
        return self._aborted

    @property
    def abort_reason(self) -> AbortReason:
        """The reason the stream was aborted."""
        return self._abort_reason

    @property
    def accumulated_output(self) -> str:
        """All text accumulated so far."""
        return self._accumulated_output

    def get_metrics(self) -> dict:
        """Get current streaming verification metrics.

        Returns:
            Dictionary of metrics.
        """
        if self.verifier is None:
            return {}

        metrics = self.verifier.get_metrics()
        metrics["chunks_processed"] = self._chunks_processed
        metrics["accumulated_chars"] = len(self._accumulated_output)
        if self._start_time > 0:
            metrics["elapsed_ms"] = (time.time() - self._start_time) * 1000
        return metrics

    # ── Generator Wrappers ───────────────────────────────────────────

    def wrap_sync(
        self, generator: Generator[str, None, None]
    ) -> Generator[str, None, None]:
        """Wrap a synchronous text generator with streaming verification.

        Args:
            generator: A sync generator yielding text chunks.

        Yields:
            Text chunks (transparent passthrough). Stops early if aborted.
        """
        for chunk in generator:
            if self._aborted:
                break
            self.process_chunk(chunk)
            yield chunk

        # Final flush
        if not self._aborted:
            self.process_chunk("", is_final=True)

    async def wrap_async(
        self, generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Wrap an asynchronous text generator with streaming verification.

        Args:
            generator: An async generator yielding text chunks.

        Yields:
            Text chunks (transparent passthrough). Stops early if aborted.
        """
        async for chunk in generator:
            if self._aborted:
                break
            self.process_chunk(chunk)
            yield chunk

        # Final flush
        if not self._aborted:
            self.process_chunk("", is_final=True)

    # ── Simulated Streams (for testing) ─────────────────────────────

    def simulate_stream(
        self,
        chunks: list[str],
        run_id: str | None = None,
        task: str = "",
    ) -> dict[str, Any]:
        """Simulate a streaming LLM response for testing.

        Processes a list of text chunks through the full verification
        pipeline without needing a real LLM. Useful for deterministic
        testing and benchmarking.

        Args:
            chunks: List of text chunks simulating streaming LLM output.
            run_id: Run identifier.
            task: Task description.

        Returns:
            Dictionary with metrics, claims, and results.
        """
        self.start(run_id=run_id, task=task)

        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            self.process_chunk(chunk, is_final=is_final)
            if self._aborted:
                break

        return {
            "metrics": self.get_metrics(),
            "aborted": self._aborted,
            "abort_reason": self._abort_reason.value,
            "abort_at_chunk": self._chunks_processed if self._aborted else -1,
            "accumulated_output": self._accumulated_output,
        }

    # ── Async Methods ────────────────────────────────────────────────

    async def process_chunk_async(self, text: str, is_final: bool = False) -> str:
        """Async variant of process_chunk for use in async contexts.

        Same behavior as process_chunk but is awaitable.

        Args:
            text: Raw text chunk.
            is_final: Whether this is the last chunk.

        Returns:
            The same text chunk (transparent passthrough).
        """
        # Offload to thread for CPU-bound claim extraction
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.process_chunk, text, is_final
        )
        return result
