"""
Streaming Verification — real-time claim checking during agent generation.

Intercepts streaming LLM output at the token/chunk level, extracts factual
claims as they emerge, verifies each claim against a ground-truth evidence
store, and can abort generation early when hallucination is detected.

Unlike post-hoc verification (which checks the full answer after generation),
streaming verification catches hallucinations mid-response — reducing latency
and preventing bad outputs from reaching users.

Modules:
- state: Data models (StreamingConfig, StreamChunk, StreamingClaim, etc.)
- claim_extractor: Sentence-level claim extraction from partial text
- verifier: Real-time claim checking with abort thresholds
- interceptor: SSE/async-generator wrapper with stream control
"""

from agentops.streaming.claim_extractor import ClaimExtractor
from agentops.streaming.interceptor import StreamingInterceptor
from agentops.streaming.state import (
    AbortReason,
    StreamChunk,
    StreamingClaim,
    StreamingConfig,
    StreamingMetrics,
    StreamingRun,
    StreamingVerificationResult,
    VerificationStrategy,
)
from agentops.streaming.verifier import StreamingVerifier

__all__ = [
    "StreamingConfig",
    "StreamChunk",
    "StreamingClaim",
    "StreamingVerificationResult",
    "StreamingMetrics",
    "StreamingRun",
    "AbortReason",
    "VerificationStrategy",
    "ClaimExtractor",
    "StreamingVerifier",
    "StreamingInterceptor",
]
