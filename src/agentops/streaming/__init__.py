"""
Streaming Verification — real-time claim checking during agent generation.

Intercepts streaming LLM output at the token/chunk level, extracts factual
claims as they emerge, verifies each claim against a ground-truth evidence
store, and can abort generation early when hallucination is detected.

Unlike post-hoc verification (which checks the full answer after generation),
streaming verification catches hallucinations mid-response — reducing latency
and preventing bad outputs from reaching users.

Streaming Performance Evaluation — latency, throughput, and partial-output
quality measurement for streaming LLM responses. Captures time-to-first-token,
inter-token latency (P50/P90/P95/P99), tokens-per-second throughput, stall
detection, and partial-output quality snapshots.

Modules:
- state: Data models (StreamingConfig, StreamChunk, StreamingClaim, etc.)
- claim_extractor: Sentence-level claim extraction from partial text
- verifier: Real-time claim checking with abort thresholds
- interceptor: SSE/async-generator wrapper with stream control
- perf: Performance evaluation (TTFT, ITL, TPS, partial quality, regression)
"""

from agentops.streaming.claim_extractor import ClaimExtractor
from agentops.streaming.interceptor import StreamingInterceptor
from agentops.streaming.perf import (
    STREAMING_PERF_BENCHMARK,
    StreamingBenchmarkQuery,
    StreamingPerfCapture,
    StreamingPerfConfig,
    StreamingPerfEvaluator,
    StreamingPerfMetrics,
    StreamingPerfRegression,
    StreamingPerfRegressionResult,
    StreamingPerfResult,
    TokenTiming,
    simulate_stream,
)
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
    # State
    "StreamingConfig",
    "StreamChunk",
    "StreamingClaim",
    "StreamingVerificationResult",
    "StreamingMetrics",
    "StreamingRun",
    "AbortReason",
    "VerificationStrategy",
    # Verification
    "ClaimExtractor",
    "StreamingVerifier",
    "StreamingInterceptor",
    # Performance
    "StreamingPerfConfig",
    "TokenTiming",
    "StreamingPerfMetrics",
    "StreamingPerfResult",
    "StreamingPerfCapture",
    "StreamingPerfEvaluator",
    "StreamingPerfRegression",
    "StreamingPerfRegressionResult",
    "StreamingBenchmarkQuery",
    "STREAMING_PERF_BENCHMARK",
    "simulate_stream",
]
