"""Tests for streaming verification state models."""

import pytest
from agentops.streaming.state import (
    StreamingConfig,
    StreamChunk,
    StreamingClaim,
    StreamingVerificationResult,
    StreamingMetrics,
    StreamingRun,
    AbortReason,
    VerificationStrategy,
)


class TestVerificationStrategy:
    def test_all_strategies_exist(self):
        assert VerificationStrategy.STRICT == "strict"
        assert VerificationStrategy.THRESHOLD == "threshold"
        assert VerificationStrategy.LENIENT == "lenient"
        assert VerificationStrategy.ACCUMULATING == "accumulating"

    def test_strategy_values_are_strings(self):
        assert isinstance(VerificationStrategy.STRICT.value, str)
        assert isinstance(VerificationStrategy.THRESHOLD.value, str)


class TestAbortReason:
    def test_all_reasons_exist(self):
        reasons = list(AbortReason)
        assert len(reasons) >= 6

    def test_none_means_no_abort(self):
        assert AbortReason.NONE.value == "none"


class TestStreamingConfig:
    def test_default_config(self):
        config = StreamingConfig()
        assert config.strategy == VerificationStrategy.THRESHOLD
        assert config.abort_threshold == 0.30
        assert config.chunk_overlap == 50
        assert config.min_claim_length == 20
        assert config.max_buffer_chars == 500

    def test_custom_config(self):
        config = StreamingConfig(
            strategy=VerificationStrategy.STRICT,
            abort_threshold=0.10,
            min_claim_length=30,
        )
        assert config.strategy == VerificationStrategy.STRICT
        assert config.abort_threshold == 0.10
        assert config.min_claim_length == 30

    def test_invalid_abort_threshold_low(self):
        with pytest.raises(ValueError):
            StreamingConfig(abort_threshold=-0.1)

    def test_invalid_abort_threshold_high(self):
        with pytest.raises(ValueError):
            StreamingConfig(abort_threshold=1.5)

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ValueError):
            StreamingConfig(chunk_overlap=-1)

    def test_invalid_min_claim_length(self):
        with pytest.raises(ValueError):
            StreamingConfig(min_claim_length=3)

    def test_boundary_abort_threshold_zero(self):
        config = StreamingConfig(abort_threshold=0.0)
        assert config.abort_threshold == 0.0

    def test_boundary_abort_threshold_one(self):
        config = StreamingConfig(abort_threshold=1.0)
        assert config.abort_threshold == 1.0


class TestStreamChunk:
    def test_create_chunk(self):
        chunk = StreamChunk(index=0, text="Hello")
        assert chunk.index == 0
        assert chunk.text == "Hello"
        assert not chunk.is_final

    def test_final_chunk(self):
        chunk = StreamChunk(index=5, text="Done", is_final=True)
        assert chunk.is_final
        assert chunk.index == 5

    def test_timestamp(self):
        chunk = StreamChunk(index=0, text="test", timestamp_ms=123.45)
        assert chunk.timestamp_ms == 123.45


class TestStreamingClaim:
    def test_create_claim(self):
        claim = StreamingClaim(text="The system uses Kubernetes.")
        assert claim.text == "The system uses Kubernetes."
        assert claim.chunk_indices == []
        assert claim.confidence == 1.0
        assert not claim.is_partial

    def test_claim_with_entities(self):
        claim = StreamingClaim(
            text="Kubernetes v1.28 is used.",
            entities=["Kubernetes", "v1.28"],
            confidence=0.85,
        )
        assert "Kubernetes" in claim.entities
        assert claim.confidence == 0.85

    def test_partial_claim(self):
        claim = StreamingClaim(
            text="The system uses...",
            is_partial=True,
        )
        assert claim.is_partial

    def test_char_offsets(self):
        claim = StreamingClaim(
            text="test",
            start_char=100,
            end_char=104,
        )
        assert claim.start_char == 100
        assert claim.end_char == 104


class TestStreamingVerificationResult:
    def test_grounded_result(self):
        claim = StreamingClaim(text="Kubernetes is used.")
        result = StreamingVerificationResult(
            claim=claim,
            grounded=True,
            evidence_chunks=["doc-1"],
            score=0.95,
            latency_ms=5.0,
        )
        assert result.grounded
        assert len(result.evidence_chunks) == 1
        assert result.score == 0.95

    def test_ungrounded_result(self):
        claim = StreamingClaim(text="Docker Swarm is used.")
        result = StreamingVerificationResult(
            claim=claim,
            grounded=False,
            contradictory_evidence=["doc-2"],
            score=0.0,
        )
        assert not result.grounded
        assert len(result.contradictory_evidence) == 1

    def test_error_result(self):
        claim = StreamingClaim(text="test")
        result = StreamingVerificationResult(
            claim=claim,
            grounded=False,
            error="Timeout",
        )
        assert result.error == "Timeout"


class TestStreamingMetrics:
    def test_default_metrics(self):
        m = StreamingMetrics()
        assert m.total_claims == 0
        assert m.grounded_claims == 0
        assert m.ungrounded_claims == 0
        assert m.groundedness == 1.0  # Default when no claims
        assert not m.aborted

    def test_metrics_to_dict(self):
        m = StreamingMetrics(
            run_id="test-1",
            total_claims=10,
            grounded_claims=8,
            ungrounded_claims=2,
            groundedness=0.80,
            aborted=True,
            abort_reason=AbortReason.THRESHOLD_EXCEEDED,
        )
        d = m.to_dict()
        assert d["run_id"] == "test-1"
        assert d["total_claims"] == 10
        assert d["groundedness"] == 0.8
        assert d["aborted"] is True
        assert d["abort_reason"] == "threshold_exceeded"


class TestStreamingRun:
    def test_create_run(self):
        run = StreamingRun(
            run_id="r1",
            task="Test task",
        )
        assert run.run_id == "r1"
        assert run.task == "Test task"
        assert run.accumulated_text == ""
        assert not run.aborted

    def test_ungrounded_rate_empty(self):
        run = StreamingRun(run_id="r1", task="test")
        assert run.ungrounded_rate() == 0.0

    def test_ungrounded_rate_with_results(self):
        run = StreamingRun(run_id="r1", task="test")
        claim = StreamingClaim(text="test")
        run.results = [
            StreamingVerificationResult(claim=claim, grounded=True),
            StreamingVerificationResult(claim=claim, grounded=False),
            StreamingVerificationResult(claim=claim, grounded=True),
            StreamingVerificationResult(claim=claim, grounded=False),
        ]
        assert run.ungrounded_rate() == 0.5

    def test_should_abort_strict_grounded(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
            config=StreamingConfig(strategy=VerificationStrategy.STRICT),
        )
        claim = StreamingClaim(text="test")
        run.results = [StreamingVerificationResult(claim=claim, grounded=True)]
        should, reason = run.should_abort()
        assert not should

    def test_should_abort_strict_ungrounded(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
            config=StreamingConfig(strategy=VerificationStrategy.STRICT),
        )
        claim = StreamingClaim(text="test")
        run.results = [StreamingVerificationResult(claim=claim, grounded=False)]
        should, reason = run.should_abort()
        assert should
        assert reason == AbortReason.UNGROUNDED_CLAIM

    def test_should_abort_strict_contradiction(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
            config=StreamingConfig(
                strategy=VerificationStrategy.STRICT,
                abort_on_contradiction=True,
            ),
        )
        claim = StreamingClaim(text="test")
        run.results = [
            StreamingVerificationResult(
                claim=claim,
                grounded=False,
                contradictory_evidence=["doc-1"],
            )
        ]
        should, reason = run.should_abort()
        assert should
        assert reason == AbortReason.CONTRADICTION

    def test_should_abort_threshold_not_exceeded(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
            config=StreamingConfig(
                strategy=VerificationStrategy.THRESHOLD,
                abort_threshold=0.50,
            ),
        )
        claim = StreamingClaim(text="test")
        run.results = [
            StreamingVerificationResult(claim=claim, grounded=True),
            StreamingVerificationResult(claim=claim, grounded=True),
            StreamingVerificationResult(claim=claim, grounded=False),
        ]
        should, reason = run.should_abort()
        assert not should  # 33% < 50%

    def test_should_abort_threshold_exceeded(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
            config=StreamingConfig(
                strategy=VerificationStrategy.THRESHOLD,
                abort_threshold=0.30,
            ),
        )
        claim = StreamingClaim(text="test")
        run.results = [
            StreamingVerificationResult(claim=claim, grounded=True),
            StreamingVerificationResult(claim=claim, grounded=False),
            StreamingVerificationResult(claim=claim, grounded=False),
        ]
        should, reason = run.should_abort()
        assert should  # 67% > 30%
        assert reason == AbortReason.THRESHOLD_EXCEEDED

    def test_should_abort_lenient_never(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
            config=StreamingConfig(strategy=VerificationStrategy.LENIENT),
        )
        claim = StreamingClaim(text="test")
        run.results = [
            StreamingVerificationResult(claim=claim, grounded=False),
            StreamingVerificationResult(claim=claim, grounded=False),
            StreamingVerificationResult(claim=claim, grounded=False),
        ]
        should, reason = run.should_abort()
        assert not should  # LENIENT never aborts

    def test_already_aborted(self):
        run = StreamingRun(
            run_id="r1",
            task="test",
        )
        run.aborted = True
        run.abort_reason = AbortReason.TIMEOUT
        should, reason = run.should_abort()
        assert should
        assert reason == AbortReason.TIMEOUT
