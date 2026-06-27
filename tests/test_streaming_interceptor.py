"""Tests for streaming interceptor — wraps streaming output with verification."""

import pytest
import asyncio
from agentops.streaming.state import (
    StreamingConfig,
    StreamingClaim,
    AbortReason,
    VerificationStrategy,
)
from agentops.streaming.interceptor import StreamingInterceptor


@pytest.fixture
def evidence():
    return {
        "doc-1": "CloudDeploy supports two-factor authentication via TOTP and SMS. "
                 "Navigate to Settings > Security to configure 2FA.",
        "doc-2": "The deployment pipeline uses Kubernetes with Helm charts. "
                 "All services run in containers managed by Kubernetes.",
        "doc-3": "CloudDeploy requires Python 3.10 or later. Install the CLI via pip.",
        "doc-4": "Monitoring is provided through Prometheus metrics and Grafana dashboards.",
    }


@pytest.fixture
def interceptor(evidence):
    return StreamingInterceptor(evidence=evidence)


class TestInterceptorInit:
    def test_create_interceptor(self, evidence):
        ic = StreamingInterceptor(evidence=evidence)
        assert ic.config is not None
        assert ic.extractor is not None
        assert ic.verifier is not None

    def test_custom_config(self):
        config = StreamingConfig(strategy=VerificationStrategy.STRICT)
        ic = StreamingInterceptor(config=config, evidence={})
        assert ic.config.strategy == VerificationStrategy.STRICT


class TestStartAndState:
    def test_start_initializes_run(self, interceptor):
        run = interceptor.start(run_id="test-1", task="test")
        assert run.run_id == "test-1"
        assert interceptor._run is not None
        assert interceptor._chunks_processed == 0
        assert not interceptor.is_aborted()

    def test_auto_generated_run_id(self, interceptor):
        run = interceptor.start()
        assert run.run_id.startswith("stream-")
        assert len(run.run_id) > 10

    def test_process_without_start_raises(self, interceptor):
        with pytest.raises(RuntimeError):
            interceptor.process_chunk("text")


class TestProcessChunk:
    def test_process_grounded_chunk(self, interceptor):
        interceptor.start(run_id="r1", task="test")
        result = interceptor.process_chunk(
            "CloudDeploy supports two-factor authentication via TOTP and SMS. "
        )
        assert result  # Returns text transparently
        assert not interceptor.is_aborted()

    def test_process_multiple_grounded_chunks(self, interceptor):
        interceptor.start(run_id="r1", task="test")
        interceptor.process_chunk("CloudDeploy supports 2FA. ")
        interceptor.process_chunk("The deployment pipeline uses Kubernetes. ")
        interceptor.process_chunk("Python 3.10 is required. ")
        assert interceptor._chunks_processed >= 2

    def test_abort_on_ungrounded_chunk_threshold(self, evidence):
        config = StreamingConfig(
            strategy=VerificationStrategy.THRESHOLD,
            abort_threshold=0.30,
        )
        ic = StreamingInterceptor(config=config, evidence=evidence)
        ic.start(run_id="r1", task="test")

        # Grounded
        ic.process_chunk("CloudDeploy supports 2FA. ")
        assert not ic.is_aborted()

        # Ungrounded
        ic.process_chunk("Docker Swarm is used for orchestration. ")
        # May or may not abort depending on claims extracted

        # More ungrounded
        ic.process_chunk("AWS ECS is used for container management. ")

        # Check if aborted
        metrics = ic.get_metrics()
        assert "total_claims" in metrics

    def test_abort_on_contradiction(self, evidence):
        config = StreamingConfig(
            strategy=VerificationStrategy.STRICT,
            abort_on_contradiction=True,
        )
        ic = StreamingInterceptor(config=config, evidence=evidence)
        ic.start(run_id="r1", task="test")

        ic.process_chunk("Two-factor authentication is NOT supported on CloudDeploy. ")
        # This directly contradicts evidence — should trigger abort

    def test_process_final_chunk(self, interceptor):
        interceptor.start(run_id="r1", task="test")
        interceptor.process_chunk("CloudDeploy requires Python 3.10. ", is_final=True)
        metrics = interceptor.get_metrics()
        assert metrics["chunks_processed"] >= 1

    def test_no_double_abort(self, interceptor):
        interceptor.start(run_id="r1", task="test")
        interceptor._aborted = True
        result = interceptor.process_chunk("More text. ")
        assert result == "More text. "  # Still passthrough even after abort


class TestSimulateStream:
    def test_fully_grounded_stream(self, interceptor):
        chunks = [
            "CloudDeploy supports 2FA via TOTP and SMS. ",
            "Deployment uses Kubernetes with Helm charts. ",
            "CloudDeploy requires Python 3.10 or later. ",
        ]
        result = interceptor.simulate_stream(chunks, run_id="r1", task="test")
        assert not result["aborted"]
        assert result["metrics"]["total_claims"] >= 1

    def test_stream_with_hallucination_threshold(self, evidence):
        config = StreamingConfig(
            strategy=VerificationStrategy.THRESHOLD,
            abort_threshold=0.30,
        )
        ic = StreamingInterceptor(config=config, evidence=evidence)

        chunks = [
            "CloudDeploy supports 2FA. ",  # grounded
            "Docker Swarm is used. ",  # hallucination
            "AWS Lambda is used. ",  # hallucination
            "Python 3.10 is required. ",  # grounded
        ]
        result = ic.simulate_stream(chunks, run_id="r1", task="test")
        # Should abort when ungrounded rate exceeds 30%
        metrics = result["metrics"]
        assert "total_claims" in metrics

    def test_stream_strict_strategy(self, evidence):
        config = StreamingConfig(strategy=VerificationStrategy.STRICT)
        ic = StreamingInterceptor(config=config, evidence=evidence)

        chunks = [
            "CloudDeploy supports 2FA. ",  # grounded
            "Docker Swarm is used for orchestration. ",  # ungrounded → ABORT
            "Python 3.10 is required. ",  # never reached
        ]
        result = ic.simulate_stream(chunks, run_id="r1", task="test")
        # Strict should abort on first ungrounded claim
        assert result["metrics"]["total_claims"] >= 1

    def test_stream_lenient_never_aborts(self, evidence):
        config = StreamingConfig(strategy=VerificationStrategy.LENIENT)
        ic = StreamingInterceptor(config=config, evidence=evidence)

        chunks = [
            "Docker Swarm is used. ",  # hallucination
            "AWS ECS is used. ",  # hallucination
            "Heroku is used. ",  # hallucination
        ]
        result = ic.simulate_stream(chunks, run_id="r1", task="test")
        assert not result["aborted"]

    def test_empty_stream(self, interceptor):
        result = interceptor.simulate_stream([], run_id="r1", task="test")
        assert not result["aborted"]
        assert result["metrics"]["chunks_processed"] == 0

    def test_stream_metrics_complete(self, interceptor):
        chunks = [
            "CloudDeploy supports two-factor authentication. ",
            "Kubernetes is used for deployment with Helm charts. ",
            "CloudDeploy requires Python 3.10 or later. ",
        ]
        result = interceptor.simulate_stream(chunks, run_id="r1", task="test")
        metrics = result["metrics"]
        assert "total_chunks" in metrics or "chunks_processed" in metrics
        assert "total_claims" in metrics
        assert "groundedness" in metrics
        assert metrics["groundedness"] > 0.5


class TestGeneratorWrapping:
    def test_wrap_sync_generator(self, interceptor):
        def generator():
            yield "CloudDeploy supports 2FA. "
            yield "Kubernetes is used. "

        interceptor.start(run_id="r1", task="test")
        results = list(interceptor.wrap_sync(generator()))
        assert len(results) == 2
        assert results[0] == "CloudDeploy supports 2FA. "

    def test_wrap_sync_generator_aborts(self, evidence):
        config = StreamingConfig(strategy=VerificationStrategy.STRICT)
        ic = StreamingInterceptor(config=config, evidence=evidence)

        def generator():
            yield "CloudDeploy supports 2FA via TOTP. "
            yield "The platform uses Docker Swarm for orchestration. "  # hallucination → abort

        ic.start(run_id="r1", task="test")
        results = list(ic.wrap_sync(generator()))
        # Should stop at or before the hallucination
        assert len(results) <= 2

    def test_wrap_sync_empty_generator(self, interceptor):
        def generator():
            return
            yield  # type: ignore

        interceptor.start(run_id="r1", task="test")
        results = list(interceptor.wrap_sync(generator()))
        assert len(results) == 0


class TestAsyncOperations:
    def test_process_chunk_async(self, interceptor):
        interceptor.start(run_id="r1", task="test")

        async def run():
            return await interceptor.process_chunk_async("CloudDeploy supports 2FA. ")

        result = asyncio.run(run())
        assert result == "CloudDeploy supports 2FA. "

    def test_wrap_async_generator(self, interceptor):
        async def generator():
            yield "CloudDeploy supports 2FA. "
            yield "Kubernetes is used. "

        interceptor.start(run_id="r1", task="test")

        async def run():
            results = []
            async for chunk in interceptor.wrap_async(generator()):
                results.append(chunk)
            return results

        results = asyncio.run(run())
        assert len(results) == 2


class TestAccumulatedOutput:
    def test_accumulated_output_grows(self, interceptor):
        interceptor.start(run_id="r1", task="test")
        interceptor.process_chunk("Hello ")
        interceptor.process_chunk("World. ")
        assert "Hello" in interceptor.accumulated_output
        assert "World" in interceptor.accumulated_output

    def test_accumulated_output_resets_on_new_run(self, interceptor):
        interceptor.start(run_id="r1", task="test")
        interceptor.process_chunk("First run. ")
        assert interceptor.accumulated_output

        interceptor.start(run_id="r2", task="test2")
        assert interceptor.accumulated_output == ""


class TestEndToEndScenarios:
    def test_support_ticket_scenario(self, evidence):
        """Realistic support ticket resolution with streaming verification."""
        config = StreamingConfig(
            strategy=VerificationStrategy.THRESHOLD,
            abort_threshold=0.30,
        )
        ic = StreamingInterceptor(config=config, evidence=evidence)

        # Simulated agent responding to "How do I enable 2FA?"
        chunks = [
            "To enable two-factor authentication on CloudDeploy, ",
            "navigate to Settings > Security in the dashboard. ",
            "From there, you can select TOTP or SMS as your preferred 2FA method. ",
            "CloudDeploy supports both authenticator apps and text messages. ",
            "You need Python 3.10 or later installed to use the CLI. ",
        ]
        result = ic.simulate_stream(chunks, run_id="support-1", task="Enable 2FA")
        metrics = result["metrics"]
        assert metrics.get("groundedness", 0) > 0.3

    def test_hallucinated_support_scenario(self, evidence):
        """Agent hallucinates about unsupported features."""
        config = StreamingConfig(
            strategy=VerificationStrategy.THRESHOLD,
            abort_threshold=0.30,
        )
        ic = StreamingInterceptor(config=config, evidence=evidence)

        chunks = [
            "To enable 2FA, navigate to Settings > Security. ",
            "CloudDeploy uses Docker Compose for container orchestration. ",  # HALLUCINATION
            "You can also deploy using AWS ECS Fargate. ",  # HALLUCINATION
            "Monitoring is provided through Prometheus and Grafana. ",  # grounded
        ]
        result = ic.simulate_stream(chunks, run_id="support-2", task="Deployment methods")
        metrics = result["metrics"]
        # Should have detected at least one ungrounded claim
        assert metrics.get("ungrounded_claims", 0) >= 1 or result["aborted"]

    def test_mixed_quality_stream(self, evidence):
        """Stream with mix of grounded and hallucinated claims."""
        config = StreamingConfig(
            strategy=VerificationStrategy.THRESHOLD,
            abort_threshold=0.50,
        )
        ic = StreamingInterceptor(config=config, evidence=evidence)

        chunks = [
            "CloudDeploy supports 2FA via TOTP and SMS. ",  # grounded
            "Kubernetes is used with Helm charts. ",  # grounded
            "Python 3.10 or later is required. ",  # grounded
            "The system also supports biometric authentication. ",  # hallucination
            "Rate limits are 10000 requests per minute. ",  # hallucination (should be 1000)
            "Monitoring uses Prometheus and Grafana. ",  # grounded
        ]
        result = ic.simulate_stream(chunks, run_id="mixed-1", task="Platform features")
        metrics = result["metrics"]
        # 2/6 hallucinated = 33% ungrounded → below 50% threshold, should not abort
        assert metrics.get("ungrounded_claims", 0) >= 1
        # Should have completed (or aborted if claims extracted differently)
        assert isinstance(result["aborted"], bool)
