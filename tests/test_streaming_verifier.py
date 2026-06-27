"""Tests for streaming claim verification."""

import pytest
from agentops.streaming.state import (
    StreamingClaim,
    StreamingConfig,
    StreamingVerificationResult,
    VerificationStrategy,
    AbortReason,
)
from agentops.streaming.verifier import StreamingVerifier


@pytest.fixture
def sample_evidence():
    return {
        "doc-1": "CloudDeploy supports two-factor authentication via TOTP and SMS. "
                 "Navigate to Settings > Security to configure 2FA.",
        "doc-2": "The deployment pipeline uses Kubernetes with Helm charts. "
                 "All services run in containers managed by Kubernetes.",
        "doc-3": "CloudDeploy requires Python 3.10 or later. Install the CLI via pip.",
        "doc-4": "Monitoring is provided through Prometheus metrics and Grafana dashboards. "
                 "Alerting is configured in alertmanager.yml.",
        "doc-5": "Rate limits: 1000 requests per minute for API, 100 requests per minute for webhooks.",
    }


@pytest.fixture
def verifier(sample_evidence):
    v = StreamingVerifier(evidence=sample_evidence)
    v.set_evidence(sample_evidence)
    v.start_run("test-1", "Test task")
    return v


class TestStreamingVerifierInit:
    def test_create_verifier(self):
        v = StreamingVerifier()
        assert v.config is not None
        assert v.evidence == {}

    def test_set_evidence(self):
        v = StreamingVerifier()
        v.set_evidence({"doc-1": "CloudDeploy uses Kubernetes."})
        assert len(v.evidence) == 1
        assert "kubernetes" in v._evidence_flat  # stored lowercase

    def test_start_run(self):
        v = StreamingVerifier()
        run = v.start_run("r1", "Test task")
        assert run.run_id == "r1"
        assert run.task == "Test task"
        assert v.run == run


class TestVerifyClaim:
    def test_grounded_claim(self, verifier):
        claim = StreamingClaim(text="CloudDeploy supports two-factor authentication via TOTP and SMS.")
        result = verifier.verify_claim(claim)
        assert result.grounded
        assert result.score > 0.0

    def test_ungrounded_claim(self, verifier):
        claim = StreamingClaim(text="CloudDeploy uses Docker Swarm for orchestration.")
        result = verifier.verify_claim(claim)
        assert not result.grounded
        assert result.score < 0.5

    def test_partially_grounded_claim(self, verifier):
        claim = StreamingClaim(
            text="CloudDeploy supports 2FA and uses AWS Lambda for serverless computing."
        )
        result = verifier.verify_claim(claim)
        # "2FA" is grounded, "AWS Lambda" is not — should be partially grounded
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_no_evidence_loaded(self):
        v = StreamingVerifier()
        v.start_run("test", "task")
        claim = StreamingClaim(text="Some claim.")
        result = v.verify_claim(claim)
        assert not result.grounded
        assert result.error == "No evidence loaded"

    def test_verification_latency(self, verifier):
        claim = StreamingClaim(text="CloudDeploy requires Python 3.10.")
        result = verifier.verify_claim(claim)
        assert result.latency_ms >= 0

    def test_entity_verification_version(self, verifier):
        verifier.config.track_entities = True
        claim = StreamingClaim(
            text="CloudDeploy v2.5.0 uses Python 3.10.",
            entities=["v2.5.0", "Python 3.10"],
        )
        result = verifier.verify_claim(claim)
        # Python 3.10 is in evidence, v2.5.0 is not
        assert result.score < 1.0  # Not perfect because of hallucinated version

    def test_entity_verification_url(self, verifier):
        verifier.config.track_entities = True
        claim = StreamingClaim(
            text="See https://docs.clouddeploy.com for details.",
            entities=["https://docs.clouddeploy.com"],
        )
        result = verifier.verify_claim(claim)
        # URL not in evidence
        assert isinstance(result.score, float)

    def test_empty_claim(self, verifier):
        claim = StreamingClaim(text="")
        result = verifier.verify_claim(claim)
        # Should handle empty claims gracefully
        assert isinstance(result.grounded, bool)

    def test_very_short_claim(self, verifier):
        claim = StreamingClaim(text="OK.")
        result = verifier.verify_claim(claim)
        assert isinstance(result.grounded, bool)


class TestBatchVerification:
    def test_verify_multiple_claims(self, verifier):
        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA via TOTP and SMS."),
            StreamingClaim(text="The deployment pipeline uses Kubernetes with Helm charts."),
            StreamingClaim(text="Docker Swarm is used for container orchestration."),
        ]
        results = verifier.verify_claims(claims)
        assert len(results) == 3
        assert results[0].grounded  # 2FA claim is grounded
        assert results[1].grounded  # Kubernetes claim has multiple key terms matching
        assert not results[2].grounded  # Docker Swarm claim is hallucinated


class TestAbortChecking:
    def test_no_abort_when_all_grounded(self, verifier):
        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA via TOTP and SMS."),
            StreamingClaim(text="The deployment pipeline uses Kubernetes with Helm charts."),
        ]
        verifier.verify_claims(claims)
        should, reason = verifier.check_abort()
        assert not should

    def test_abort_on_ungrounded_exceeds_threshold(self, verifier):
        verifier.config.strategy = VerificationStrategy.THRESHOLD
        verifier.config.abort_threshold = 0.30

        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA."),  # grounded
            StreamingClaim(text="Docker Swarm is used."),  # ungrounded
            StreamingClaim(text="AWS ECS is used."),  # ungrounded
        ]
        verifier.verify_claims(claims)
        should, reason = verifier.check_abort()
        assert should
        assert reason == AbortReason.THRESHOLD_EXCEEDED

    def test_no_abort_below_threshold(self, verifier):
        verifier.config.strategy = VerificationStrategy.THRESHOLD
        verifier.config.abort_threshold = 0.80

        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA."),  # grounded
            StreamingClaim(text="Kubernetes is used."),  # grounded
            StreamingClaim(text="Python 3.10 is required."),  # grounded
            StreamingClaim(text="Docker Swarm is used."),  # ungrounded
        ]
        verifier.verify_claims(claims)
        should, reason = verifier.check_abort()
        assert not should  # 25% < 80%

    def test_strict_abort_first_ungrounded(self, verifier):
        verifier.config.strategy = VerificationStrategy.STRICT

        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA."),  # grounded
            StreamingClaim(text="Docker Swarm is used."),  # ungrounded → ABORT
        ]
        verifier.verify_claims(claims)
        should, reason = verifier.check_abort()
        assert should
        assert reason == AbortReason.UNGROUNDED_CLAIM

    def test_lenient_never_aborts(self, verifier):
        verifier.config.strategy = VerificationStrategy.LENIENT

        claims = [
            StreamingClaim(text="Docker Swarm is used."),  # ungrounded
            StreamingClaim(text="AWS Lambda is used."),  # ungrounded
            StreamingClaim(text="Heroku is used."),  # ungrounded
        ]
        verifier.verify_claims(claims)
        should, reason = verifier.check_abort()
        assert not should


class TestMetrics:
    def test_get_metrics_empty(self, verifier):
        metrics = verifier.get_metrics()
        assert metrics["total_claims"] == 0
        assert metrics["groundedness"] == 1.0

    def test_get_metrics_after_verification(self, verifier):
        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA."),
            StreamingClaim(text="Python 3.10 is required."),
            StreamingClaim(text="Docker Swarm is used."),
        ]
        verifier.verify_claims(claims)
        metrics = verifier.get_metrics()
        assert metrics["total_claims"] == 3
        assert metrics["grounded_claims"] >= 1
        assert metrics["ungrounded_claims"] >= 1
        assert 0.0 < metrics["groundedness"] < 1.0

    def test_metrics_after_abort(self, verifier):
        verifier.config.strategy = VerificationStrategy.STRICT
        claims = [
            StreamingClaim(text="CloudDeploy supports 2FA."),
            StreamingClaim(text="Docker Swarm is used."),
        ]
        verifier.verify_claims(claims)
        verifier.check_abort()
        metrics = verifier.get_metrics()
        assert metrics.get("aborted") == True
        assert metrics["abort_reason"] == "ungrounded_claim"


class TestKeyTermExtraction:
    def test_extract_key_terms(self, verifier):
        terms = verifier._extract_key_terms("clouddeploy supports kubernetes orchestration")
        assert "clouddeploy" in terms
        assert "kubernetes" in terms
        assert "orchestration" in terms

    def test_stop_word_filtering(self, verifier):
        terms = verifier._extract_key_terms("the system is very good and works well")
        # Stop words should be filtered
        assert "the" not in terms
        assert "is" not in terms
        assert "and" not in terms

    def test_bigram_extraction(self, verifier):
        terms = verifier._extract_key_terms("two-factor authentication is supported")
        # Should include bigrams
        assert "two-factor" in terms
        assert "authentication" in terms
        assert "two-factor authentication" in terms

    def test_short_word_filtering(self, verifier):
        terms = verifier._extract_key_terms("x y z")
        # Words shorter than 2 chars should be filtered
        assert "x" not in terms


class TestEntityVerification:
    def test_version_is_hallucination(self, verifier):
        assert verifier._is_likely_hallucination("v3.2.1") is True
        assert verifier._is_likely_hallucination("2.5.0") is True

    def test_config_key_is_hallucination(self, verifier):
        assert verifier._is_likely_hallucination("auth_token") is True

    def test_camelcase_is_hallucination(self, verifier):
        assert verifier._is_likely_hallucination("CloudDeployConfig") is True

    def test_url_is_hallucination(self, verifier):
        assert verifier._is_likely_hallucination("https://example.com") is True

    def test_path_is_hallucination(self, verifier):
        assert verifier._is_likely_hallucination("/etc/config.yaml") is True

    def test_plain_word_not_hallucination(self, verifier):
        assert verifier._is_likely_hallucination("hello") is False
        assert verifier._is_likely_hallucination("world") is False
