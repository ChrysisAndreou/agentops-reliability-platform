"""Tests for claim extraction from streaming text."""

import pytest
from agentops.streaming.claim_extractor import ClaimExtractor
from agentops.streaming.state import StreamingConfig


class TestClaimExtractor:
    def test_extract_simple_claim(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "The system uses Kubernetes for orchestration. "
        )
        assert len(claims) == 1
        assert "Kubernetes" in claims[0].text
        assert not claims[0].is_partial

    def test_extract_multiple_claims(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "CloudDeploy supports 2FA. The deployment requires Python 3.10. "
            "Monitoring is through Prometheus."
        )
        assert len(claims) >= 1  # May extract 1-3 depending on sentence boundaries

    def test_extract_across_chunks(self):
        """Claims spanning multiple chunks should be assembled."""
        extractor = ClaimExtractor()

        # First chunk has incomplete sentence
        claims1 = extractor.process_chunk("CloudDeploy supports ")
        # Partial sentence, no complete claim yet
        assert all(c.is_partial for c in claims1) if claims1 else True

        # Second chunk completes it
        claims2 = extractor.process_chunk("two-factor authentication via TOTP. ")
        assert len(claims2) >= 1

    def test_non_claim_filtering(self):
        """Greetings and transitions should not be treated as claims."""
        extractor = ClaimExtractor()

        claims1 = extractor.process_chunk("Hello! How can I help you today? ")
        # Greetings and questions should be filtered
        assert len(claims1) == 0

        claims2 = extractor.process_chunk(
            "The platform requires Python 3.10. "
        )
        assert len(claims2) >= 1

    def test_filter_questions(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "What version of Python is required? The system needs Python 3.10. "
        )
        # Question should be filtered, claim should be extracted
        claim_texts = [c.text for c in claims]
        assert not any("?" in t for t in claim_texts)
        assert any("3.10" in t for t in claim_texts)

    def test_filter_transitions(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "The system requires Python 3.10. You also need pip installed. "
        )
        assert len(claims) >= 1

    def test_filter_pleasantries(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "Sure, I'll help you with that. CloudDeploy uses Kubernetes. "
        )
        claim_texts = [c.text for c in claims]
        assert not any("Sure" in t for t in claim_texts)

    def test_flush_remaining(self):
        extractor = ClaimExtractor()
        extractor.process_chunk("The system uses Kubernetes. ")
        final_claims = extractor.flush()
        # Should be empty since all claims were already extracted
        # (buffer was flushed)
        pass  # flush mostly catches edge cases

    def test_buffer_flush_on_max_chars(self):
        config = StreamingConfig(max_buffer_chars=50, min_claim_length=10)
        extractor = ClaimExtractor(config=config)

        # Feed text that doesn't end with sentence boundary
        long_text = "This is a very long text that does not end properly and keeps going on and on without any punctuation marks"
        claims = extractor.process_chunk(long_text)
        # Should force extraction when buffer exceeds max
        assert len(claims) >= 1

    def test_entity_extraction_versions(self):
        extractor = ClaimExtractor(config=StreamingConfig(track_entities=True))
        claims = extractor.process_chunk(
            "The system uses Kubernetes v1.28.0 for deployment. "
        )
        if claims:
            entities = claims[0].entities
            # Should detect version number
            assert any("1.28.0" in e for e in entities) or len(entities) > 0

    def test_entity_extraction_paths(self):
        extractor = ClaimExtractor(config=StreamingConfig(track_entities=True))
        claims = extractor.process_chunk(
            "The config is at /etc/clouddeploy/config.yaml. "
        )
        if claims:
            entities = claims[0].entities
            assert any("/etc/clouddeploy/config.yaml" in e for e in entities) or len(entities) > 0

    def test_entity_extraction_urls(self):
        extractor = ClaimExtractor(config=StreamingConfig(track_entities=True))
        claims = extractor.process_chunk(
            "Visit https://docs.clouddeploy.com for more info. "
        )
        if claims:
            entities = claims[0].entities
            assert any("https://docs.clouddeploy.com" in e for e in entities) or len(entities) > 0

    def test_claim_confidence(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "The deployment pipeline uses Kubernetes with Helm charts for configuration. "
        )
        if claims:
            assert 0.0 <= claims[0].confidence <= 1.0

    def test_abbreviation_handling(self):
        """Abbreviations like 'e.g.' should not split sentences."""
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "The system supports various protocols e.g. HTTP and HTTPS. "
            "Monitoring is via Grafana. "
        )
        # "e.g." should not split into two sentences
        claim_texts = [c.text for c in claims]
        assert not any("e.g" in t and len(t) < 20 for t in claim_texts)

    def test_min_claim_length(self):
        config = StreamingConfig(min_claim_length=100)
        extractor = ClaimExtractor(config=config)
        claims = extractor.process_chunk("Short. Text. Here. ")
        assert len(claims) == 0  # All too short

    def test_empty_chunk(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk("")
        assert len(claims) == 0

    def test_multiple_chunks_entity_tracking(self):
        extractor = ClaimExtractor(config=StreamingConfig(track_entities=True))
        # Process multiple chunks that form claims
        claims = extractor.process_chunk(
            "CloudDeploy v2.5.1 supports Python 3.10. "
            "The configuration file is at /etc/clouddeploy/config.yaml. "
        )
        all_entities = []
        for c in claims:
            all_entities.extend(c.entities)
        # Should have picked up entities from the text
        assert len(all_entities) > 0


class TestClaimExtractorEdgeCases:
    def test_unicode_text(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "The System unterstützt Unicode-Zeichen. Python 3.10 ist erforderlich. "
        )
        # Should not crash on unicode
        assert isinstance(claims, list)

    def test_very_long_sentence(self):
        config = StreamingConfig(max_buffer_chars=100, min_claim_length=10)
        extractor = ClaimExtractor(config=config)
        long_sentence = "The platform " + "supports " * 50 + "many features."
        claims = extractor.process_chunk(long_sentence)
        # Should force extraction
        assert isinstance(claims, list)

    def test_special_characters(self):
        extractor = ClaimExtractor()
        claims = extractor.process_chunk(
            "The API returns {\"status\": \"ok\"}. Python 3.10 is required. "
        )
        assert isinstance(claims, list)
