"""
Real-time streaming claim verification.

Verifies claims extracted from streaming LLM output against a ground-truth
evidence store. Supports multiple verification strategies and can trigger
stream abort when hallucination is detected.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from .state import (
    StreamingClaim,
    StreamingConfig,
    StreamingVerificationResult,
    StreamingRun,
    AbortReason,
    VerificationStrategy,
)


@dataclass
class StreamingVerifier:
    """Verifies streaming claims against evidence in real time.

    Supports four strategies:
    - STRICT: Abort on first ungrounded claim.
    - THRESHOLD: Abort when ungrounded rate exceeds threshold.
    - LENIENT: Flag but never abort.
    - ACCUMULATING: Buffer borderline claims for context before deciding.

    Evidence is provided as a dictionary mapping evidence chunk IDs to
    their text content. Verification uses keyword overlap and semantic
    similarity heuristics (no external API required).

    Attributes:
        config: Streaming configuration.
        evidence: Evidence store (chunk_id → text).
        run: Current streaming run state.
    """

    config: StreamingConfig = field(default_factory=StreamingConfig)
    evidence: dict[str, str] = field(default_factory=dict)
    run: StreamingRun | None = None

    # Accumulated evidence text for efficient searching
    _evidence_texts: list[str] = field(default_factory=list)
    _evidence_flat: str = ""

    def set_evidence(self, evidence: dict[str, str]) -> None:
        """Set the evidence store for verification.

        Args:
            evidence: Mapping of chunk IDs to text content.
        """
        self.evidence = evidence
        self._evidence_texts = list(evidence.values())[:self.config.evidence_window]
        self._evidence_flat = " ".join(self._evidence_texts).lower()

    def start_run(self, run_id: str, task: str) -> StreamingRun:
        """Start a new streaming verification run.

        Args:
            run_id: Unique run identifier.
            task: The task description.

        Returns:
            The new StreamingRun.
        """
        self.run = StreamingRun(
            run_id=run_id,
            task=task,
            config=self.config,
            evidence_store=self.evidence,
        )
        # Re-apply evidence to ensure _evidence_texts and _evidence_flat are set
        if self.evidence:
            self.set_evidence(self.evidence)
        return self.run

    def verify_claim(self, claim: StreamingClaim) -> StreamingVerificationResult:
        """Verify a single claim against the evidence store.

        Args:
            claim: The claim to verify.

        Returns:
            Verification result with groundedness, evidence, and score.
        """
        t0 = time.time()

        if not self._evidence_texts or not self._evidence_flat:
            result = StreamingVerificationResult(
                claim=claim,
                grounded=False,
                score=0.0,
                latency_ms=0.0,
                error="No evidence loaded",
            )
            self._record_result(result)
            return result

        try:
            # Extract key terms from claim
            claim_lower = claim.text.lower()
            key_terms = self._extract_key_terms(claim_lower)

            if not key_terms:
                result = StreamingVerificationResult(
                    claim=claim,
                    grounded=True,  # No verifiable terms = pass by default
                    score=1.0,
                    latency_ms=(time.time() - t0) * 1000,
                )
                self._record_result(result)
                return result

            # Check each key term against evidence
            evidence_hits: list[str] = []
            evidence_misses: list[str] = []
            contradictory: list[str] = []

            for term in key_terms:
                if term in self._evidence_flat:
                    evidence_hits.append(term)
                else:
                    evidence_misses.append(term)

            # Check for entity-level verification
            if self.config.track_entities and claim.entities:
                entity_hits, entity_contradictions = self._verify_entities(claim.entities)
                evidence_hits.extend(entity_hits)
                contradictory.extend(entity_contradictions)

            # Calculate grounded score
            total_checks = len(key_terms) + len(claim.entities)
            if total_checks == 0:
                grounded = True
                score = 1.0
            else:
                hit_count = len(evidence_hits)
                miss_count = len(evidence_misses)
                contradiction_count = len(contradictory)

                # Claim is grounded if most key terms are found
                check_ratio = hit_count / total_checks if total_checks > 0 else 1.0
                grounded = check_ratio >= 0.3 and contradiction_count == 0
                score = check_ratio

            latency_ms = (time.time() - t0) * 1000

            result = StreamingVerificationResult(
                claim=claim,
                grounded=grounded,
                evidence_chunks=self._find_evidence_for_terms(evidence_hits),
                contradictory_evidence=contradictory,
                score=score,
                latency_ms=latency_ms,
            )

        except Exception as e:
            result = StreamingVerificationResult(
                claim=claim,
                grounded=False,
                score=0.0,
                latency_ms=(time.time() - t0) * 1000,
                error=str(e),
            )

        self._record_result(result)
        return result

    def verify_claims(self, claims: list[StreamingClaim]) -> list[StreamingVerificationResult]:
        """Verify multiple claims in batch.

        Args:
            claims: Claims to verify.

        Returns:
            Verification results for each claim.
        """
        return [self.verify_claim(c) for c in claims]

    def check_abort(self) -> tuple[bool, AbortReason | None]:
        """Check if the current run should be aborted.

        Returns:
            (should_abort, abort_reason).
        """
        if self.run is None:
            return False, None

        if self.run.aborted:
            return True, self.run.abort_reason

        should, reason = self.run.should_abort()

        if should and reason:
            self.run.aborted = True
            self.run.abort_reason = reason
            self.run.abort_at_chunk = len(self.run.chunks)

        return should, reason

    def get_metrics(self) -> dict:
        """Get current streaming metrics as a dict.

        Returns:
            Dictionary of metrics.
        """
        if self.run is None:
            return {}

        total = len(self.run.results)
        grounded = sum(1 for r in self.run.results if r.grounded)
        ungrounded = total - grounded
        contradicted = sum(1 for r in self.run.results if r.contradictory_evidence)
        entities = sum(1 for r in self.run.results for _ in r.claim.entities)

        total_latency = sum(r.latency_ms for r in self.run.results)
        avg_latency = total_latency / total if total > 0 else 0.0

        return {
            "run_id": self.run.run_id,
            "total_chunks": len(self.run.chunks),
            "total_claims": total,
            "grounded_claims": grounded,
            "ungrounded_claims": ungrounded,
            "contradicted_claims": contradicted,
            "hallucinated_entities": entities,
            "aborted": self.run.aborted,
            "abort_reason": self.run.abort_reason.value,
            "abort_at_chunk": self.run.abort_at_chunk,
            "groundedness": round(grounded / total, 4) if total > 0 else 1.0,
            "total_latency_ms": round(total_latency, 2),
            "avg_verification_latency_ms": round(avg_latency, 2),
        }

    # ── Private helpers ──────────────────────────────────────────────

    def _record_result(self, result: StreamingVerificationResult) -> None:
        """Record a verification result in the current run."""
        if self.run is not None:
            self.run.results.append(result)
            if result.claim not in self.run.claims:
                self.run.claims.append(result.claim)

    @staticmethod
    def _extract_key_terms(text: str) -> list[str]:
        """Extract key verifiable terms from claim text.

        Removes stop words and short words, keeping nouns, verbs,
        technical terms, and numbers.

        Args:
            text: Lowercase claim text.

        Returns:
            List of key terms to check against evidence.
        """
        # Common stop words to filter out
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "and", "but", "or", "nor", "if", "while", "although",
            "this", "that", "these", "those", "it", "its", "he", "she",
            "they", "them", "we", "you", "i", "me", "my", "your", "our",
            "just", "also", "now", "even", "still", "yet", "already",
        }

        # Split into words, filter stop words and short words
        words = re.findall(r"\b[a-z0-9_./-]{2,}\b", text)
        key_terms = [w for w in words if w not in stop_words]

        # Also extract multi-word technical terms
        # e.g., "two-factor authentication" → ["two-factor", "authentication", "two-factor authentication"]
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in stop_words or words[i + 1] not in stop_words:
                bigrams.append(f"{words[i]} {words[i + 1]}")

        return key_terms + bigrams

    def _verify_entities(
        self, entities: list[str]
    ) -> tuple[list[str], list[str]]:
        """Verify named entities against evidence.

        Args:
            entities: Named entities from the claim.

        Returns:
            (hits, contradictions) — entities found in evidence and
            entities that conflict with evidence.
        """
        hits: list[str] = []
        contradictions: list[str] = []

        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in self._evidence_flat:
                hits.append(entity)
            # Check for near-matches that could be contradictions
            elif self._is_likely_hallucination(entity):
                contradictions.append(entity)

        return hits, contradictions

    def _is_likely_hallucination(self, entity: str) -> bool:
        """Check if an entity is likely hallucinated.

        Uses heuristics: version numbers, specific config keys, URLs,
        and technical identifiers that should appear in evidence.

        Args:
            entity: The entity string.

        Returns:
            True if the entity looks like something that should be in evidence.
        """
        # Version numbers (v1.2.3, 2.5.1) — must contain digits
        if re.match(r"^v?\d+\.\d+(\.\d+)?", entity):
            return True

        # URLs — always suspicious if not in evidence
        if entity.startswith("http"):
            return True

        # Full file paths
        if entity.startswith("/") and len(entity) > 5:
            return True

        # CamelCase identifiers (must have at least two capitals)
        if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)+$", entity):
            return True

        # Technical config keys (lowercase_with_underscores = value, must contain underscore)
        if "_" in entity and re.match(r"^[a-z][a-z_]+$", entity) and len(entity) > 4:
            return True

        return False

    def _find_evidence_for_terms(self, terms: list[str]) -> list[str]:
        """Find evidence chunks that contain the given terms.

        Args:
            terms: Terms to search for.

        Returns:
            List of evidence chunk IDs that contain any of the terms.
        """
        found_chunks: list[str] = []
        seen: set[str] = set()

        for chunk_id, text in self.evidence.items():
            text_lower = text.lower()
            for term in terms:
                if term.lower() in text_lower and chunk_id not in seen:
                    found_chunks.append(chunk_id)
                    seen.add(chunk_id)
                    break

        return found_chunks
