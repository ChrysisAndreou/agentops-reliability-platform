"""
Claim extraction from streaming text.

Extracts factual claims from streaming LLM output at the sentence level.
Handles partial sentences that span chunk boundaries, detects whether text
is a factual claim vs. transitional/structural text, and extracts named
entities for entity-level verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .state import StreamingClaim, StreamingConfig

# ── Patterns ────────────────────────────────────────────────────────────

# Sentence boundary patterns (ordered by priority)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Factual claim indicators — sentences starting with these are likely claims
_CLAIM_STARTERS = re.compile(
    r"^(The|A|An|It|This|That|These|Those|There|"
    r"[A-Z][a-z]+ (is|are|was|were|has|have|had|will|would|can|could|should|must|may|might|"
    r"provides|offers|supports|enables|allows|requires|needs|uses|"
    r"consists|contains|includes|features|"
    r"you|users|administrators|developers|engineers))",
    re.IGNORECASE,
)

# Non-claim patterns — sentences matching these are NOT factual claims
_NON_CLAIM_PATTERNS = [
    re.compile(r"^(Hi|Hello|Hey|Dear|Greetings|Welcome)", re.IGNORECASE),
    re.compile(r"^(Sure|Of course|Absolutely|Certainly|Happy to|I('ll| will| can))", re.IGNORECASE),
    re.compile(r"^(Let me|I('ll| will) (try|attempt|see|check|look|find|search|get|grab|pull))", re.IGNORECASE),
    re.compile(r"^(How|What|Why|When|Where|Who|Which|Is|Are|Do|Does|Can|Could|Would|Should|May)", re.IGNORECASE),
    re.compile(r"^(Thank|Thanks|Please|Note|Important|Warning|Caution|Tip|Pro tip)", re.IGNORECASE),
    re.compile(r"^(First|Second|Third|Finally|Lastly|Next|Then|Now|Also|Additionally|Furthermore|Moreover|However|Therefore|Thus|Hence|So|Because|Since|Although|While|Whereas|But|Yet|Still|Instead|Otherwise|Meanwhile|Consequently)", re.IGNORECASE),
    re.compile(r"^(In (conclusion|summary|other words|short|brief|essence|general|particular|addition|contrast|comparison|practice|theory|this case|that case|the following|the above))", re.IGNORECASE),
    re.compile(r"^(As (a result|an example|mentioned|noted|shown|described|discussed|you can see|we can see|I mentioned))", re.IGNORECASE),
    re.compile(r"^(Here('s| is)|Below|Above|The following)", re.IGNORECASE),
    re.compile(r"^[\"'`(<\[]", re.IGNORECASE),  # Starts with quote, paren, bracket
    re.compile(r"^(To (do|get|start|begin|use|enable|configure|set up|install|run|execute))", re.IGNORECASE),
]

# Entity patterns for entity tracking
_ENTITY_PATTERNS = [
    # Configuration keys / parameters
    re.compile(r"\b([a-z_]+)\s*=\s*[\"'`]?([^\"'`,\s]+)[\"'`]?", re.IGNORECASE),
    # Versions
    re.compile(r"\b(v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)\b"),
    # File paths
    re.compile(r"\b(/(?:[a-zA-Z0-9._-]+/)*[a-zA-Z0-9._-]+)\b"),
    # URLs
    re.compile(r"\b(https?://[^\s]+)\b"),
    # Uppercase acronyms
    re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b"),
    # Tool/API names
    re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"),  # CamelCase identifiers
]

# Sentence terminator characters that should not be split mid-sentence
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "etc", "vs", "fig", "eq", "dept", "approx", "inc", "ltd", "co",
    "i.e", "e.g", "al", "no", "vol", "pg", "pp", "ch", "sec",
}


@dataclass
class ClaimExtractor:
    """Extracts factual claims from streaming text chunks.

    Buffers partial sentences across chunk boundaries, classifies each
    sentence as a factual claim or non-claim, and extracts named entities.

    Attributes:
        config: Streaming configuration.
    """

    config: StreamingConfig = field(default_factory=StreamingConfig)
    _buffer: str = ""
    _claim_indices: int = 0
    _total_chunks: int = 0

    def process_chunk(self, text: str) -> list[StreamingClaim]:
        """Process a new text chunk and extract any completed claims.

        Args:
            text: Raw text chunk from the LLM stream.

        Returns:
            List of completed claims extracted from this chunk + buffer.
        """
        self._total_chunks += 1
        self._buffer += text

        claims: list[StreamingClaim] = []

        # Extract complete sentences
        while True:
            sentence, remainder = self._extract_next_sentence(self._buffer)
            if sentence is None:
                break

            self._buffer = remainder

            # Classify and extract claims
            if self._is_factual_claim(sentence):
                claim = self._create_claim(sentence)
                claims.append(claim)

        # Force extraction if buffer exceeds max
        if len(self._buffer) > self.config.max_buffer_chars:
            remaining = self._buffer.strip()
            if len(remaining) >= self.config.min_claim_length:
                if self._is_factual_claim(remaining):
                    claim = self._create_claim(remaining, is_partial=True)
                    claims.append(claim)
            self._buffer = ""

        return claims

    def flush(self) -> list[StreamingClaim]:
        """Extract any remaining claims from the buffer at stream end.

        Returns:
            Any claims still in the buffer.
        """
        claims: list[StreamingClaim] = []
        remaining = self._buffer.strip()

        if len(remaining) >= self.config.min_claim_length:
            # Split remaining text into sentences
            parts = _SENTENCE_END.split(remaining)
            for part in parts:
                part = part.strip()
                if len(part) >= self.config.min_claim_length:
                    if self._is_factual_claim(part):
                        claim = self._create_claim(part, is_partial=False)
                        claims.append(claim)

        self._buffer = ""
        return claims

    # ── Private helpers ──────────────────────────────────────────────

    def _extract_next_sentence(self, text: str) -> tuple[str | None, str]:
        """Extract the next complete sentence from text.

        Returns:
            (sentence, remainder) or (None, text) if no complete sentence.
        """
        # Find sentence boundaries
        matches = list(_SENTENCE_END.finditer(text))
        if not matches:
            return None, text

        for match in matches:
            end = match.end()
            candidate = text[:end].strip()

            # Check if the split point is after an abbreviation
            if self._is_abbreviation_break(candidate):
                continue

            return candidate, text[end:]

        return None, text

    def _is_abbreviation_break(self, candidate: str) -> bool:
        """Check if a sentence boundary is actually an abbreviation."""
        words = candidate.split()
        if not words:
            return False
        last_word = words[-1].rstrip(".!?").lower()
        return last_word in _ABBREVIATIONS

    def _is_factual_claim(self, text: str) -> bool:
        """Determine if text is a factual claim vs. transitional/structural text.

        A factual claim makes an assertion about the world that can be
        verified against evidence.

        Args:
            text: The sentence to classify.

        Returns:
            True if this is a factual claim.
        """
        stripped = text.strip()
        if len(stripped) < self.config.min_claim_length:
            return False

        # Check non-claim patterns first
        for pattern in _NON_CLAIM_PATTERNS:
            if pattern.match(stripped):
                return False

        # Heuristic: factual claims start with claim-like patterns
        if _CLAIM_STARTERS.match(stripped):
            return True

        # Default: if it's substantial text, treat as potential claim
        return len(stripped) >= self.config.min_claim_length * 2

    def _create_claim(
        self, text: str, is_partial: bool = False
    ) -> StreamingClaim:
        """Create a StreamingClaim from text.

        Args:
            text: The claim text.
            is_partial: Whether this claim may span into future chunks.

        Returns:
            A new StreamingClaim.
        """
        entities = self._extract_entities(text) if self.config.track_entities else []

        self._claim_indices += 1
        start_char = max(0, len(self._buffer) - len(text) - 1)

        return StreamingClaim(
            text=text.strip(),
            chunk_indices=[self._total_chunks],
            confidence=self._claim_confidence(text),
            entities=entities,
            is_partial=is_partial,
            start_char=start_char,
            end_char=start_char + len(text),
        )

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """Extract named entities from claim text.

        Args:
            text: The claim text.

        Returns:
            List of entity strings found.
        """
        entities: list[str] = []
        for pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                entity = match.group(0)
                if entity not in entities:
                    entities.append(entity)
        return entities

    @staticmethod
    def _claim_confidence(text: str) -> float:
        """Estimate confidence that text is a factual claim.

        Heuristic based on claim indicators, length, and structure.

        Args:
            text: The claim text.

        Returns:
            Confidence score (0.0–1.0).
        """
        score = 0.5
        stripped = text.strip()

        if _CLAIM_STARTERS.match(stripped):
            score += 0.2

        # Longer text with specific details is more likely a claim
        words = stripped.split()
        if len(words) > 10:
            score += 0.1
        if len(words) > 20:
            score += 0.1

        # Specific numbers and technical terms increase confidence
        if re.search(r"\d+", stripped):
            score += 0.05
        if re.search(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+", stripped):
            score += 0.05

        return min(score, 1.0)
