"""
Multi-provider pricing catalog for LLM token costs.

Models real-world pricing structures across major inference providers.
All prices are per 1M tokens (standard industry convention). Supports:
- Standard input/output pricing
- Cached input pricing (Anthropic, OpenAI prompt caching)
- Batch processing discounts (OpenAI 50% off, Anthropic 50% off)
- Rate limits (TPM/RPM by tier)

Price data reflects publicly documented pricing tiers as of mid-2025.
Module is designed to be updateable as providers change their pricing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass(frozen=True)
class PricingTier:
    """Rate limits for a pricing tier (tokens per minute, requests per minute)."""

    tpm: int
    rpm: int

    @property
    def tps(self) -> float:
        """Tokens per second (derived from TPM)."""
        return self.tpm / 60.0


@dataclass(frozen=True)
class ModelPricing:
    """Complete pricing profile for a single model.

    All prices in USD per 1M tokens.
    Rate limits are per-minute caps at the standard tier.
    """

    provider: str
    model_id: str
    display_name: str
    price_input: float  # USD per 1M input tokens
    price_output: float  # USD per 1M output tokens
    price_cached_input: float = 0.0  # Cached prompt tokens (Anthropic, OpenAI)
    batch_discount: float = 0.0  # Fractional discount for batch processing (0.0-1.0)
    rate_limit: Optional[PricingTier] = None
    context_window: int = 128_000
    supports_caching: bool = False
    supports_vision: bool = False
    supports_tools: bool = True

    @property
    def price_batch_input(self) -> float:
        """Input price with batch discount applied."""
        return self.price_input * (1.0 - self.batch_discount)

    @property
    def price_batch_output(self) -> float:
        """Output price with batch discount applied."""
        return self.price_output * (1.0 - self.batch_discount)

    def cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        """Compute total cost for a given token usage.

        Args:
            input_tokens: Number of (non-cached) input tokens.
            output_tokens: Number of output tokens.
            cached_input_tokens: Number of cached input tokens (prompt cache hit).
            batch: Whether to apply batch processing discount.

        Returns:
            Total cost in USD.
        """
        if batch:
            p_in = self.price_batch_input
            p_out = self.price_batch_output
            # Cached tokens in batch mode: typically no additional discount beyond batch
            p_cached = self.price_cached_input * (1.0 - self.batch_discount) if self.price_cached_input else 0
        else:
            p_in = self.price_input
            p_out = self.price_output
            p_cached = self.price_cached_input

        total = 0.0
        if input_tokens > 0:
            total += (input_tokens / 1_000_000) * p_in
        if output_tokens > 0:
            total += (output_tokens / 1_000_000) * p_out
        if cached_input_tokens > 0 and p_cached > 0:
            total += (cached_input_tokens / 1_000_000) * p_cached

        return round(total, 6)


# ── Provider Catalogs ──────────────────────────────────────────────────


class ProviderCatalog:
    """Registry of model pricing entries, organized by provider.

    Usage:
        >>> catalog = ProviderCatalog.default()
        >>> pricing = catalog.get("claude-sonnet-4-20250514")
        >>> cost = pricing.cost(input_tokens=1000, output_tokens=500)
        >>> print(f"${cost:.4f}")
    """

    _models: ClassVar[dict[str, ModelPricing]] = {}

    @classmethod
    def default(cls) -> ProviderCatalog:
        """Return the default catalog with all known providers loaded."""
        catalog = ProviderCatalog()
        catalog._register_many(_OPENAI_MODELS)
        catalog._register_many(_ANTHROPIC_MODELS)
        catalog._register_many(_COHERE_MODELS)
        catalog._register_many(_MISTRAL_MODELS)
        catalog._register_many(_GOOGLE_MODELS)
        catalog._register_many(_GROQ_MODELS)
        catalog._register_many(_DEEPSEEK_MODELS)
        return catalog

    def _register_many(self, models: list[ModelPricing]) -> None:
        for m in models:
            self._models[m.model_id] = m

    def get(self, model_id: str) -> Optional[ModelPricing]:
        """Look up pricing by model ID."""
        return self._models.get(model_id)

    def list_providers(self) -> list[str]:
        """List unique providers in the catalog."""
        return sorted(set(m.provider for m in self._models.values()))

    def list_models(self, provider: Optional[str] = None) -> list[ModelPricing]:
        """List all models, optionally filtered by provider."""
        models = list(self._models.values())
        if provider:
            models = [m for m in models if m.provider.lower() == provider.lower()]
        return sorted(models, key=lambda m: (m.provider, m.price_input))

    def __len__(self) -> int:
        return len(self._models)


# ── Convenience Functions ──────────────────────────────────────────────

# Singleton catalog — initialized lazily.
_catalog: Optional[ProviderCatalog] = None


def _get_catalog() -> ProviderCatalog:
    global _catalog
    if _catalog is None:
        _catalog = ProviderCatalog.default()
    return _catalog


def get_pricing(model_id: str) -> Optional[ModelPricing]:
    """Get pricing information for a specific model.

    Args:
        model_id: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514').

    Returns:
        ModelPricing if found, None otherwise.
    """
    return _get_catalog().get(model_id)


def get_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Compute the cost for a specific model and token usage.

    Args:
        model_id: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_input_tokens: Cached/prompt-cache-hit input tokens.
        batch: Apply batch pricing discount.

    Returns:
        Cost in USD, or 0.0 if model not found.
    """
    pricing = _get_catalog().get(model_id)
    if pricing is None:
        return 0.0
    return pricing.cost(input_tokens, output_tokens, cached_input_tokens, batch)


def list_models(provider: Optional[str] = None) -> list[ModelPricing]:
    """List all registered models, optionally filtered by provider."""
    return _get_catalog().list_models(provider)


def estimate_tokens(text: str, model_id: str = "gpt-4o") -> int:
    """Rough token count estimate (4 chars ≈ 1 token for English text).

    This is a fast heuristic; for precise token counts use a tokenizer.
    Sufficient for cost estimation where ±20% is acceptable.

    Args:
        text: Input text to estimate.
        model_id: Model identifier (unused in heuristic mode, retained for API compatibility).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Rough heuristic: ~4 characters per token for English text.
    # This is conservative (overestimates slightly for long words,
    # underestimates for code/whitespace).
    return max(1, math.ceil(len(text) / 4.0))


# ── Pricing Data ───────────────────────────────────────────────────────

_OPENAI_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="OpenAI",
        model_id="gpt-4o",
        display_name="GPT-4o",
        price_input=2.50,
        price_output=10.00,
        price_cached_input=1.25,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=450_000, rpm=500),
        context_window=128_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="OpenAI",
        model_id="gpt-4o-mini",
        display_name="GPT-4o mini",
        price_input=0.15,
        price_output=0.60,
        price_cached_input=0.075,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=2_000_000, rpm=500),
        context_window=128_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="OpenAI",
        model_id="o3",
        display_name="o3",
        price_input=10.00,
        price_output=40.00,
        price_cached_input=5.00,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=150_000, rpm=50),
        context_window=200_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="OpenAI",
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        price_input=2.00,
        price_output=8.00,
        price_cached_input=1.00,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=300_000, rpm=300),
        context_window=1_000_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="OpenAI",
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        price_input=0.40,
        price_output=1.60,
        price_cached_input=0.20,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=1_000_000, rpm=300),
        context_window=1_000_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="OpenAI",
        model_id="o4-mini",
        display_name="o4-mini",
        price_input=1.10,
        price_output=4.40,
        price_cached_input=0.55,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=250_000, rpm=100),
        context_window=200_000,
        supports_caching=True,
    ),
]

_ANTHROPIC_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="Anthropic",
        model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        price_input=15.00,
        price_output=75.00,
        price_cached_input=3.75,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=20_000, rpm=50),
        context_window=200_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="Anthropic",
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        price_input=3.00,
        price_output=15.00,
        price_cached_input=0.75,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=80_000, rpm=100),
        context_window=200_000,
        supports_caching=True,
        supports_vision=True,
    ),
    ModelPricing(
        provider="Anthropic",
        model_id="claude-haiku-3-5-20241022",
        display_name="Claude 3.5 Haiku",
        price_input=0.80,
        price_output=4.00,
        price_cached_input=0.20,
        batch_discount=0.50,
        rate_limit=PricingTier(tpm=200_000, rpm=200),
        context_window=200_000,
        supports_caching=True,
        supports_vision=False,
    ),
]

_COHERE_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="Cohere",
        model_id="command-r-plus",
        display_name="Command R+",
        price_input=2.50,
        price_output=10.00,
        batch_discount=0.25,
        context_window=128_000,
    ),
    ModelPricing(
        provider="Cohere",
        model_id="command-r",
        display_name="Command R",
        price_input=0.50,
        price_output=1.50,
        batch_discount=0.25,
        context_window=128_000,
    ),
    ModelPricing(
        provider="Cohere",
        model_id="command-a",
        display_name="Command A",
        price_input=0.15,
        price_output=0.60,
        batch_discount=0.25,
        context_window=256_000,
    ),
]

_MISTRAL_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="Mistral",
        model_id="mistral-large-2",
        display_name="Mistral Large 2",
        price_input=2.00,
        price_output=6.00,
        batch_discount=0.0,
        context_window=128_000,
    ),
    ModelPricing(
        provider="Mistral",
        model_id="mistral-small",
        display_name="Mistral Small",
        price_input=0.10,
        price_output=0.30,
        batch_discount=0.0,
        context_window=32_000,
    ),
    ModelPricing(
        provider="Mistral",
        model_id="codestral",
        display_name="Codestral",
        price_input=0.30,
        price_output=0.90,
        batch_discount=0.0,
        context_window=256_000,
    ),
    ModelPricing(
        provider="Mistral",
        model_id="pixtral-large",
        display_name="Pixtral Large",
        price_input=2.00,
        price_output=6.00,
        batch_discount=0.0,
        context_window=128_000,
        supports_vision=True,
    ),
]

_GOOGLE_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="Google",
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        price_input=1.25,
        price_output=10.00,
        rate_limit=PricingTier(tpm=100_000, rpm=60),
        context_window=1_000_000,
        supports_vision=True,
    ),
    ModelPricing(
        provider="Google",
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        price_input=0.15,
        price_output=0.60,
        rate_limit=PricingTier(tpm=2_000_000, rpm=200),
        context_window=1_000_000,
        supports_vision=True,
    ),
]

_GROQ_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="Groq",
        model_id="llama-3.3-70b",
        display_name="Llama 3.3 70B",
        price_input=0.59,
        price_output=0.79,
        rate_limit=PricingTier(tpm=30_000, rpm=30),
        context_window=128_000,
    ),
    ModelPricing(
        provider="Groq",
        model_id="mixtral-8x7b",
        display_name="Mixtral 8x7B",
        price_input=0.24,
        price_output=0.24,
        rate_limit=PricingTier(tpm=30_000, rpm=30),
        context_window=32_000,
    ),
    ModelPricing(
        provider="Groq",
        model_id="deepseek-r1-distill-llama-70b",
        display_name="DeepSeek R1 Distill Llama 70B",
        price_input=0.75,
        price_output=0.99,
        rate_limit=PricingTier(tpm=30_000, rpm=30),
        context_window=128_000,
    ),
]

_DEEPSEEK_MODELS: list[ModelPricing] = [
    ModelPricing(
        provider="DeepSeek",
        model_id="deepseek-v3",
        display_name="DeepSeek V3",
        price_input=0.27,
        price_output=1.10,
        rate_limit=PricingTier(tpm=100_000, rpm=50),
        context_window=128_000,
    ),
    ModelPricing(
        provider="DeepSeek",
        model_id="deepseek-r1",
        display_name="DeepSeek R1",
        price_input=0.55,
        price_output=2.19,
        rate_limit=PricingTier(tpm=50_000, rpm=30),
        context_window=128_000,
    ),
]
