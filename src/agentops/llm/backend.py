"""
Pluggable LLM backends for AgentOps.

Provides abstract and concrete LLM backends that support multiple providers
(OpenAI, Anthropic, DeepSeek, and OpenAI-compatible endpoints) without
requiring langchain. All backends share a common interface and are
configurable via environment variables.

Environment variables:
    OPENAI_API_KEY      — OpenAI API key
    OPENAI_BASE_URL     — Custom base URL (for DeepSeek, local models, etc.)
    ANTHROPIC_API_KEY   — Anthropic API key

Usage:
    from agentops.llm import create_backend, OpenAIBackend, AnthropicBackend

    # Auto-detect from environment
    backend = create_backend()

    # Explicit provider
    backend = OpenAIBackend(model="gpt-4o")
    backend = AnthropicBackend(model="claude-3-sonnet-20240229")
    backend = OpenAIBackend(model="deepseek-chat", base_url="https://api.deepseek.com/v1")

    # Use the backend
    response = backend.chat("Hello, world!")
    response = backend.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Structured response from an LLM backend call."""

    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    raw_response: Any = None

    @property
    def total_cost(self) -> float:
        return self.cost_usd


class LLMBackend(ABC):
    """Abstract base for LLM backends.

    All backends must implement `_chat_impl` and provide cost tracking.
    """

    def __init__(self, model: str, provider: str, temperature: float = 0.0,
                 max_tokens: int = 4096):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    @abstractmethod
    def _chat_impl(self, messages: list[dict[str, str]],
                   **kwargs: Any) -> LLMResponse:
        """Provider-specific chat implementation."""
        ...

    def chat(self, prompt: str | list[dict[str, str]],
             system: str | None = None,
             **kwargs: Any) -> LLMResponse:
        """Send a chat completion request.

        Accepts either a plain string prompt or a list of message dicts.
        Returns a structured LLMResponse with content, tokens, cost, and latency.
        """
        if isinstance(prompt, str):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = list(prompt)

        t0 = time.perf_counter()
        response = self._chat_impl(messages, **kwargs)
        latency = (time.perf_counter() - t0) * 1000
        response.latency_ms = latency

        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens
        self._total_cost += response.cost_usd
        self._call_count += 1

        return response

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_stats(self) -> None:
        """Reset accumulated token and cost counters."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    def health_check(self) -> bool:
        """Quick check that the backend can reach its API."""
        try:
            response = self.chat("ping", max_tokens=5)
            return bool(response.content)
        except Exception:
            return False


# ── OpenAI Backend ────────────────────────────────────────────────────

# Pricing per 1M tokens (input, output) as of 2025-2026
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # DeepSeek (via OpenAI-compatible endpoint)
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


class OpenAIBackend(LLMBackend):
    """OpenAI and OpenAI-compatible backend.

    Supports any provider with an OpenAI-compatible API (DeepSeek,
    Groq, Together, local vLLM/Ollama, etc.) by setting `base_url`.

    Requires: pip install openai
    Environment: OPENAI_API_KEY, OPENAI_BASE_URL (optional)
    """

    def __init__(self, model: str = "gpt-4o",
                 base_url: str | None = None,
                 api_key: str | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        provider = "openai"
        if base_url and "deepseek" in base_url:
            provider = "deepseek"
        super().__init__(model=model, provider=provider,
                         temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _chat_impl(self, messages: list[dict[str, str]],
                   **kwargs: Any) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Cost calculation
        pricing = OPENAI_PRICING.get(self.model, (2.50, 10.00))
        cost = (input_tokens / 1_000_000) * pricing[0] + \
               (output_tokens / 1_000_000) * pricing[1]

        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=cost,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )


# ── Anthropic Backend ─────────────────────────────────────────────────

ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend.

    Requires: pip install anthropic
    Environment: ANTHROPIC_API_KEY
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: str | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        super().__init__(model=model, provider="anthropic",
                         temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _chat_impl(self, messages: list[dict[str, str]],
                   **kwargs: Any) -> LLMResponse:
        # Anthropic extracts system message from the array
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        response = self.client.messages.create(
            model=self.model,
            system=system_msg if system_msg else None,
            messages=user_messages,
            temperature=self.temperature,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        pricing = ANTHROPIC_PRICING.get(self.model, (3.00, 15.00))
        cost = (input_tokens / 1_000_000) * pricing[0] + \
               (output_tokens / 1_000_000) * pricing[1]

        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        return LLMResponse(
            content=content,
            model=self.model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            finish_reason=response.stop_reason or "stop",
            raw_response=response,
        )


# ── Factory ───────────────────────────────────────────────────────────

def create_backend(model: str | None = None,
                   provider: str | None = None,
                   **kwargs: Any) -> LLMBackend:
    """Create an LLM backend auto-detected from environment.

    Detection order:
    1. If `provider` is explicitly given, use that.
    2. If ANTHROPIC_API_KEY is set and no OPENAI_API_KEY, use Anthropic.
    3. If OPENAI_API_KEY is set, use OpenAI (which also covers DeepSeek
       when OPENAI_BASE_URL points to a DeepSeek endpoint).
    4. If neither key is set, raise ValueError.

    Args:
        model: Model name override.
        provider: Force a specific provider ("openai", "anthropic", "deepseek").
        **kwargs: Passed to the backend constructor.

    Returns:
        An initialized LLMBackend instance.
    """
    if provider:
        provider = provider.lower()

    if provider == "anthropic":
        return AnthropicBackend(
            model=model or "claude-3-5-sonnet-20241022", **kwargs)
    elif provider in ("openai", "deepseek"):
        base_url = kwargs.pop("base_url", None)
        if provider == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com/v1"
        return OpenAIBackend(
            model=model or ("deepseek-chat" if provider == "deepseek" else "gpt-4o"),
            base_url=base_url, **kwargs)

    # Auto-detect
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    if has_openai:
        is_deepseek = "deepseek" in base_url.lower()
        return OpenAIBackend(
            model=model or ("deepseek-chat" if is_deepseek else "gpt-4o"),
            base_url=base_url or None, **kwargs)
    elif has_anthropic:
        return AnthropicBackend(
            model=model or "claude-3-5-sonnet-20241022", **kwargs)
    else:
        raise ValueError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "in the environment. For DeepSeek, set OPENAI_API_KEY and "
            "OPENAI_BASE_URL=https://api.deepseek.com/v1")


def available_backends() -> list[str]:
    """Return list of available backend provider names based on environment."""
    available = []
    if os.environ.get("OPENAI_API_KEY"):
        base = os.environ.get("OPENAI_BASE_URL", "")
        if "deepseek" in base.lower():
            available.append("deepseek")
        else:
            available.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    return available


def check_availability() -> dict[str, bool]:
    """Return a dict of provider -> available (based on env keys)."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "deepseek": bool(os.environ.get("OPENAI_API_KEY")
                         and "deepseek" in os.environ.get("OPENAI_BASE_URL", "").lower()),
    }
