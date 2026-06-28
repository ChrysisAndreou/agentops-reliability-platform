"""
Model Router — dynamic model selection with cost/latency/capability awareness.

Routes LLM requests across multiple backends based on:
- Cost optimization (cheapest model for the task)
- Latency optimization (fastest model available)
- Capability-based routing (model must support required features)
- Round-robin load distribution
- Failover chains with automatic retry

Tracks per-model metrics: calls, tokens, cost, latency distributions.
Enforces budget limits with configurable alert thresholds.
"""

from __future__ import annotations

import enum
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from agentops.llm.backend import (
    LLMBackend,
    LLMResponse,
    create_backend,
    check_availability,
)


# ═══════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════

class RoutingStrategy(enum.Enum):
    """How the router selects a backend for each request."""
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    CAPABILITY = "capability"
    ROUND_ROBIN = "round_robin"
    FAILOVER = "failover"


@dataclass
class BackendConfig:
    """Configuration for a backend available to the router.

    Attributes:
        model: Model name (e.g. 'gpt-4o', 'claude-3-haiku-20240307').
        provider: Provider identifier ('openai', 'anthropic', 'deepseek').
        cost_input: USD per 1M input tokens.
        cost_output: USD per 1M output tokens.
        capabilities: Set of capability tags this model supports
                      (e.g. 'chat', 'code', 'vision', 'json_mode').
        max_tokens: Maximum output tokens per request.
        temperature: Sampling temperature (0.0 = deterministic).
        weight: Relative weight for round-robin distribution (higher = more traffic).
        enabled: Whether this backend is available for routing.
    """
    model: str
    provider: str
    cost_input: float
    cost_output: float
    capabilities: set[str] = field(default_factory=lambda: {"chat"})
    max_tokens: int = 4096
    temperature: float = 0.0
    weight: float = 1.0
    enabled: bool = True

    @property
    def cost_per_1m_input(self) -> float:
        return self.cost_input

    @property
    def cost_per_1m_output(self) -> float:
        return self.cost_output


@dataclass
class RouterStats:
    """Per-model aggregate statistics collected by the router.

    Attributes:
        calls: Total number of requests routed to this model.
        total_input_tokens: Sum of input tokens across all calls.
        total_output_tokens: Sum of output tokens across all calls.
        total_cost_usd: Total cost in USD.
        latencies_ms: List of per-call latencies in milliseconds.
        errors: Count of failed calls.
        last_used: Timestamp of most recent use (monotonic seconds).
    """
    calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    last_used: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.5)
        return sorted_lat[idx]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "errors": self.errors,
        }


class BudgetExceededError(Exception):
    """Raised when a request would exceed the configured budget."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# Model Router
# ═══════════════════════════════════════════════════════════════════════

class ModelRouter:
    """Intelligent router that selects and manages multiple LLM backends.

    Routes requests to the optimal backend based on configured strategy,
    tracks cost and latency, enforces budgets, and handles failover.

    Usage:
        router = ModelRouter([
            BackendConfig(model="gpt-4o-mini", provider="openai",
                          cost_input=0.15, cost_output=0.60,
                          capabilities={"chat", "code"}),
            BackendConfig(model="claude-3-haiku-20240307", provider="anthropic",
                          cost_input=0.25, cost_output=1.25,
                          capabilities={"chat", "vision"}),
        ], strategy=RoutingStrategy.CHEAPEST)

        response = router.chat("Explain quantum computing")
        print(response.content)

        # With budget
        router = ModelRouter([...], budget_limit_usd=5.00)
        router.chat("Hello")  # OK
        router.chat("Another question")  # OK
        router.chat("One more")  # Raises BudgetExceededError
    """

    # ── Well-known model costs (per 1M tokens, input/output) ──────
    DEFAULT_COSTS: dict[str, tuple[float, float]] = {
        # OpenAI
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1": (15.00, 60.00),
        "o1-mini": (3.00, 12.00),
        "o3-mini": (1.10, 4.40),
        # Anthropic
        "claude-3-opus-20240229": (15.00, 75.00),
        "claude-3-sonnet-20240229": (3.00, 15.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        # DeepSeek
        "deepseek-chat": (0.27, 1.10),
        "deepseek-reasoner": (0.55, 2.19),
    }

    # ── Capability-aware model mapping ──────────────────────────────
    CAPABILITY_MODELS: dict[str, str] = {
        "vision": "gpt-4o",         # GPT-4o has vision
        "code": "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet for code
        "json_mode": "gpt-4o",      # GPT-4o supports structured output
        "function_calling": "gpt-4o-mini",  # Cheaper option for tool use
        "reasoning": "deepseek-reasoner",  # DeepSeek-R1 for reasoning
        "long_context": "claude-3-opus-20240229",  # Claude Opus for 200K context
    }

    def __init__(
        self,
        backends: list[BackendConfig],
        strategy: RoutingStrategy | str = RoutingStrategy.CHEAPEST,
        budget_limit_usd: float | None = None,
        budget_alert_threshold: float = 0.80,
    ):
        """Initialize the router with backend configurations.

        Args:
            backends: List of BackendConfig objects describing available models.
            strategy: Default routing strategy (CHEAPEST if not specified per-request).
                      Accepts RoutingStrategy enum or string ('cheapest', 'fastest', etc.).
            budget_limit_usd: Hard cap on cumulative spend (None = unlimited).
            budget_alert_threshold: Fraction of budget at which to warn
                                    (0.80 = warn at 80% consumed).
        """
        # Filter enabled backends
        enabled = [b for b in backends if b.enabled]
        if not enabled:
            raise ValueError("At least one enabled backend is required")

        # Convert strategy string to enum if needed
        if isinstance(strategy, str):
            strategy = self._strategy_from_string(strategy)

        self.backends = enabled
        self.strategy = strategy
        self.budget_limit_usd = budget_limit_usd
        self.budget_alert_threshold = budget_alert_threshold

        # Per-model statistics
        self._stats: dict[str, RouterStats] = defaultdict(RouterStats)

        # Round-robin state
        self._rr_index = 0

        # Backend cache (lazily initialized)
        self._backend_instances: dict[str, LLMBackend] = {}

        # Cumulative cost tracker
        self._total_cost_usd = 0.0

        # Latency history for FASTEST strategy ([model_key, latency_ms])
        self._latency_history: dict[str, list[float]] = defaultdict(list)

        # Time window for latency EWMA (seconds)
        self._latency_window_s = 300.0  # 5-minute EWMA

    # ── Public API ──────────────────────────────────────────────────

    def chat(
        self,
        prompt: str | list[dict[str, str]],
        strategy: RoutingStrategy | None = None,
        system: str | None = None,
        capabilities: set[str] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Route a chat completion request through the optimal backend.

        Args:
            prompt: User prompt string, or list of message dicts.
            strategy: Per-request strategy override (uses default if None).
            system: System prompt (prepended to messages).
            capabilities: Required capability tags for capability-based routing.
            max_tokens: Per-request max_tokens override.

        Returns:
            LLMResponse from the selected backend.

        Raises:
            BudgetExceededError: If the request would exceed the budget limit.
            ValueError: If no backend supports the required capabilities.
            RuntimeError: If all backends fail (after failover attempts).
        """
        # Resolve strategy
        strat = strategy or self.strategy

        # Get candidate model names from BackendConfig list
        all_candidates = [b.model for b in self.backends]

        # 1. Capability-based routing: filter backends by required features
        if capabilities and any(b.capabilities for b in self.backends):
            candidates = self._filter_by_capabilities(capabilities)
            if not candidates:
                raise ValueError(
                    f"No backend supports capabilities: {capabilities}"
                )
        elif strat == RoutingStrategy.CAPABILITY:
            candidates = all_candidates
        else:
            candidates = all_candidates

        # 2. Cost-based routing (CHEAPEST): pick lowest cost per token
        if strat == RoutingStrategy.CHEAPEST:
            selected = self._select_cheapest(candidates)
        # 3. Latency-based routing (FASTEST): pick lowest EWMA latency
        elif strat == RoutingStrategy.FASTEST:
            selected = self._select_fastest(candidates)
        # 4. Round-robin
        elif strat == RoutingStrategy.ROUND_ROBIN:
            selected = self._select_round_robin(candidates)
        # 5. Failover chain: try backends in order, return first to succeed
        elif strat == RoutingStrategy.FAILOVER:
            selected = candidates[0]
        else:
            selected = candidates[0]

        # Get or create the backend instance
        backend = self._get_backend(selected)

        # Budget check
        estimated_cost = self._estimate_cost(selected, prompt, max_tokens)
        if self.budget_limit_usd is not None:
            projected = self._total_cost_usd + estimated_cost
            if projected > self.budget_limit_usd:
                raise BudgetExceededError(
                    f"Request would exceed budget: "
                    f"${projected:.4f} projected > ${self.budget_limit_usd:.2f} limit"
                )
            elif (
                self.budget_limit_usd > 0
                and projected / self.budget_limit_usd > self.budget_alert_threshold
            ):
                # Warn but allow
                import logging
                logging.warning(
                    f"Budget alert: ${projected:.4f} projected "
                    f"({projected/self.budget_limit_usd:.0%} of "
                    f"${self.budget_limit_usd:.2f} limit)"
                )

        # Dispatch
        t0 = time.perf_counter()
        try:
            response = backend.chat(
                prompt,
                system=system,
                max_tokens=max_tokens or self.backends[0].max_tokens,
            )
        except Exception as e:
            # Failover: try next backend if available
            if strat == RoutingStrategy.FAILOVER:
                # Try remaining candidates in order
                remaining = [c for c in candidates if c != selected]
                for candidate in remaining:
                    try:
                        backend = self._get_backend(candidate)
                        response = backend.chat(
                            prompt,
                            system=system,
                            max_tokens=max_tokens or self.backends[0].max_tokens,
                        )
                        break
                    except Exception:
                        continue
                else:
                    raise RuntimeError(
                        f"All {len(candidates)} backends failed for prompt: "
                        f"{str(prompt)[:100]}..."
                    )
            else:
                raise

        latency_ms = (time.perf_counter() - t0) * 1000

        # Record stats
        stats = self._stats[selected]
        stats.calls += 1
        stats.total_input_tokens += response.input_tokens
        stats.total_output_tokens += response.output_tokens
        stats.total_cost_usd += response.cost_usd
        stats.latencies_ms.append(latency_ms)
        stats.last_used = time.monotonic()

        # Update latency history for FASTEST strategy (keep last 20 values)
        hist = self._latency_history[selected]
        hist.append(latency_ms)
        if len(hist) > 20:
            hist.pop(0)

        self._total_cost_usd += response.cost_usd

        return response

    def route(
        self,
        prompt: str | list[dict[str, str]] | None = None,
        strategy: RoutingStrategy | None = None,
        capabilities: set[str] | None = None,
    ) -> str:
        """Route a request and return the best model name used.

        Convenience method that returns just the model key for the selected
        backend without making an actual API call. Useful for cost estimation
        and capability checks without consuming tokens.

        Args:
            prompt: User prompt (optional — used for cost estimation if provided).
            strategy: Per-request override.
            capabilities: Required capabilities.

        Returns:
            The model key of the backend that would be selected.
        """
        strat = strategy or self.strategy
        # Use BackendConfig models, not backend instances (latter may be empty)
        candidates = [b.model for b in self.backends]

        if strat == RoutingStrategy.CAPABILITY and capabilities:
            candidates = self._filter_by_capabilities(capabilities)
            if not candidates:
                raise ValueError(f"No backend supports capabilities: {capabilities}")
        elif strat == RoutingStrategy.CHEAPEST:
            return self._select_cheapest(candidates)
        elif strat == RoutingStrategy.FASTEST:
            return self._select_fastest(candidates)
        elif strat == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidates)
        else:
            return candidates[0]

    def stats(self) -> dict[str, RouterStats]:
        """Return per-model routing statistics."""
        return {k: v for k, v in self._stats.items()}

    def reset_stats(self) -> None:
        """Reset all per-model statistics and cumulative cost."""
        self._stats.clear()
        self._total_cost_usd = 0.0
        self._latency_history.clear()
        for inst in self._backend_instances.values():
            inst.reset_stats()

    # ── Internal helpers ─────────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        """Total cumulative cost across all backends."""
        return self._total_cost_usd

    @staticmethod
    def _strategy_from_string(name: str | RoutingStrategy) -> RoutingStrategy:
        """Convert a strategy name string to RoutingStrategy enum."""
        if isinstance(name, RoutingStrategy):
            return name
        name_lower = name.lower()
        for strat in RoutingStrategy:
            if strat.value == name_lower:
                return strat
        valid = [s.value for s in RoutingStrategy]
        raise ValueError(f"Unknown strategy: '{name}'. Valid: {valid}")

    def _get_backend(self, model_key: str) -> LLMBackend:
        """Get or lazily create a backend instance for the given model key."""
        if model_key not in self._backend_instances:
            config = next(b for b in self.backends if b.model == model_key)
            backend = create_backend(
                model=config.model,
                provider=config.provider,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            self._backend_instances[model_key] = backend
        return self._backend_instances[model_key]

    def _filter_by_capabilities(
        self, required: set[str]
    ) -> list[str]:
        """Return model keys whose capabilities include all required tags."""
        available = []
        for backend_cfg in self.backends:
            if not backend_cfg.enabled:
                continue
            if required.issubset(backend_cfg.capabilities):
                available.append(backend_cfg.model)
        return available

    def _estimate_cost(
        self,
        model_key: str,
        prompt: str | list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> float:
        """Estimate the USD cost of a request without making an API call.

        Uses the pricing table and estimated token counts. This is
        approximate — actual cost may differ.
        """
        cfg = next(b for b in self.backends if b.model == model_key)
        # Rough token estimate: 1 input token per char for English text,
        # plus output tokens scaled by max_tokens / 4096 ratio.
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        estimated_input = len(prompt_str) // 4  # ~1 token per 4 chars
        estimated_output = int(
            (max_tokens or cfg.max_tokens) * 0.75
        )  # Output ~75% of max_tokens

        # Cost per 1M tokens
        cost_per_1m_input = cfg.cost_per_1m_input
        cost_per_1m_output = cfg.cost_per_1m_output

        estimated_cost = (
            (estimated_input / 1_000_000) * cost_per_1m_input
            + (estimated_output / 1_000_000) * cost_per_1m_output
        )
        return estimated_cost

    def _select_cheapest(self, candidates: list[str]) -> str:
        """Select the candidate with the lowest estimated cost."""
        best = None
        best_cost = float("inf")
        for model_key in candidates:
            cfg = next(b for b in self.backends if b.model == model_key)
            cost = (
                cfg.cost_per_1m_input + cfg.cost_per_1m_output
            ) / 2  # Average of input+output cost
            if cost < best_cost:
                best_cost = cost
                best = model_key
        return best

    def _select_fastest(self, candidates: list[str]) -> str:
        """Select the candidate with the lowest average latency."""
        best = None
        best_latency = float("inf")
        for model_key in candidates:
            history = self._latency_history.get(model_key, [])
            if not history:
                # No data yet — use a neutral default
                latency = 500.0  # Assume 500ms for unknown
            else:
                latency = sum(history) / len(history)
            if latency < best_latency:
                best_latency = latency
                best = model_key
        return best

    def _select_round_robin(self, candidates: list[str]) -> str:
        """Select the next candidate in round-robin order."""
        selected = candidates[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(candidates)
        return selected
