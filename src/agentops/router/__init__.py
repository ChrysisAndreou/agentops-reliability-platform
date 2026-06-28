"""
AgentOps Model Router — Dynamic model selection and cost optimization.

Routes LLM requests across multiple providers/models based on configurable
strategies: cheapest, fastest, capability-based, round-robin, and failover.
Tracks per-model cost/latency/token metrics and enforces budget limits.

Usage:
    from agentops.router import ModelRouter, BackendConfig, RoutingStrategy

    router = ModelRouter([
        BackendConfig(model="gpt-4o-mini", provider="openai",
                      cost_input=0.15, cost_output=0.60,
                      capabilities={"chat", "code"}),
        BackendConfig(model="claude-3-haiku-20240307", provider="anthropic",
                      cost_input=0.25, cost_output=1.25,
                      capabilities={"chat", "vision"}),
    ], strategy=RoutingStrategy.CHEAPEST)

    response = router.chat("What is the capital of France?")
    print(router.stats())  # Per-model metrics
"""

from agentops.router.router import (
    BackendConfig,
    ModelRouter,
    RoutingStrategy,
    RouterStats,
    BudgetExceededError,
)

__all__ = [
    "BackendConfig",
    "ModelRouter",
    "RoutingStrategy",
    "RouterStats",
    "BudgetExceededError",
]
