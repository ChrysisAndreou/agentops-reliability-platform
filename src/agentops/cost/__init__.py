"""
Cost Economics & Token Optimization — production cost management for AI agents.

Provides multi-provider pricing models, hierarchical cost tracking with budget
enforcement, optimization strategies for reducing token spend, and cost
efficiency evaluation with quality-vs-cost trade-off analysis.

This module fills a critical gap in the AgentOps platform: understanding not
just whether agents work correctly (reliability, safety, evaluation), but
whether they work *efficiently*. For production AI systems, cost management
is as important as correctness — uncontrolled token spend can make an otherwise
successful agent economically nonviable.

Modules:
    pricing: Multi-provider pricing catalog — 20+ model entries across OpenAI,
             Anthropic, Cohere, Mistral, Google, Groq, and DeepSeek. Supports
             input/output/cached token pricing, rate limits, batch discounts.
    tracker: Hierarchical cost tracking — call → agent → session → project
             accounting with soft/hard budget limits, real-time budget checking,
             and structured cost breakdowns.
    optimizer: Cost optimization strategies — prompt caching detection, context
               window pruning, model routing (expensive→cheap fallback), tool
               call bundling, and what-if cost simulation.
    eval: Cost efficiency evaluation — cost-per-task, cost-per-success metrics,
          budget compliance reporting, cost-vs-quality trade-off analysis, and
          cost regression detection.

Example usage:
    >>> from agentops.cost import CostTracker, get_cost, optimize
    >>> tracker = CostTracker(budget_daily=5.00)
    >>> tracker.record("claude-sonnet-4", input_tokens=1500, output_tokens=300)
    >>> tracker.record("claude-sonnet-4", input_tokens=2200, output_tokens=450)
    >>> print(tracker.summary())
    >>> if tracker.budget_exceeded():
    ...     savings = optimize.bundle_tool_calls(agent_trace)
"""

from agentops.cost.pricing import (
    ModelPricing,
    PricingTier,
    ProviderCatalog,
    get_cost,
    get_pricing,
    list_models,
    estimate_tokens,
)

from agentops.cost.tracker import (
    Budget,
    BudgetAlert,
    BudgetStatus,
    CostBreakdown,
    CostRecord,
    CostTracker,
    TokenCounter,
)

from agentops.cost.optimizer import (
    CacheableRegion,
    ContextPruner,
    ModelRouter,
    OptimizationPlan,
    Optimizer,
    ToolBundler,
    estimate_cache_savings,
)

from agentops.cost.eval import (
    BudgetComplianceReport,
    CostEfficiencyMetrics,
    CostEvalReport,
    CostRegression,
    CostVsQuality,
    evaluate_cost_efficiency,
    format_cost_report,
)

__all__ = [
    # Pricing
    "ModelPricing",
    "PricingTier",
    "ProviderCatalog",
    "get_cost",
    "get_pricing",
    "list_models",
    "estimate_tokens",
    # Tracker
    "Budget",
    "BudgetAlert",
    "BudgetStatus",
    "CostBreakdown",
    "CostRecord",
    "CostTracker",
    "TokenCounter",
    # Optimizer
    "CacheableRegion",
    "ContextPruner",
    "ModelRouter",
    "OptimizationPlan",
    "Optimizer",
    "ToolBundler",
    "estimate_cache_savings",
    # Evaluation
    "BudgetComplianceReport",
    "CostEfficiencyMetrics",
    "CostEvalReport",
    "CostRegression",
    "CostVsQuality",
    "evaluate_cost_efficiency",
    "format_cost_report",
]
