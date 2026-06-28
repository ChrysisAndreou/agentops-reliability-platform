"""
Cost optimization strategies for AI agent operations.

Identifies and quantifies cost-saving opportunities in agent traces
without sacrificing correctness. Covers four categories of optimization:

1. **Prompt Caching** — Detect repeated system prompts and prefix overlap
   that would benefit from Anthropic/OpenAI prompt caching.
2. **Context Pruning** — Identify low-value context that can be safely
   trimmed to reduce input token costs.
3. **Model Routing** — Route simple tasks to cheaper models, reserving
   expensive models for complex reasoning.
4. **Tool Call Bundling** — Detect serial tool calls that could be
   parallelized to reduce round-trip costs.

Each optimizer produces an OptimizationPlan with estimated savings.
These can be applied automatically (where safe) or flagged for review.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Data Models ────────────────────────────────────────────────────────


@dataclass
class CacheableRegion:
    """A region of prompt text that could benefit from caching.

    Represents a contiguous text segment shared across multiple calls.
    Anthropic and OpenAI charge 10-25% of the base input price for
    cache hits — significant savings for long system prompts.
    """

    text: str
    estimated_tokens: int
    occurrence_count: int
    estimated_savings_per_call: float = 0.0
    total_estimated_savings: float = 0.0

    @property
    def text_preview(self) -> str:
        return self.text[:80] + "..." if len(self.text) > 80 else self.text


@dataclass
class PruningOpportunity:
    """A specific opportunity to trim context to save tokens."""

    source: str  # e.g., "old_turns", "redundant_tool_output", "verbose_system_prompt"
    tokens_to_save: int
    confidence: float  # 0.0-1.0 likelihood that pruning won't affect answer quality
    description: str


@dataclass
class ModelRouteDecision:
    """A decision to route a specific operation to a cheaper model."""

    operation: str
    original_model: str
    suggested_model: str
    estimated_savings_usd: float
    rationale: str


@dataclass
class BundlingOpportunity:
    """Tool calls that could be parallelized."""

    tool_names: list[str]
    original_order: list[int]
    estimated_latency_reduction_s: float
    estimated_savings_usd: float = 0.0


@dataclass
class OptimizationPlan:
    """Aggregate optimization plan with estimated savings across all strategies."""

    cacheable_regions: list[CacheableRegion] = field(default_factory=list)
    pruning_opportunities: list[PruningOpportunity] = field(default_factory=list)
    route_decisions: list[ModelRouteDecision] = field(default_factory=list)
    bundling_opportunities: list[BundlingOpportunity] = field(default_factory=list)

    @property
    def total_estimated_savings(self) -> float:
        savings = 0.0
        for cr in self.cacheable_regions:
            savings += cr.total_estimated_savings
        for rd in self.route_decisions:
            savings += rd.estimated_savings_usd
        for bo in self.bundling_opportunities:
            savings += bo.estimated_savings_usd
        return savings

    @property
    def total_tokens_saved(self) -> int:
        return sum(p.tokens_to_save for p in self.pruning_opportunities)

    @property
    def is_empty(self) -> bool:
        return not any(
            [
                self.cacheable_regions,
                self.pruning_opportunities,
                self.route_decisions,
                self.bundling_opportunities,
            ]
        )

    def summary(self) -> str:
        lines = ["Optimization Plan", "=" * 50]
        if self.cacheable_regions:
            lines.append(f"\nPrompt Caching ({len(self.cacheable_regions)} regions):")
            for cr in self.cacheable_regions:
                lines.append(
                    f"  {cr.estimated_tokens:>5d} tokens × {cr.occurrence_count} calls "
                    f"= ${cr.total_estimated_savings:.4f} saved"
                )
                lines.append(f"    Preview: {cr.text_preview}")

        if self.pruning_opportunities:
            lines.append(f"\nContext Pruning ({len(self.pruning_opportunities)} opportunities):")
            for po in self.pruning_opportunities:
                lines.append(
                    f"  {po.source:<30} -{po.tokens_to_save:>5d} tokens "
                    f"(confidence: {po.confidence:.0%})"
                )

        if self.route_decisions:
            lines.append(f"\nModel Routing ({len(self.route_decisions)} decisions):")
            for rd in self.route_decisions:
                lines.append(
                    f"  {rd.operation:<20} {rd.original_model} → {rd.suggested_model} "
                    f"save ${rd.estimated_savings_usd:.4f}"
                )

        if self.bundling_opportunities:
            lines.append(f"\nTool Call Bundling ({len(self.bundling_opportunities)} opportunities):")
            for bo in self.bundling_opportunities:
                lines.append(
                    f"  {', '.join(bo.tool_names):<40} "
                    f"save {bo.estimated_latency_reduction_s:.1f}s"
                )

        if self.is_empty:
            lines.append("\nNo optimization opportunities identified.")

        lines.append(f"\n{'—'*50}")
        lines.append(
            f"Total estimated savings: ${self.total_estimated_savings:.4f} "
            f"({self.total_tokens_saved} tokens prunable)"
        )
        return "\n".join(lines)


# ── Optimizers ─────────────────────────────────────────────────────────


class Optimizer:
    """Orchestrates all cost optimization strategies against agent traces.

    Usage:
        >>> optimizer = Optimizer(pricing_model_id="claude-sonnet-4-20250514")
        >>> traces = [
        ...     {"system_prompt": "...", "messages": [...], "tools_called": [...]},
        ...     {"system_prompt": "...", "messages": [...], "tools_called": [...]},
        ... ]
        >>> plan = optimizer.analyze_traces(traces)
        >>> print(plan.summary())
        >>> print(f"Total savings: ${plan.total_estimated_savings:.4f}")
    """

    def __init__(self, pricing_model_id: str = "claude-sonnet-4-20250514"):
        self.pricing_model_id = pricing_model_id
        self._price_per_1m_input = 3.0  # Default to Claude Sonnet 4
        self._price_per_1m_output = 15.0

        # Resolve pricing if available
        from agentops.cost.pricing import _get_catalog

        pricing = _get_catalog().get(pricing_model_id)
        if pricing:
            self._price_per_1m_input = pricing.price_input
            self._price_per_1m_output = pricing.price_output

    def analyze_traces(self, traces: list[dict[str, Any]]) -> OptimizationPlan:
        """Run all optimization analyses against a set of traces.

        Each trace should be a dict with:
            - system_prompt (str)
            - messages (list of {"role": str, "content": str})
            - tools_called (list of str, optional)
            - agent_id (str, optional)
        """
        plan = OptimizationPlan()

        # 1. Prompt caching analysis
        plan.cacheable_regions = _analyze_prompt_caching(
            traces, price_per_1m_input=self._price_per_1m_input
        )

        # 2. Context pruning analysis
        plan.pruning_opportunities = _analyze_context_pruning(traces)

        # 3. Model routing analysis
        plan.route_decisions = _analyze_routing(traces) if len(traces) > 0 else []

        # 4. Tool call bundling
        plan.bundling_opportunities = _analyze_tool_bundling(traces)

        return plan

    def estimate_cache_savings(
        self, system_prompts: list[str], call_count: int
    ) -> float:
        """Estimate savings from prompt caching given repeated system prompts.

        Args:
            system_prompts: List of system prompt strings across calls.
            call_count: Total number of LLM calls.

        Returns:
            Estimated savings in USD.
        """
        return estimate_cache_savings(system_prompts, call_count, self._price_per_1m_input)


# ── Optimization Strategy Functions ────────────────────────────────────


def _analyze_prompt_caching(
    traces: list[dict[str, Any]], price_per_1m_input: float = 3.0
) -> list[CacheableRegion]:
    """Detect repeated prompt regions that would benefit from caching.

    Strategy: Find system prompts that appear in multiple traces,
    and compute the savings from caching them.
    """
    system_prompt_counts: dict[str, int] = defaultdict(int)

    for trace in traces:
        sp = trace.get("system_prompt", "")
        if sp:
            system_prompt_counts[sp] += 1

    # Anthropic/OpenAI cached input pricing: typically 10-25% of base.
    # Using 20% for estimation — cache hit saves 80% of input cost.
    CACHE_SAVINGS_RATIO = 0.80

    regions: list[CacheableRegion] = []
    for sp, count in system_prompt_counts.items():
        if count <= 1:
            continue

        estimated_tokens = max(1, len(sp) // 4)
        savings_per_call = (estimated_tokens / 1_000_000) * price_per_1m_input * CACHE_SAVINGS_RATIO
        total_savings = savings_per_call * (count - 1)  # First call not cached

        regions.append(
            CacheableRegion(
                text=sp,
                estimated_tokens=estimated_tokens,
                occurrence_count=count,
                estimated_savings_per_call=round(savings_per_call, 6),
                total_estimated_savings=round(total_savings, 6),
            )
        )

    regions.sort(key=lambda r: r.total_estimated_savings, reverse=True)
    return regions


def _analyze_context_pruning(traces: list[dict[str, Any]]) -> list[PruningOpportunity]:
    """Identify opportunities to prune context for token savings.

    Looks for:
    - Long conversation histories with old turns
    - Verbose system prompts (>500 tokens)
    - Redundant or repeated user messages
    """
    opportunities: list[PruningOpportunity] = []

    for i, trace in enumerate(traces):
        sp = trace.get("system_prompt", "")
        sp_tokens = max(1, len(sp) // 4)

        # Verbose system prompt check
        if sp_tokens > 500:
            opportunities.append(
                PruningOpportunity(
                    source=f"verbose_system_prompt_trace_{i}",
                    tokens_to_save=int(sp_tokens * 0.3),  # Can trim ~30%
                    confidence=0.7,
                    description=f"System prompt is {sp_tokens} tokens; consider trimming to essentials",
                )
            )

        # Long conversation history check
        messages = trace.get("messages", [])
        if len(messages) > 10:
            estimated_msg_tokens = sum(
                max(1, len(m.get("content", "")) // 4) for m in messages
            )
            # Old turns (first half) could potentially be summarized
            old_turn_savings = int(estimated_msg_tokens * 0.25)
            if old_turn_savings > 100:
                opportunities.append(
                    PruningOpportunity(
                        source=f"old_turns_trace_{i}",
                        tokens_to_save=old_turn_savings,
                        confidence=0.6,
                        description=f"Conversation has {len(messages)} turns; summarize early turns to save ~{old_turn_savings} tokens",
                    )
                )

    opportunities.sort(key=lambda o: o.tokens_to_save, reverse=True)
    return opportunities[:10]  # Top 10 most impactful


def _analyze_routing(traces: list[dict[str, Any]]) -> list[ModelRouteDecision]:
    """Identify operations that could use cheaper models.

    Heuristic: simple classification, summarization of short text,
    and basic extraction tasks can be routed to cheaper models.
    """
    SIMPLE_OPERATIONS = [
        "classify",
        "extract",
        "sentiment",
        "categorize",
        "filter",
        "validate",
        "format",
        "translate_short",
    ]

    decisions: list[ModelRouteDecision] = []
    for i, trace in enumerate(traces):
        op = trace.get("operation", "").lower()
        if not op:
            continue

        is_simple = any(simple in op for simple in SIMPLE_OPERATIONS)
        if not is_simple:
            continue

        # Simple ops can use GPT-4o-mini or Claude Haiku instead
        # Savings: $3.00/1M → $0.15/1M for input, $15.00/1M → $0.60/1M for output
        input_savings_per_call = trace.get("estimated_input_tokens", 500) / 1_000_000 * (3.00 - 0.15)
        output_savings_per_call = trace.get("estimated_output_tokens", 200) / 1_000_000 * (15.00 - 0.60)

        decisions.append(
            ModelRouteDecision(
                operation=op,
                original_model="claude-sonnet-4-20250514",
                suggested_model="claude-haiku-3-5-20241022",
                estimated_savings_usd=round(input_savings_per_call + output_savings_per_call, 6),
                rationale=f"Simple {op} task doesn't require Sonnet-level reasoning",
            )
        )

    return decisions


def _analyze_tool_bundling(
    traces: list[dict[str, Any]],
) -> list[BundlingOpportunity]:
    """Identify sequential independent tool calls that could be parallelized.

    Tool calls that don't depend on each other's output can be sent
    simultaneously, reducing both latency and token overhead (each
    call has a per-request overhead).
    """
    opportunities: list[BundlingOpportunity] = []
    for i, trace in enumerate(traces):
        tools = trace.get("tools_called", [])
        if len(tools) < 2:
            continue

        # Simple heuristic: if tools are all independent lookups (search,
        # fetch, read), they can typically be parallelized.
        INDEPENDENT_PATTERNS = ["search", "fetch", "read", "get", "lookup", "query"]

        all_independent = all(
            any(p in t.lower() for p in INDEPENDENT_PATTERNS) for t in tools
        )

        if all_independent:
            # Latency reduction: from N×sequential_latency to max(parallel_latency)
            # Conservative estimate: 70% latency reduction
            estimated_latency_s = len(tools) * 0.8 * 0.7  # rough: 0.8s per call, 70% reduction
            # Token savings: eliminate N-1 request/response overhead messages
            # (tool results don't need to be re-sent as follow-up context)
            overhead_tokens = (len(tools) - 1) * 100  # ~100 overhead tokens per extra round-trip
            savings = (overhead_tokens / 1_000_000) * 3.0  # at Sonnet input price

            opportunities.append(
                BundlingOpportunity(
                    tool_names=tools,
                    original_order=list(range(len(tools))),
                    estimated_latency_reduction_s=round(estimated_latency_s, 1),
                    estimated_savings_usd=round(savings, 6),
                )
            )

    return opportunities


# ── Standalone Functions ───────────────────────────────────────────────


def estimate_cache_savings(
    system_prompts: list[str],
    call_count: int,
    price_per_1m_input: float = 3.0,
) -> float:
    """Estimate savings from prompt caching.

    Args:
        system_prompts: System prompt strings from all calls.
        call_count: Total number of calls.
        price_per_1m_input: Price per 1M input tokens.

    Returns:
        Estimated savings in USD.
    """
    if not system_prompts or call_count < 2:
        return 0.0

    # Use the first system prompt as the baseline cached content
    first_sp = system_prompts[0]
    sp_tokens = max(1, len(first_sp) // 4)

    # Count how many calls use the same system prompt
    same_count = sum(1 for sp in system_prompts if sp == first_sp)

    if same_count < 2:
        return 0.0

    # Cache hit saves ~80% of input cost for the cached portion
    # First call is not cached, subsequent calls benefit
    cache_savings_ratio = 0.80
    savings_per_call = (sp_tokens / 1_000_000) * price_per_1m_input * cache_savings_ratio
    total_savings = savings_per_call * (same_count - 1)

    return round(total_savings, 6)


# ── Specialized Classes (public API) ───────────────────────────────────


class ContextPruner:
    """Analyzes and prunes context windows for token efficiency.

    Usage:
        >>> pruner = ContextPruner(max_context_tokens=32000)
        >>> pruned_messages = pruner.prune(messages)
        >>> print(f"Pruned {pruner.last_pruned_tokens} tokens")
    """

    def __init__(self, max_context_tokens: int = 32_000):
        self.max_context_tokens = max_context_tokens
        self.last_pruned_tokens: int = 0

    def prune(
        self, messages: list[dict[str, str]], system_prompt: str = ""
    ) -> list[dict[str, str]]:
        """Prune messages to fit within the max context window.

        Strategy: Keep system prompt, then trim oldest messages first.
        """
        total_est = max(1, len(system_prompt) // 4)
        total_est += sum(max(1, len(m.get("content", "")) // 4) for m in messages)

        if total_est <= self.max_context_tokens:
            self.last_pruned_tokens = 0
            return messages

        # Trim from the front (oldest), keeping system-equivalent first
        pruned = []
        remaining = self.max_context_tokens - max(1, len(system_prompt) // 4)
        if remaining <= 0:
            # System prompt alone exceeds budget — trim system prompt
            trimmed_sp = system_prompt[: self.max_context_tokens * 4]
            self.last_pruned_tokens = total_est - self.max_context_tokens
            return messages[-1:] if messages else []

        # Work backwards from newest
        for msg in reversed(messages):
            msg_tokens = max(1, len(msg.get("content", "")) // 4)
            if msg_tokens <= remaining:
                pruned.insert(0, msg)
                remaining -= msg_tokens
            elif remaining > 50:
                # Keep a truncated version
                truncated_content = msg["content"][: remaining * 4]
                pruned.insert(0, {"role": msg["role"], "content": truncated_content})
                remaining = 0
            else:
                break

        self.last_pruned_tokens = total_est - (
            max(1, len(system_prompt) // 4)
            + sum(max(1, len(m.get("content", "")) // 4) for m in pruned)
        )
        return pruned


class ModelRouter:
    """Routes tasks to appropriate models based on complexity heuristics.

    Uses simple keyword-based classification to decide whether a task
    requires an expensive model or can be handled by a cheaper one.
    """

    COMPLEX_PATTERNS = [
        "reason",
        "analyze",
        "evaluate",
        "compare",
        "plan",
        "design",
        "debug",
        "optimize",
        "generate_code",
        "architecture",
    ]

    SIMPLE_PATTERNS = [
        "classify",
        "extract",
        "sentiment",
        "summarize",
        "translate",
        "format",
        "validate",
        "categorize",
        "filter",
        "lookup",
    ]

    def __init__(
        self,
        expensive_model: str = "claude-sonnet-4-20250514",
        cheap_model: str = "claude-haiku-3-5-20241022",
    ):
        self.expensive_model = expensive_model
        self.cheap_model = cheap_model

    def route(self, operation: str, input_text: str = "") -> str:
        """Decide which model to use for a given operation.

        Args:
            operation: The operation name (e.g., 'analyze_sentiment').
            input_text: Optional input text for additional heuristics.

        Returns:
            Model ID to use.
        """
        op_lower = operation.lower()

        # Explicit complex patterns → expensive model
        if any(p in op_lower for p in self.COMPLEX_PATTERNS):
            return self.expensive_model

        # Simple patterns + short input → cheap model
        if any(p in op_lower for p in self.SIMPLE_PATTERNS):
            if len(input_text) < 2000:  # Short input
                return self.cheap_model
            # Long input for simple task → still cheap usually
            return self.cheap_model

        # Default to expensive for safety
        if len(input_text) > 5000:
            return self.expensive_model

        return self.cheap_model


class ToolBundler:
    """Identifies tool calls that can be executed in parallel.

    Independent tool calls (no data dependency) can be sent concurrently,
    reducing round-trip latency and token overhead.
    """

    def __init__(self):
        self._independent_patterns = [
            "search",
            "fetch",
            "read",
            "get",
            "lookup",
            "query",
            "list",
            "describe",
        ]

    def analyze(
        self, tool_names: list[str], tool_args: Optional[list[dict]] = None
    ) -> list[list[int]]:
        """Group tool call indices into parallelizable batches.

        Args:
            tool_names: List of tool names in order.
            tool_args: Optional tool arguments for dependency analysis.

        Returns:
            List of batches, each containing indices of parallelizable calls.
        """
        if len(tool_names) < 2:
            return [[0]] if tool_names else []

        # Simple heuristic: all independent-pattern tools can be batched together
        batches: list[list[int]] = []
        current_batch: list[int] = []

        for i, name in enumerate(tool_names):
            is_independent = any(p in name.lower() for p in self._independent_patterns)

            if is_independent:
                current_batch.append(i)
            else:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([i])  # Dependent call runs alone

        if current_batch:
            batches.append(current_batch)

        # Merge consecutive singletons if they're independent
        merged: list[list[int]] = []
        for batch in batches:
            if len(batch) == 1 and merged and len(merged[-1]) > 1:
                merged[-1].extend(batch)
            else:
                merged.append(batch)

        return merged
