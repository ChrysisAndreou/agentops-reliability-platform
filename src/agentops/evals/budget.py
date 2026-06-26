"""
Cost and latency budget gates for agent execution.

Enables setting per-run and aggregate budgets that gate agent behavior,
preventing runaway costs or excessive latency in production deployments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostBudget:
    """Cost budget configuration for a single agent run."""

    max_total_cost_usd: float = 0.50
    max_per_step_cost_usd: float = 0.10
    cost_per_1k_input_tokens: float = 0.0025  # GPT-4o-ish
    cost_per_1k_output_tokens: float = 0.010

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token counts."""
        return (
            (input_tokens / 1000) * self.cost_per_1k_input_tokens
            + (output_tokens / 1000) * self.cost_per_1k_output_tokens
        )


@dataclass
class LatencyBudget:
    """Latency budget configuration."""

    max_total_latency_ms: float = 120_000  # 2 minutes
    max_per_step_latency_ms: float = 30_000  # 30 seconds
    warn_at_latency_ms: float = 60_000  # Warn at 1 minute


@dataclass
class BudgetState:
    """Runtime budget tracking for an agent execution."""

    cost_budget: CostBudget = field(default_factory=CostBudget)
    latency_budget: LatencyBudget = field(default_factory=LatencyBudget)

    # Running totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    start_time_ms: float = 0.0

    # Per-step tracking
    step_count: int = 0
    step_latencies_ms: list[float] = field(default_factory=list)

    # Status
    cost_exceeded: bool = False
    latency_exceeded: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        if self.start_time_ms == 0:
            return 0
        return (time.time() * 1000) - self.start_time_ms

    @property
    def budget_remaining_pct(self) -> float:
        """Percentage of the most constrained budget remaining."""
        cost_pct = (
            1.0 - self.total_estimated_cost_usd / self.cost_budget.max_total_cost_usd
            if self.cost_budget.max_total_cost_usd > 0
            else 1.0
        )
        latency_pct = (
            1.0 - self.elapsed_ms / self.latency_budget.max_total_latency_ms
            if self.latency_budget.max_total_latency_ms > 0
            else 1.0
        )
        return max(0.0, min(cost_pct, latency_pct)) * 100

    def start(self) -> None:
        """Record the start time."""
        self.start_time_ms = time.time() * 1000

    def record_step(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
    ) -> None:
        """Record a step's resource usage."""
        self.step_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.step_latencies_ms.append(latency_ms)

        # Update cost
        step_cost = self.cost_budget.estimate_cost(input_tokens, output_tokens)
        self.total_estimated_cost_usd += step_cost

        # Check per-step cost
        if step_cost > self.cost_budget.max_per_step_cost_usd:
            self.warnings.append(
                f"Step {self.step_count} cost ${step_cost:.4f} exceeds "
                f"per-step budget ${self.cost_budget.max_per_step_cost_usd:.4f}"
            )

        # Check per-step latency
        if latency_ms > self.latency_budget.max_per_step_latency_ms:
            self.warnings.append(
                f"Step {self.step_count} latency {latency_ms:.0f}ms exceeds "
                f"per-step budget {self.latency_budget.max_per_step_latency_ms:.0f}ms"
            )

    def check_budgets(self) -> dict[str, Any]:
        """Check all budgets and return status.

        Returns:
            dict with 'ok' (bool), 'cost_exceeded', 'latency_exceeded',
            'warnings', and 'budget_remaining_pct'.
        """
        # Check total cost
        if self.total_estimated_cost_usd > self.cost_budget.max_total_cost_usd:
            self.cost_exceeded = True
            self.warnings.append(
                f"Total cost ${self.total_estimated_cost_usd:.4f} exceeds "
                f"budget ${self.cost_budget.max_total_cost_usd:.4f}"
            )

        # Check total latency
        if self.elapsed_ms > self.latency_budget.max_total_latency_ms:
            self.latency_exceeded = True
            self.warnings.append(
                f"Total latency {self.elapsed_ms:.0f}ms exceeds "
                f"budget {self.latency_budget.max_total_latency_ms:.0f}ms"
            )

        # Check warning threshold
        if self.elapsed_ms > self.latency_budget.warn_at_latency_ms:
            self.warnings.append(
                f"Latency warning: {self.elapsed_ms:.0f}ms elapsed "
                f"(threshold: {self.latency_budget.warn_at_latency_ms:.0f}ms)"
            )

        return {
            "ok": not (self.cost_exceeded or self.latency_exceeded),
            "cost_exceeded": self.cost_exceeded,
            "latency_exceeded": self.latency_exceeded,
            "warnings": self.warnings[-5:],  # Last 5 warnings
            "budget_remaining_pct": round(self.budget_remaining_pct, 1),
            "total_cost_usd": round(self.total_estimated_cost_usd, 4),
            "total_latency_ms": round(self.elapsed_ms, 0),
            "steps": self.step_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


@dataclass
class BudgetGateResult:
    """Result of a budget gate check."""

    allowed: bool
    reason: str = ""
    budget_state: BudgetState | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"allowed": self.allowed, "reason": self.reason}
        if self.budget_state:
            d["budget"] = self.budget_state.check_budgets()
        return d


class BudgetGate:
    """Gate agent execution on cost and latency budgets.

    Usage:
        gate = BudgetGate(cost_budget=CostBudget(max_total_cost_usd=0.50))
        budget_state = BudgetState()
        budget_state.start()

        for step in workflow:
            budget_state.record_step(input_tokens=500, output_tokens=200, latency_ms=5000)
            result = gate.check(budget_state)
            if not result.allowed:
                raise BudgetExceededError(result.reason)
    """

    def __init__(
        self,
        cost_budget: CostBudget | None = None,
        latency_budget: LatencyBudget | None = None,
    ):
        self.cost_budget = cost_budget or CostBudget()
        self.latency_budget = latency_budget or LatencyBudget()

    def check(self, state: BudgetState) -> BudgetGateResult:
        """Check if execution should continue within budgets."""
        status = state.check_budgets()

        if status["cost_exceeded"]:
            return BudgetGateResult(
                allowed=False,
                reason=f"Cost budget exceeded: ${status['total_cost_usd']:.4f} > "
                       f"${self.cost_budget.max_total_cost_usd:.4f}",
                budget_state=state,
            )

        if status["latency_exceeded"]:
            return BudgetGateResult(
                allowed=False,
                reason=f"Latency budget exceeded: {status['total_latency_ms']:.0f}ms > "
                       f"{self.latency_budget.max_total_latency_ms:.0f}ms",
                budget_state=state,
            )

        return BudgetGateResult(allowed=True, budget_state=state)

    def should_abort(self, state: BudgetState) -> bool:
        """Quick check: should execution abort? (without detailed status)."""
        return (
            state.total_estimated_cost_usd > self.cost_budget.max_total_cost_usd
            or state.elapsed_ms > self.latency_budget.max_total_latency_ms
        )


class BudgetExceededError(Exception):
    """Raised when a budget gate blocks execution."""

    def __init__(self, reason: str, budget_state: BudgetState | None = None):
        super().__init__(reason)
        self.reason = reason
        self.budget_state = budget_state


# ── Pre-built budget configurations ──────────────────────────────────

STRICT_BUDGET = CostBudget(
    max_total_cost_usd=0.25,
    max_per_step_cost_usd=0.05,
)

NORMAL_BUDGET = CostBudget(
    max_total_cost_usd=0.50,
    max_per_step_cost_usd=0.10,
)

GENEROUS_BUDGET = CostBudget(
    max_total_cost_usd=2.00,
    max_per_step_cost_usd=0.50,
)

FAST_LATENCY = LatencyBudget(
    max_total_latency_ms=30_000,
    max_per_step_latency_ms=10_000,
    warn_at_latency_ms=15_000,
)

NORMAL_LATENCY = LatencyBudget(
    max_total_latency_ms=120_000,
    max_per_step_latency_ms=30_000,
    warn_at_latency_ms=60_000,
)

RELAXED_LATENCY = LatencyBudget(
    max_total_latency_ms=300_000,
    max_per_step_latency_ms=60_000,
    warn_at_latency_ms=180_000,
)
