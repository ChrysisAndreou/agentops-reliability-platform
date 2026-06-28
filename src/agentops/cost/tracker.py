"""
Hierarchical cost tracking with budget enforcement for AI agent operations.

Tracks token usage and cost across four levels of granularity:
    call → agent → session → project

Provides soft and hard budget limits with configurable alert thresholds.
Designed for production agent systems where uncontrolled spend is a
real operational risk — a runaway agent loop or misconfigured model
selection can generate thousands of dollars in unexpected costs.

Unlike simple token counters, this module provides:
- Structured cost breakdowns by provider, model, agent, and operation type
- Budget enforcement with soft alerts and hard stops
- Cumulative tracking with real-time budget utilization
- Session/project-level accounting for multi-agent systems
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from agentops.cost.pricing import _get_catalog


class BudgetStatus(Enum):
    """Status of a budget relative to its limits."""

    OK = auto()
    """Within normal operating range."""
    WARNING = auto()
    """Approaching or passed soft limit."""
    EXCEEDED = auto()
    """Hard limit exceeded."""
    EXHAUSTED = auto()
    """Budget fully consumed (100%+)."""


@dataclass
class Budget:
    """A spending budget for cost management.

    Supports both soft (alert-only) and hard (enforcement) limits,
    alert thresholds as a fraction of the limit, and multiple
    time horizons (daily, weekly, monthly, per-session).
    """

    name: str
    limit_usd: float
    soft_limit_usd: float = 0.0
    alert_threshold: float = 0.80  # Alert at 80% utilization by default
    horizon: str = "daily"  # daily, weekly, monthly, per-session

    def __post_init__(self):
        if self.soft_limit_usd == 0.0:
            self.soft_limit_usd = self.limit_usd * self.alert_threshold

    def check(self, spent_usd: float) -> BudgetStatus:
        """Determine budget status given current spend."""
        if spent_usd >= self.limit_usd:
            if spent_usd >= self.limit_usd * 1.5:
                return BudgetStatus.EXHAUSTED
            return BudgetStatus.EXCEEDED
        if spent_usd >= self.soft_limit_usd:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def remaining(self, spent_usd: float) -> float:
        """Remaining budget in USD."""
        return max(0.0, self.limit_usd - spent_usd)

    def utilization(self, spent_usd: float) -> float:
        """Current utilization as a fraction (0.0-1.0+)."""
        if self.limit_usd <= 0:
            return 0.0
        return spent_usd / self.limit_usd


@dataclass
class BudgetAlert:
    """A budget-related alert for monitoring and notification."""

    budget_name: str
    status: BudgetStatus
    spent: float
    limit: float
    utilization: float
    timestamp: float = field(default_factory=time.time)
    message: str = ""


@dataclass
class CostRecord:
    """A single cost event — one LLM call with token usage and pricing."""

    record_id: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    agent_id: str = ""
    session_id: str = ""
    operation: str = ""  # e.g., "classify", "generate", "summarize"
    timestamp: float = field(default_factory=time.time)


@dataclass
class TokenCounter:
    """Cumulative token statistics for a specific scope (model, provider, agent, etc.)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, record: CostRecord) -> None:
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cached_input_tokens += record.cached_input_tokens
        self.cost_usd += record.cost_usd
        self.call_count += 1


@dataclass
class CostBreakdown:
    """Structured cost breakdown by multiple dimensions."""

    by_provider: dict[str, TokenCounter] = field(default_factory=dict)
    by_model: dict[str, TokenCounter] = field(default_factory=dict)
    by_agent: dict[str, TokenCounter] = field(default_factory=dict)
    by_operation: dict[str, TokenCounter] = field(default_factory=dict)
    total: TokenCounter = field(default_factory=TokenCounter)

    def _get_or_create(self, mapping: dict, key: str) -> TokenCounter:
        if key not in mapping:
            mapping[key] = TokenCounter()
        return mapping[key]

    def add(self, record: CostRecord) -> None:
        self._get_or_create(self.by_provider, record.provider).record(record)
        self._get_or_create(self.by_model, record.model_id).record(record)
        if record.agent_id:
            self._get_or_create(self.by_agent, record.agent_id).record(record)
        if record.operation:
            self._get_or_create(self.by_operation, record.operation).record(record)
        self.total.record(record)


class CostTracker:
    """Hierarchical cost tracker for AI agent operations.

    Tracks costs at call/agent/session/project levels with budget
    enforcement. Supports multiple simultaneous budgets with different
    horizons (daily spend cap, per-session limit, project total).

    Usage:
        >>> tracker = CostTracker(
        ...     project_id="customer-support-bot",
        ...     budgets=[Budget("daily", limit_usd=50.00, horizon="daily")],
        ... )
        >>> tracker.new_session(session_id="chat-abc123")
        >>> tracker.record(
        ...     model_id="claude-sonnet-4-20250514",
        ...     input_tokens=1500,
        ...     output_tokens=300,
        ...     agent_id="support-agent",
        ...     operation="answer_question",
        ... )
        >>> status = tracker.check_budgets()
        >>> print(tracker.summary())
    """

    def __init__(
        self,
        project_id: str = "",
        budgets: Optional[list[Budget]] = None,
        alert_callback: Optional[Callable[[BudgetAlert], None]] = None,
    ):
        self.project_id = project_id or str(uuid.uuid4())[:8]
        self.budgets: list[Budget] = budgets or []
        self.alert_callback = alert_callback

        # Current session state
        self.session_id: str = ""
        self._records: list[CostRecord] = []
        self._session_records: list[CostRecord] = []

        # Cumulative accounting
        self._project_breakdown = CostBreakdown()
        self._session_breakdown = CostBreakdown()
        self._budget_alerts: list[BudgetAlert] = []

        # Timestamps
        self.created_at = time.time()
        self.session_started_at: Optional[float] = None

    def new_session(self, session_id: str = "") -> str:
        """Start a new tracking session. Returns the session ID."""
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.session_started_at = time.time()
        self._session_records.clear()
        self._session_breakdown = CostBreakdown()
        return self.session_id

    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        agent_id: str = "",
        operation: str = "",
        session_id: str = "",
    ) -> CostRecord:
        """Record a cost event and return the CostRecord.

        Automatically computes cost from the pricing catalog.
        If the model is not found in the catalog, cost is 0.0.
        """
        pricing = _get_catalog().get(model_id)
        cost = (
            pricing.cost(input_tokens, output_tokens, cached_input_tokens)
            if pricing
            else 0.0
        )
        provider = pricing.provider if pricing else "unknown"

        record = CostRecord(
            record_id=str(uuid.uuid4())[:12],
            model_id=model_id,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cost_usd=cost,
            agent_id=agent_id,
            session_id=session_id or self.session_id,
            operation=operation,
        )

        self._records.append(record)
        self._session_records.append(record)
        self._project_breakdown.add(record)
        self._session_breakdown.add(record)

        # Check budgets after each record
        self._check_budgets()

        return record

    @property
    def total_cost(self) -> float:
        """Total project cost in USD."""
        return self._project_breakdown.total.cost_usd

    @property
    def session_cost(self) -> float:
        """Current session cost in USD."""
        return self._session_breakdown.total.cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens across all calls."""
        return self._project_breakdown.total.input_tokens + self._project_breakdown.total.output_tokens

    @property
    def call_count(self) -> int:
        """Total number of LLM calls tracked."""
        return self._project_breakdown.total.call_count

    def _check_budgets(self) -> list[BudgetAlert]:
        """Check all budgets against current spend and return any alerts."""
        alerts: list[BudgetAlert] = []
        for budget in self.budgets:
            spent = self._budget_spend_for(budget)
            status = budget.check(spent)
            if status != BudgetStatus.OK:
                alert = BudgetAlert(
                    budget_name=budget.name,
                    status=status,
                    spent=spent,
                    limit=budget.limit_usd,
                    utilization=budget.utilization(spent),
                    message=f"[{status.name}] {budget.name}: ${spent:.2f} / ${budget.limit_usd:.2f} ({budget.utilization(spent):.0%})",
                )
                alerts.append(alert)
                self._budget_alerts.append(alert)
                if self.alert_callback:
                    self.alert_callback(alert)
        return alerts

    def _budget_spend_for(self, budget: Budget) -> float:
        """Compute the relevant spend for a budget's horizon."""
        if budget.horizon == "per-session":
            return self.session_cost
        # For daily/weekly/monthly, we use project total (simplified).
        # In production, you'd filter records by date range.
        return self.total_cost

    def check_budgets(self) -> list[BudgetAlert]:
        """Check all budgets and return any new alerts."""
        return self._check_budgets()

    def budget_exceeded(self) -> bool:
        """Return True if any hard budget has been exceeded."""
        for budget in self.budgets:
            spent = self._budget_spend_for(budget)
            if budget.check(spent) in (BudgetStatus.EXCEEDED, BudgetStatus.EXHAUSTED):
                return True
        return False

    def get_breakdown(self, scope: str = "project") -> CostBreakdown:
        """Get the cost breakdown for a given scope ('project' or 'session')."""
        if scope == "session":
            return self._session_breakdown
        return self._project_breakdown

    def get_alerts(self, since: Optional[float] = None) -> list[BudgetAlert]:
        """Get budget alerts, optionally filtered by timestamp."""
        if since is None:
            return list(self._budget_alerts)
        return [a for a in self._budget_alerts if a.timestamp >= since]

    def summary(self) -> str:
        """Generate a human-readable cost summary."""
        bd = self._project_breakdown
        lines = [
            f"Cost Summary — {self.project_id}",
            f"{'='*50}",
            f"Total calls:       {bd.total.call_count:>8d}",
            f"Total tokens:      {bd.total.total_tokens:>8d}",
            f"  Input:           {bd.total.input_tokens:>8d}",
            f"  Output:          {bd.total.output_tokens:>8d}",
            f"  Cached input:    {bd.total.cached_input_tokens:>8d}",
            f"Total cost:        ${bd.total.cost_usd:>8.4f}",
            f"",
            f"Session cost:      ${self.session_cost:>8.4f}",
            f"Session calls:     {self._session_breakdown.total.call_count:>8d}",
            f"",
            f"By Provider:",
        ]
        for provider, counter in sorted(bd.by_provider.items()):
            lines.append(
                f"  {provider:<20} ${counter.cost_usd:>8.4f}  ({counter.call_count:>4d} calls)"
            )

        if self.budgets:
            lines.append(f"")
            lines.append(f"Budgets:")
            for budget in self.budgets:
                spent = self._budget_spend_for(budget)
                status = budget.check(spent)
                lines.append(
                    f"  {budget.name:<20} ${spent:>8.4f} / ${budget.limit_usd:>8.2f}  [{status.name}]"
                )

        return "\n".join(lines)

    def reset_session(self) -> None:
        """Reset session-level accounting (project totals preserved)."""
        self.session_id = ""
        self._session_records.clear()
        self._session_breakdown = CostBreakdown()
        self.session_started_at = None
