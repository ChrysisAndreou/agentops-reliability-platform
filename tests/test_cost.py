"""
Tests for AgentOps v0.27: Cost Economics & Token Optimization.

Covers: pricing catalog, cost tracking, budget enforcement, optimization
strategies, and efficiency evaluation.
"""

import math
import pytest
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
    PruningOpportunity,
    ToolBundler,
    estimate_cache_savings,
)
from agentops.cost.eval import (
    BudgetComplianceReport,
    CostEfficiencyMetrics,
    CostEvalReport,
    CostRegression,
    CostVsQuality,
    detect_cost_regressions,
    evaluate_cost_efficiency,
    evaluate_cost_vs_quality,
    format_cost_report,
)


# ═══════════════════════════════════════════════════════════════════════
# PRICING TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestModelPricing:
    """Tests for ModelPricing cost computation."""

    def test_cost_zero_tokens(self):
        """Cost with zero tokens should be zero."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=8.0,
        )
        assert p.cost(0, 0) == 0.0

    def test_cost_input_only(self):
        """Cost computation for input tokens only."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=8.0,
        )
        # 1000 tokens at $2/1M = $0.002
        assert p.cost(1000, 0) == pytest.approx(0.002, abs=1e-6)

    def test_cost_output_only(self):
        """Cost computation for output tokens only."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=8.0,
        )
        assert p.cost(0, 500) == pytest.approx(0.004, abs=1e-6)

    def test_cost_combined(self):
        """Cost with both input and output tokens."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=10.0,
        )
        cost = p.cost(1500, 300)
        expected = (1500 / 1_000_000) * 2.0 + (300 / 1_000_000) * 10.0
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_cost_cached_input(self):
        """Cost with cached input tokens (prompt cache hit)."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=3.0, price_output=15.0,
            price_cached_input=0.75, supports_caching=True,
        )
        # 0 non-cached input, 0 output, 10000 cached input
        cost = p.cost(input_tokens=0, output_tokens=0, cached_input_tokens=10000)
        expected = (10000 / 1_000_000) * 0.75
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_cost_mixed_cache(self):
        """Mixed cached and non-cached input."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=3.0, price_output=15.0,
            price_cached_input=0.75, supports_caching=True,
        )
        cost = p.cost(input_tokens=2000, output_tokens=500, cached_input_tokens=8000)
        expected = (2000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0 + (8000 / 1_000_000) * 0.75
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_batch_discount(self):
        """Batch pricing applies 50% discount."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=10.0,
            batch_discount=0.50,
        )
        cost_batch = p.cost(1000, 500, batch=True)
        cost_normal = p.cost(1000, 500, batch=False)
        assert cost_batch == pytest.approx(cost_normal * 0.5, abs=1e-6)

    def test_batch_with_cached(self):
        """Batch pricing with cached input."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=10.0,
            price_cached_input=0.50, batch_discount=0.50, supports_caching=True,
        )
        cost = p.cost(input_tokens=1000, cached_input_tokens=4000, batch=True)
        # Both regular and cached input get batch discount
        expected = (1000 / 1_000_000) * 1.0 + (4000 / 1_000_000) * 0.25
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_cost_million_tokens(self):
        """Cost at exactly 1M tokens."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.50, price_output=10.00,
        )
        cost = p.cost(1_000_000, 0)
        assert cost == pytest.approx(2.50, abs=1e-6)

    def test_zero_price_model(self):
        """Models with zero pricing (free tier)."""
        p = ModelPricing(
            provider="Test", model_id="free-model",
            display_name="Free", price_input=0.0, price_output=0.0,
        )
        assert p.cost(1_000_000, 500_000) == 0.0

    def test_price_properties(self):
        """Batch price properties are correctly derived."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=4.0, price_output=16.0,
            batch_discount=0.50,
        )
        assert p.price_batch_input == 2.0
        assert p.price_batch_output == 8.0

    def test_frozen_dataclass(self):
        """ModelPricing is immutable."""
        p = ModelPricing(
            provider="Test", model_id="test-model",
            display_name="Test", price_input=2.0, price_output=8.0,
        )
        with pytest.raises(Exception):
            p.price_input = 5.0


class TestPricingTier:
    """Tests for PricingTier rate limits."""

    def test_tps_derivation(self):
        """TPM → TPS derivation."""
        tier = PricingTier(tpm=450_000, rpm=500)
        assert tier.tps == pytest.approx(7500.0, abs=0.1)

    def test_low_tps(self):
        """Low TPM produces low TPS."""
        tier = PricingTier(tpm=60, rpm=10)
        assert tier.tps == pytest.approx(1.0, abs=0.01)


class TestProviderCatalog:
    """Tests for the provider catalog system."""

    def test_default_catalog_size(self):
        """Default catalog contains all providers."""
        catalog = ProviderCatalog.default()
        assert len(catalog) >= 20  # At least 20 models registered

    def test_get_known_model(self):
        """Lookup of known model returns pricing."""
        catalog = ProviderCatalog.default()
        p = catalog.get("gpt-4o")
        assert p is not None
        assert p.provider == "OpenAI"
        assert p.display_name == "GPT-4o"
        assert p.price_input == 2.50
        assert p.price_output == 10.00

    def test_get_unknown_model(self):
        """Lookup of unknown model returns None."""
        catalog = ProviderCatalog.default()
        assert catalog.get("nonexistent-model-v999") is None

    def test_list_providers(self):
        """Providers list is deduplicated and sorted."""
        catalog = ProviderCatalog.default()
        providers = catalog.list_providers()
        assert "OpenAI" in providers
        assert "Anthropic" in providers
        assert "Cohere" in providers
        assert "Mistral" in providers
        assert "Google" in providers
        assert "Groq" in providers
        assert providers == sorted(providers)

    def test_list_models_by_provider(self):
        """Filtering models by provider."""
        catalog = ProviderCatalog.default()
        openai_models = catalog.list_models("OpenAI")
        assert len(openai_models) >= 4
        for m in openai_models:
            assert m.provider == "OpenAI"

    def test_list_models_sorted(self):
        """Models sorted by provider then price."""
        catalog = ProviderCatalog.default()
        models = catalog.list_models()
        for i in range(1, len(models)):
            prev = models[i - 1]
            curr = models[i]
            assert (prev.provider, prev.price_input) <= (curr.provider, curr.price_input)

    def test_all_models_have_required_fields(self):
        """All models have provider, model_id, display_name."""
        catalog = ProviderCatalog.default()
        for model in catalog.list_models():
            assert model.provider
            assert model.model_id
            assert model.display_name
            assert model.price_input >= 0
            assert model.price_output >= 0

    def test_anthropic_models_have_caching(self):
        """Anthropic models support prompt caching."""
        catalog = ProviderCatalog.default()
        for m in catalog.list_models("Anthropic"):
            assert m.supports_caching
            assert m.price_cached_input > 0

    def test_openai_batch_discount(self):
        """OpenAI models have batch discount."""
        catalog = ProviderCatalog.default()
        for m in catalog.list_models("OpenAI"):
            assert m.batch_discount > 0

    def test_cache_models_support_caching(self):
        """Models with price_cached_input have supports_caching=True."""
        catalog = ProviderCatalog.default()
        for m in catalog.list_models():
            if m.price_cached_input > 0:
                assert m.supports_caching


class TestGetCost:
    """Tests for the get_cost convenience function."""

    def test_get_cost_gpt4o(self):
        """Get cost for GPT-4o."""
        cost = get_cost("gpt-4o", input_tokens=1000, output_tokens=200)
        expected = (1000 / 1_000_000) * 2.50 + (200 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_get_cost_unknown_model(self):
        """Unknown model returns 0.0."""
        assert get_cost("nonexistent", input_tokens=1000) == 0.0

    def test_get_cost_with_cache(self):
        """Cost with cached tokens."""
        cost = get_cost(
            "claude-sonnet-4-20250514",
            input_tokens=2000, output_tokens=500, cached_input_tokens=3000,
        )
        pricing = get_pricing("claude-sonnet-4-20250514")
        expected = pricing.cost(2000, 500, 3000)
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_get_cost_batch(self):
        """Batch mode pricing."""
        cost = get_cost("gpt-4o", input_tokens=1000, output_tokens=500, batch=True)
        pricing = get_pricing("gpt-4o")
        expected = pricing.cost(1000, 500, batch=True)
        assert cost == pytest.approx(expected, abs=1e-6)


class TestGetPricing:
    """Tests for get_pricing convenience function."""

    def test_known_models(self):
        """All known models return pricing."""
        for model_id in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514",
                         "command-r-plus", "mistral-large-2"]:
            assert get_pricing(model_id) is not None, f"Missing: {model_id}"

    def test_unknown(self):
        """Unknown model returns None."""
        assert get_pricing("made-up-model") is None


class TestListModels:
    """Tests for list_models."""

    def test_all_models(self):
        """Default returns all models."""
        models = list_models()
        assert len(models) >= 20

    def test_filter_by_provider(self):
        """Provider filter works."""
        mistral_models = list_models("Mistral")
        assert all(m.provider == "Mistral" for m in mistral_models)
        assert len(mistral_models) >= 3


class TestEstimateTokens:
    """Tests for token estimation heuristic."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_simple_text(self):
        """~4 chars per token."""
        text = "Hello, world! This is a test."  # 30 chars
        tokens = estimate_tokens(text)
        assert tokens == 8  # 30/4 = 7.5 → ceil = 8

    def test_long_text(self):
        """Longer text estimation."""
        text = "a" * 1000
        tokens = estimate_tokens(text)
        assert tokens == 250

    def test_always_at_least_1(self):
        """Short text gives at least 1 token."""
        assert estimate_tokens("ab") == 1


# ═══════════════════════════════════════════════════════════════════════
# TRACKER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestBudget:
    """Tests for Budget and budget status checking."""

    def test_budget_ok(self):
        budget = Budget("daily", limit_usd=10.0, soft_limit_usd=8.0)
        assert budget.check(5.0) == BudgetStatus.OK

    def test_budget_warning(self):
        budget = Budget("daily", limit_usd=10.0, soft_limit_usd=8.0)
        assert budget.check(9.0) == BudgetStatus.WARNING

    def test_budget_exceeded(self):
        budget = Budget("daily", limit_usd=10.0, soft_limit_usd=8.0)
        assert budget.check(12.0) == BudgetStatus.EXCEEDED

    def test_budget_exhausted(self):
        budget = Budget("daily", limit_usd=10.0, soft_limit_usd=8.0)
        assert budget.check(20.0) == BudgetStatus.EXHAUSTED

    def test_budget_remaining(self):
        budget = Budget("daily", limit_usd=10.0)
        assert budget.remaining(3.5) == 6.5
        assert budget.remaining(15.0) == 0.0

    def test_budget_utilization(self):
        budget = Budget("daily", limit_usd=10.0)
        assert budget.utilization(5.0) == 0.5
        assert budget.utilization(15.0) == 1.5

    def test_budget_soft_limit_default(self):
        """Default soft limit is alert_threshold × limit."""
        budget = Budget("daily", limit_usd=100.0, alert_threshold=0.75)
        assert budget.soft_limit_usd == 75.0

    def test_budget_explicit_soft_limit(self):
        """Explicit soft limit overrides threshold."""
        budget = Budget("daily", limit_usd=100.0, soft_limit_usd=60.0, alert_threshold=0.8)
        assert budget.soft_limit_usd == 60.0

    def test_budget_zero_limit(self):
        """Zero limit budget always exceeded."""
        budget = Budget("zero", limit_usd=0.0)
        assert budget.check(0.01) == BudgetStatus.EXHAUSTED

    def test_budget_utilization_zero_limit(self):
        """Zero limit utilization returns 0."""
        budget = Budget("zero", limit_usd=0.0)
        assert budget.utilization(5.0) == 0.0


class TestTokenCounter:
    """Tests for TokenCounter cumulative tracking."""

    def test_empty_counter(self):
        counter = TokenCounter()
        assert counter.input_tokens == 0
        assert counter.output_tokens == 0
        assert counter.total_tokens == 0
        assert counter.cost_usd == 0.0
        assert counter.call_count == 0

    def test_record_accumulation(self):
        counter = TokenCounter()
        r1 = CostRecord("r1", "m1", "p1", 100, 50, cost_usd=0.01)
        r2 = CostRecord("r2", "m1", "p1", 200, 75, cost_usd=0.02)
        counter.record(r1)
        counter.record(r2)
        assert counter.input_tokens == 300
        assert counter.output_tokens == 125
        assert counter.total_tokens == 425
        assert counter.cost_usd == 0.03
        assert counter.call_count == 2

    def test_record_cached(self):
        counter = TokenCounter()
        r = CostRecord("r1", "m1", "p1", 100, 50, cached_input_tokens=200, cost_usd=0.005)
        counter.record(r)
        assert counter.cached_input_tokens == 200


class TestCostBreakdown:
    """Tests for CostBreakdown multi-dimensional tracking."""

    def test_add_single_record(self):
        bd = CostBreakdown()
        r = CostRecord("r1", "gpt-4o", "OpenAI", 100, 50, cost_usd=0.001,
                       agent_id="agent1", operation="summarize")
        bd.add(r)
        assert bd.total.call_count == 1
        assert len(bd.by_provider) == 1
        assert len(bd.by_model) == 1
        assert len(bd.by_agent) == 1
        assert len(bd.by_operation) == 1

    def test_add_multiple_providers(self):
        bd = CostBreakdown()
        bd.add(CostRecord("r1", "gpt-4o", "OpenAI", 100, 50, cost_usd=0.001))
        bd.add(CostRecord("r2", "claude-sonnet-4-20250514", "Anthropic", 200, 75, cost_usd=0.002))
        assert len(bd.by_provider) == 2
        assert bd.total.call_count == 2

    def test_same_provider_aggregates(self):
        bd = CostBreakdown()
        bd.add(CostRecord("r1", "gpt-4o", "OpenAI", 100, 50, cost_usd=0.001))
        bd.add(CostRecord("r2", "gpt-4o", "OpenAI", 200, 75, cost_usd=0.002))
        assert len(bd.by_provider) == 1
        assert bd.by_provider["OpenAI"].call_count == 2
        assert bd.by_provider["OpenAI"].input_tokens == 300

    def test_empty_agent_and_operation(self):
        """Records without agent_id or operation don't create those dimensions."""
        bd = CostBreakdown()
        r = CostRecord("r1", "m1", "p1", 100, 50)
        bd.add(r)
        assert len(bd.by_agent) == 0
        assert len(bd.by_operation) == 0


class TestCostTracker:
    """Tests for the main CostTracker."""

    def test_initialization(self):
        tracker = CostTracker(project_id="test-proj")
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_record_single_call(self):
        tracker = CostTracker(project_id="test-proj")
        record = tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        assert record.provider == "OpenAI"
        assert record.cost_usd > 0
        assert tracker.call_count == 1
        assert tracker.total_cost > 0

    def test_record_unknown_model(self):
        """Unknown model still tracks but with zero cost."""
        tracker = CostTracker()
        record = tracker.record("unknown-model", input_tokens=1000)
        assert record.cost_usd == 0.0
        assert record.provider == "unknown"
        assert tracker.call_count == 1

    def test_session_management(self):
        tracker = CostTracker()
        sid = tracker.new_session("session-abc")
        assert sid == "session-abc"

        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        assert tracker.session_cost > 0
        assert tracker.total_cost == tracker.session_cost

    def test_session_reset(self):
        tracker = CostTracker()
        tracker.new_session()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        session1_cost = tracker.session_cost

        tracker.reset_session()
        assert tracker.session_cost == 0.0
        assert tracker.total_cost == session1_cost  # Project total preserved

    def test_multiple_sessions(self):
        tracker = CostTracker()
        tracker.new_session("s1")
        tracker.record("gpt-4o-mini", input_tokens=500)
        tracker.reset_session()
        tracker.new_session("s2")
        tracker.record("gpt-4o-mini", input_tokens=500)
        assert tracker.call_count == 2
        assert tracker.session_cost > 0

    def test_budget_tracking(self):
        """Budget enforcement triggers alerts."""
        tracker = CostTracker(
            budgets=[Budget("session-limit", limit_usd=0.001, soft_limit_usd=0.0005)]
        )
        # First call below soft limit
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
        alerts_before = len(tracker.get_alerts())

        # Large call triggers budget exceeded
        tracker.record("claude-sonnet-4-20250514", input_tokens=500_000, output_tokens=100_000)
        assert tracker.budget_exceeded()
        alerts = tracker.get_alerts()
        assert len(alerts) > alerts_before

    def test_budget_not_exceeded(self):
        """Normal usage doesn't exceed budget."""
        tracker = CostTracker(
            budgets=[Budget("daily", limit_usd=100.0)]
        )
        tracker.record("gpt-4o-mini", input_tokens=1000, output_tokens=200)
        assert not tracker.budget_exceeded()

    def test_summary_formatting(self):
        """Summary string is non-empty and contains key info."""
        tracker = CostTracker(project_id="test")
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        summary = tracker.summary()
        assert "test" in summary
        assert "Total calls" in summary
        assert "Total cost" in summary
        assert "OpenAI" in summary

    def test_summary_with_budgets(self):
        """Summary includes budget information."""
        tracker = CostTracker(
            project_id="test",
            budgets=[Budget("daily", limit_usd=10.0)]
        )
        tracker.record("gpt-4o-mini", input_tokens=500)
        summary = tracker.summary()
        assert "Budgets" in summary
        assert "daily" in summary

    def test_breakdown_by_scope(self):
        """Project and session breakdowns are separate."""
        tracker = CostTracker()
        tracker.new_session("s1")
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)

        proj_bd = tracker.get_breakdown("project")
        sess_bd = tracker.get_breakdown("session")
        assert proj_bd.total.call_count == 1
        assert sess_bd.total.call_count == 1

        tracker.reset_session()
        assert tracker.get_breakdown("session").total.call_count == 0
        assert tracker.get_breakdown("project").total.call_count == 1

    def test_record_with_agent_and_operation(self):
        tracker = CostTracker()
        tracker.new_session()
        r = tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100,
                          agent_id="billing-agent", operation="classify")
        assert r.agent_id == "billing-agent"
        assert r.operation == "classify"

    def test_alert_callback(self):
        """Alert callback is invoked on budget warning."""
        alerts_received = []
        def callback(alert):
            alerts_received.append(alert)

        tracker = CostTracker(
            budgets=[Budget("small", limit_usd=0.001, soft_limit_usd=0.0001)],
            alert_callback=callback,
        )
        tracker.record("gpt-4o-mini", input_tokens=100_000, output_tokens=10_000)
        assert len(alerts_received) > 0

    def test_multiple_budgets(self):
        """Multiple budgets tracked simultaneously."""
        tracker = CostTracker(
            budgets=[
                Budget("daily", limit_usd=50.0, horizon="daily"),
                Budget("session", limit_usd=5.0, horizon="per-session"),
            ]
        )
        tracker.new_session()
        tracker.record("gpt-4o-mini", input_tokens=1000, output_tokens=200)
        alerts = tracker.check_budgets()
        # Should be within limits for small calls
        assert not tracker.budget_exceeded()

    def test_record_with_explicit_session_id(self):
        """Record uses explicit session_id field."""
        tracker = CostTracker()
        tracker.new_session("main")
        r = tracker.record("gpt-4o-mini", input_tokens=100, session_id="other")
        assert r.session_id == "other"

    def test_high_volume_tracking(self):
        """Tracks many calls correctly."""
        tracker = CostTracker()
        for i in range(100):
            tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
        assert tracker.call_count == 100
        assert tracker.total_cost > 0


class TestCostRecord:
    """Tests for CostRecord dataclass."""

    def test_defaults(self):
        r = CostRecord("id1", "m1", "p1", 100, 50)
        assert r.cached_input_tokens == 0
        assert r.cost_usd == 0.0
        assert r.agent_id == ""
        assert r.session_id == ""
        assert r.operation == ""
        assert r.timestamp > 0


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestEstimateCacheSavings:
    """Tests for cache savings estimation."""

    def test_empty_prompts(self):
        assert estimate_cache_savings([], call_count=10) == 0.0

    def test_single_call_no_savings(self):
        """Single call has nothing to cache against."""
        savings = estimate_cache_savings(
            ["You are a helpful assistant"], call_count=1
        )
        assert savings == 0.0

    def test_repeated_system_prompt(self):
        """Repeated prompt yields savings on subsequent calls."""
        sp = "You are a helpful assistant with detailed instructions " * 10
        savings = estimate_cache_savings([sp, sp, sp], call_count=3)
        assert savings > 0.0

    def test_unique_prompts_no_savings(self):
        """All-unique prompts yield no cache savings."""
        savings = estimate_cache_savings(
            ["Prompt A", "Prompt B", "Prompt C"], call_count=3
        )
        assert savings == 0.0

    def test_partial_repetition(self):
        """Only repeated prompts contribute to savings."""
        sp = "System prompt " * 100
        savings = estimate_cache_savings(
            [sp, sp, "Different prompt", sp], call_count=4
        )
        # 3 occurrences of sp → 2 benefit from caching
        assert savings > 0.0


class TestOptimizer:
    """Tests for Optimizer orchestration."""

    def test_empty_traces(self):
        optimizer = Optimizer()
        plan = optimizer.analyze_traces([])
        assert plan.is_empty
        assert plan.total_estimated_savings == 0.0

    def test_single_trace(self):
        optimizer = Optimizer()
        traces = [{
            "system_prompt": "You are a helpful assistant",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools_called": [],
        }]
        plan = optimizer.analyze_traces(traces)
        # Single trace — no caching, no bundling
        assert len(plan.cacheable_regions) == 0  # Only 1 occurrence

    def test_repeated_system_prompts_detected(self):
        optimizer = Optimizer()
        sp = "You are a helpful AI assistant " * 50
        traces = [
            {"system_prompt": sp, "messages": [{"role": "user", "content": "Q1"}], "tools_called": []}
            for _ in range(5)
        ]
        plan = optimizer.analyze_traces(traces)
        assert len(plan.cacheable_regions) > 0
        assert plan.cacheable_regions[0].occurrence_count == 5

    def test_verbose_system_prompt_pruning(self):
        optimizer = Optimizer()
        sp = "Very detailed system instructions " * 200  # ~800 chars → ~200 tokens
        sp += " with additional context " * 200  # ~2400 chars → ~600 tokens
        # That's ~800 tokens total — above 500 threshold
        traces = [{
            "system_prompt": sp,
            "messages": [],
            "tools_called": [],
        }]
        plan = optimizer.analyze_traces(traces)
        # Should detect verbosity
        assert len(plan.pruning_opportunities) > 0

    def test_long_conversation_pruning(self):
        optimizer = Optimizer()
        messages = [
            {"role": "user", "content": f"Message {i} with some content " * 10}
            for i in range(15)
        ]
        traces = [{
            "system_prompt": "You are helpful",
            "messages": messages,
            "tools_called": [],
        }]
        plan = optimizer.analyze_traces(traces)
        # 15 messages should trigger old-turn pruning suggestion
        assert len(plan.pruning_opportunities) > 0

    def test_tool_bundling_independent(self):
        optimizer = Optimizer()
        traces = [{
            "system_prompt": "You are helpful",
            "messages": [{"role": "user", "content": "Query"}],
            "tools_called": ["search_kb", "fetch_user_profile", "lookup_order"],
        }]
        plan = optimizer.analyze_traces(traces)
        # 3 independent lookups → can be parallelized
        assert len(plan.bundling_opportunities) > 0

    def test_tool_bundling_dependent(self):
        optimizer = Optimizer()
        traces = [{
            "system_prompt": "You are helpful",
            "messages": [{"role": "user", "content": "Query"}],
            "tools_called": ["create_order", "send_email"],
        }]
        plan = optimizer.analyze_traces(traces)
        # These are write operations — not classified as independent
        # Our heuristic only bundles search/fetch/read/get/lookup/query patterns
        assert len(plan.bundling_opportunities) == 0

    def test_routing_simple_operations(self):
        optimizer = Optimizer()
        traces = [{
            "system_prompt": "You are helpful",
            "messages": [{"role": "user", "content": "Classify this"}],
            "tools_called": [],
            "operation": "classify_sentiment",
        }]
        plan = optimizer.analyze_traces(traces)
        # classify → simple operation → routing suggestion
        assert len(plan.route_decisions) > 0
        assert plan.route_decisions[0].suggested_model == "claude-haiku-3-5-20241022"

    def test_routing_complex_operations_stay_expensive(self):
        optimizer = Optimizer()
        traces = [{
            "system_prompt": "You are helpful",
            "messages": [{"role": "user", "content": "Design a system"}],
            "tools_called": [],
            "operation": "design_architecture",
        }]
        plan = optimizer.analyze_traces(traces)
        # design → complex → no routing suggestion
        assert len(plan.route_decisions) == 0

    def test_estimate_cache_savings_integration(self):
        optimizer = Optimizer()
        sp = "System prompt " * 100
        savings = optimizer.estimate_cache_savings([sp, sp, sp], call_count=3)
        assert savings > 0.0

    def test_optimization_plan_total_savings(self):
        optimizer = Optimizer()
        sp = "System prompt " * 50
        traces = [
            {"system_prompt": sp, "messages": [{"role": "user", "content": "Q"}],
             "tools_called": ["search_x", "fetch_y"], "operation": "classify"}
            for _ in range(4)
        ]
        plan = optimizer.analyze_traces(traces)
        assert plan.total_estimated_savings > 0.0 or plan.total_tokens_saved > 0

    def test_optimization_plan_summary(self):
        optimizer = Optimizer()
        sp = "System " * 50
        traces = [
            {"system_prompt": sp, "messages": [{"role": "user", "content": "Q"}],
             "tools_called": ["search_x", "get_y"], "operation": "classify"}
            for _ in range(3)
        ]
        plan = optimizer.analyze_traces(traces)
        summary = plan.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestCacheableRegion:
    """Tests for CacheableRegion."""

    def test_text_preview(self):
        cr = CacheableRegion(
            text="A" * 200, estimated_tokens=50,
            occurrence_count=5, estimated_savings_per_call=0.01,
            total_estimated_savings=0.04,
        )
        assert len(cr.text_preview) <= 83  # 80 + "..."
        assert cr.text_preview.endswith("...")


class TestContextPruner:
    """Tests for ContextPruner."""

    def test_no_pruning_needed(self):
        pruner = ContextPruner(max_context_tokens=10_000)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = pruner.prune(messages)
        assert len(result) == len(messages)
        assert pruner.last_pruned_tokens == 0

    def test_pruning_old_messages(self):
        pruner = ContextPruner(max_context_tokens=50)  # Very small window
        messages = [
            {"role": "user", "content": "Message " + "x" * 100},  # ~25 tokens
            {"role": "assistant", "content": "Response " + "y" * 100},
            {"role": "user", "content": "Message 2 " + "z" * 100},
        ]
        result = pruner.prune(messages)
        # Should prune oldest messages
        assert len(result) <= len(messages)
        assert pruner.last_pruned_tokens > 0

    def test_pruning_with_system_prompt(self):
        pruner = ContextPruner(max_context_tokens=60)
        system_prompt = "System " * 30  # ~30 tokens
        messages = [
            {"role": "user", "content": "A" * 200},  # ~50 tokens
        ]
        result = pruner.prune(messages, system_prompt=system_prompt)
        assert len(result) <= 1


class TestModelRouter:
    """Tests for ModelRouter complexity-based routing."""

    def test_complex_to_expensive(self):
        router = ModelRouter()
        for op in ["analyze_sentiment", "evaluate_response", "design_workflow",
                    "debug_code", "plan_deployment", "compare_options"]:
            assert router.route(op) == router.expensive_model, f"Failed for: {op}"

    def test_simple_to_cheap(self):
        router = ModelRouter()
        for op in ["classify_text", "extract_entities", "summarize_article",
                    "translate_text", "format_output"]:
            assert router.route(op) == router.cheap_model, f"Failed for: {op}"

    def test_long_input_to_expensive(self):
        router = ModelRouter()
        long_input = "x" * 6000
        # Unknown operations with long input route to expensive for safety
        result = router.route("unknown_operation", input_text=long_input)
        assert result == router.expensive_model

    def test_default_simple_to_cheap(self):
        router = ModelRouter()
        # Unknown operations default to cheap for short input
        assert router.route("unknown_operation", input_text="short") == router.cheap_model


class TestToolBundler:
    """Tests for ToolBundler parallelization analysis."""

    def test_empty_tools(self):
        bundler = ToolBundler()
        assert bundler.analyze([]) == []

    def test_single_tool(self):
        bundler = ToolBundler()
        result = bundler.analyze(["search_kb"])
        assert result == [[0]]

    def test_independent_tools_grouped(self):
        bundler = ToolBundler()
        tools = ["search_kb", "fetch_user", "lookup_order"]
        result = bundler.analyze(tools)
        # All 3 should be in one batch
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_mixed_dependency(self):
        bundler = ToolBundler()
        tools = ["search_kb", "create_order", "fetch_user", "send_email"]
        result = bundler.analyze(tools)
        # Independent tools grouped, dependent ones separate
        total_tools = sum(len(batch) for batch in result)
        assert total_tools == 4


# ═══════════════════════════════════════════════════════════════════════
# EVAL TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestCostEfficiencyMetrics:
    """Tests for CostEfficiencyMetrics."""

    def test_success_rate(self):
        metrics = CostEfficiencyMetrics(
            total_cost=1.0, total_tasks=10, successful_tasks=8,
            failed_tasks=2, cost_per_task=0.10, cost_per_success=0.125,
            avg_input_tokens_per_task=500.0, avg_output_tokens_per_task=200.0,
            cache_hit_rate=0.15, avg_cost_per_1k_tokens=0.002,
        )
        assert metrics.success_rate == 0.8

    def test_success_rate_zero_tasks(self):
        metrics = CostEfficiencyMetrics(
            total_cost=0, total_tasks=0, successful_tasks=0,
            failed_tasks=0, cost_per_task=0.0, cost_per_success=0.0,
            avg_input_tokens_per_task=0.0, avg_output_tokens_per_task=0.0,
            cache_hit_rate=0.0, avg_cost_per_1k_tokens=0.0,
        )
        assert metrics.success_rate == 0.0


class TestEvaluateCostEfficiency:
    """Tests for evaluate_cost_efficiency."""

    def test_basic_evaluation(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        task_results = [{"task_id": "1", "success": True}]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert isinstance(report, CostEvalReport)
        assert report.metrics.total_tasks == 1
        assert report.metrics.successful_tasks == 1
        assert report.metrics.cost_per_task > 0

    def test_multiple_tasks(self):
        tracker = CostTracker()
        for i in range(10):
            tracker.record("gpt-4o-mini", input_tokens=200, output_tokens=50)
        task_results = [
            {"task_id": str(i), "success": i % 3 != 0}  # 2/3 success rate
            for i in range(10)
        ]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert report.metrics.total_tasks == 10
        assert report.metrics.cost_per_success > report.metrics.cost_per_task

    def test_with_budgets(self):
        tracker = CostTracker(
            budgets=[Budget("daily", limit_usd=10.0)]
        )
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        task_results = [{"task_id": "1", "success": True}]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert len(report.compliance) == 1
        assert report.compliance[0].budget_name == "daily"

    def test_recommendations_generated(self):
        tracker = CostTracker(
            budgets=[Budget("tiny", limit_usd=0.000001)]
        )
        tracker.record("claude-sonnet-4-20250514", input_tokens=100_000, output_tokens=10_000)
        task_results = [{"task_id": "1", "success": False}]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert len(report.recommendations) > 0

    def test_all_success(self):
        tracker = CostTracker()
        for i in range(5):
            tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
        task_results = [{"task_id": str(i), "success": True} for i in range(5)]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert report.metrics.success_rate == 1.0
        assert report.metrics.cost_per_success == report.metrics.cost_per_task


class TestCostVsQuality:
    """Tests for CostVsQuality and Pareto frontier."""

    def test_dominates(self):
        a = CostVsQuality("A", total_cost=1.0, success_rate=0.9, cost_per_success=1.11)
        b = CostVsQuality("B", total_cost=2.0, success_rate=0.8, cost_per_success=2.5)
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_no_domination_when_tied(self):
        a = CostVsQuality("A", total_cost=1.0, success_rate=0.9, cost_per_success=1.11)
        b = CostVsQuality("B", total_cost=1.0, success_rate=0.9, cost_per_success=1.11)
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_tradeoff_no_domination(self):
        a = CostVsQuality("A", total_cost=1.0, success_rate=0.8, cost_per_success=1.25)
        b = CostVsQuality("B", total_cost=2.0, success_rate=0.95, cost_per_success=2.11)
        # Neither dominates — tradeoff between cost and quality
        assert not a.dominates(b)
        assert not b.dominates(a)


class TestEvaluateCostVsQuality:
    """Tests for evaluate_cost_vs_quality."""

    def test_pareto_frontier(self):
        configs = [
            {"config_name": "cheap_model", "total_cost": 0.50, "success_rate": 0.70},
            {"config_name": "mid_model", "total_cost": 1.50, "success_rate": 0.85},
            {"config_name": "expensive_model", "total_cost": 5.00, "success_rate": 0.95},
            {"config_name": "dominated", "total_cost": 3.00, "success_rate": 0.80},
        ]
        all_points, frontier = evaluate_cost_vs_quality(configs)
        assert len(all_points) == 4
        # dominated (cost 3.0, SR 0.80) is dominated by mid_model (cost 1.50, SR 0.85)
        assert len(frontier) <= 3
        frontier_configs = {p.config_name for p in frontier}
        assert "dominated" not in frontier_configs

    def test_single_point(self):
        configs = [{"config_name": "only", "total_cost": 1.0, "success_rate": 0.9}]
        all_points, frontier = evaluate_cost_vs_quality(configs)
        assert len(frontier) == 1

    def test_empty_configs(self):
        all_points, frontier = evaluate_cost_vs_quality([])
        assert len(all_points) == 0
        assert len(frontier) == 0

    def test_frontier_sorted_by_cost(self):
        configs = [
            {"config_name": "expensive", "total_cost": 5.0, "success_rate": 0.99},
            {"config_name": "cheap", "total_cost": 0.5, "success_rate": 0.70},
            {"config_name": "mid", "total_cost": 2.0, "success_rate": 0.90},
        ]
        _, frontier = evaluate_cost_vs_quality(configs)
        # Should be sorted by cost_per_success ascending
        for i in range(1, len(frontier)):
            assert frontier[i - 1].cost_per_success <= frontier[i].cost_per_success


class TestDetectCostRegressions:
    """Tests for cost regression detection."""

    def test_no_regression(self):
        versions = [
            {"version": "v1", "cost": 1.00},
            {"version": "v2", "cost": 1.05},
        ]
        regressions = detect_cost_regressions(versions, threshold_pct=10.0)
        assert len(regressions) == 1
        assert not regressions[0].is_regression

    def test_regression_detected(self):
        versions = [
            {"version": "v1", "cost": 1.00},
            {"version": "v2", "cost": 1.75},  # 75% increase — regression!
        ]
        regressions = detect_cost_regressions(versions, threshold_pct=10.0)
        assert regressions[0].is_regression
        assert regressions[0].severity == "critical"

    def test_warning_regression(self):
        versions = [
            {"version": "v1", "cost": 1.00},
            {"version": "v2", "cost": 1.30},  # 30% increase — warning
        ]
        regressions = detect_cost_regressions(versions, threshold_pct=10.0)
        assert regressions[0].is_regression
        assert regressions[0].severity == "warning"

    def test_multiple_versions(self):
        versions = [
            {"version": "v1", "cost": 1.00},
            {"version": "v2", "cost": 0.90},
            {"version": "v3", "cost": 1.50},
            {"version": "v4", "cost": 1.55},
        ]
        regressions = detect_cost_regressions(versions, threshold_pct=10.0)
        assert len(regressions) == 3  # 3 transitions
        # v2→v3 is a regression (0.90→1.50, +66.7%)
        assert regressions[1].is_regression
        assert regressions[1].severity == "critical"
        # v1→v2 is a decrease (not regression)
        assert not regressions[0].is_regression

    def test_zero_cost_base(self):
        """Zero cost base handles gracefully."""
        versions = [
            {"version": "v1", "cost": 0.0},
            {"version": "v2", "cost": 1.0},
        ]
        regressions = detect_cost_regressions(versions, threshold_pct=10.0)
        assert regressions[0].cost_change_pct == 0.0


class TestFormatCostReport:
    """Tests for report formatting."""

    def test_text_format(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        task_results = [{"task_id": "1", "success": True}]
        report = evaluate_cost_efficiency(tracker, task_results)
        text = format_cost_report(report, format="text")
        assert isinstance(text, str)
        assert "Cost Efficiency" in text

    def test_markdown_format(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        task_results = [{"task_id": "1", "success": True}]
        report = evaluate_cost_efficiency(tracker, task_results)
        md = format_cost_report(report, format="markdown")
        assert "# Cost Efficiency Report" in md
        assert "|" in md

    def test_report_is_healthy(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=100)
        task_results = [{"task_id": "1", "success": True}]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert report.is_healthy

    def test_report_has_regressions(self):
        """Adding regressions to a report."""
        tracker = CostTracker()
        task_results = [{"task_id": "1", "success": True}]
        report = evaluate_cost_efficiency(tracker, task_results)
        report.regressions = [
            CostRegression("v1", "v2", 1.0, 1.80, 80.0, True, "critical")
        ]
        assert not report.is_healthy


class TestCostRegressionDescription:
    """Tests for CostRegression.description property."""

    def test_regression_description(self):
        cr = CostRegression("v1.0", "v2.0", 1.00, 1.50, 50.0, True, "critical")
        desc = cr.description
        assert "v1.0" in desc
        assert "v2.0" in desc
        assert "50.0%" in desc
        assert "⚠️" in desc

    def test_improvement_description(self):
        cr = CostRegression("v1.0", "v2.0", 1.00, 0.80, -20.0, False, "info")
        desc = cr.description
        assert "✓" in desc
        assert "20.0%" in desc

    def test_direction_language(self):
        cr_up = CostRegression("v1", "v2", 1.0, 1.5, 50.0, True, "critical")
        assert "increase" in cr_up.description

        cr_down = CostRegression("v1", "v2", 1.0, 0.5, -50.0, False, "info")
        assert "decrease" in cr_down.description


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndWorkflow:
    """Integration tests: tracker → optimizer → eval pipeline."""

    def test_full_pipeline(self):
        """Complete workflow: record costs, optimize, evaluate."""
        # 1. Track costs
        tracker = CostTracker(
            project_id="demo-agent",
            budgets=[Budget("daily", limit_usd=5.0, soft_limit_usd=3.0)]
        )
        tracker.new_session("demo-session")

        # Simulate 20 agent calls with mixed operations
        for i in range(20):
            model = "gpt-4o-mini" if i % 3 != 0 else "gpt-4o"
            operation = "classify" if i % 2 == 0 else "generate_response"
            tracker.record(
                model, input_tokens=300 + i * 10, output_tokens=100 + i * 5,
                agent_id="main-agent", operation=operation,
            )

        assert tracker.call_count == 20
        assert tracker.total_cost > 0

        # 2. Evaluate cost efficiency
        task_results = [
            {"task_id": str(i), "success": i % 5 != 0}  # 80% success rate
            for i in range(20)
        ]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert report.metrics.total_tasks == 20
        assert report.metrics.success_rate == 0.8
        assert report.metrics.cost_per_success > 0

        # 3. Check budget compliance
        assert len(report.compliance) == 1
        assert report.compliance[0].budget_name == "daily"

        # 4. Check recommendations
        assert len(report.recommendations) > 0

    def test_optimize_from_tracker_data(self):
        """Feed tracker data into optimizer."""
        tracker = CostTracker()
        sp = "You are a helpful agent " * 50

        for i in range(10):
            tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=200,
                          agent_id="agent", operation="classify")

        # Build traces from tracker records
        traces = [
            {"system_prompt": sp, "messages": [{"role": "user", "content": f"Task {i}"}],
             "tools_called": ["search_kb", "fetch_data"], "operation": "classify"}
            for i in range(10)
        ]

        optimizer = Optimizer()
        plan = optimizer.analyze_traces(traces)

        assert plan.cacheable_regions or plan.route_decisions or plan.bundling_opportunities
        assert isinstance(plan.summary(), str)

    def test_cost_vs_quality_analysis(self):
        """Compare different agent configurations for cost efficiency."""
        configs = [
            {"config_name": "gpt4o_full", "total_cost": 3.50, "success_rate": 0.95},
            {"config_name": "gpt4o_mini_basic", "total_cost": 0.45, "success_rate": 0.72},
            {"config_name": "gpt4o_mini_with_guardrails", "total_cost": 0.60, "success_rate": 0.80},
            {"config_name": "sonnet_4", "total_cost": 4.20, "success_rate": 0.97},
            {"config_name": "haiku_3_5", "total_cost": 0.30, "success_rate": 0.65},
        ]
        all_points, frontier = evaluate_cost_vs_quality(configs)

        # Frontier should have non-dominated points
        assert len(frontier) >= 2
        # gpt4o_mini_basic (0.45, 0.72) is NOT dominated by haiku (0.30, 0.65) — haiku is cheaper but worse quality
        # haiku (0.30, 0.65) is dominated by gpt4o_mini_basic? No — 0.30 < 0.45, 0.65 < 0.72. Different tradeoff.
        # Both should be on the frontier
        frontier_names = {p.config_name for p in frontier}
        assert len(frontier_names) >= 2

    def test_realistic_agent_costs(self):
        """Realistic cost scenario: GPT-4o primary with Claude fallback."""
        tracker = CostTracker(
            project_id="production-agent",
            budgets=[Budget("monthly", limit_usd=200.0, soft_limit_usd=150.0)]
        )

        # Simulate 1000 calls over a day
        for i in range(1000):
            if i % 5 == 0:
                # Complex reasoning → expensive model
                tracker.record("gpt-4o", input_tokens=2000, output_tokens=500,
                              agent_id="reasoner", operation="analyze")
            else:
                # Simple tasks → cheap model
                tracker.record("gpt-4o-mini", input_tokens=300, output_tokens=100,
                              agent_id="classifier", operation="classify")

        total_cost = tracker.total_cost
        # 200 GPT-4o calls + 800 GPT-4o-mini calls
        gpt4o_cost = get_cost("gpt-4o", input_tokens=2000, output_tokens=500) * 200
        gpt4mini_cost = get_cost("gpt-4o-mini", input_tokens=300, output_tokens=100) * 800
        expected = gpt4o_cost + gpt4mini_cost
        assert total_cost == pytest.approx(expected, rel=1e-3)

        # Report
        task_results = [{"task_id": str(i), "success": True} for i in range(1000)]
        report = evaluate_cost_efficiency(tracker, task_results)
        assert report.metrics.total_tasks == 1000

    def test_budget_exhaustion_scenario(self):
        """Agent runs until budget is exhausted."""
        tracker = CostTracker(
            budgets=[Budget("hard-limit", limit_usd=0.001)]
        )
        call_count = 0
        for i in range(100):
            tracker.record("gpt-4o", input_tokens=10000)  # Expensive!
            call_count += 1
            if tracker.budget_exceeded():
                break
        assert call_count < 100  # Should stop before 100 calls
        assert tracker.budget_exceeded()
