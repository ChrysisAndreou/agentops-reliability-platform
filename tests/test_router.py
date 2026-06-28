"""
Tests for agentops.router — Model Router with cost/latency/capability routing.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agentops.llm.backend import LLMResponse, LLMBackend
from agentops.router import (
    BackendConfig,
    ModelRouter,
    RoutingStrategy,
    RouterStats,
    BudgetExceededError,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def _make_response(content="test response", model="test-model",
                   provider="test", input_tokens=10, output_tokens=20,
                   cost_usd=0.0001, latency_ms=50.0):
    return LLMResponse(
        content=content, model=model, provider=provider,
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost_usd, latency_ms=latency_ms,
    )


@pytest.fixture
def mock_backend_factory():
    """Patch create_backend to return MagicMock instances."""
    with patch("agentops.router.router.create_backend") as factory:
        def make_mock(**kwargs):
            mock = MagicMock(spec=LLMBackend)
            mock.model = kwargs.get("model", "mock")
            mock.provider = kwargs.get("provider", "mock")
            mock.temperature = kwargs.get("temperature", 0.0)
            mock.max_tokens = kwargs.get("max_tokens", 4096)
            mock.chat = MagicMock(return_value=_make_response(
                model=mock.model, provider=mock.provider))
            mock.reset_stats = MagicMock()
            return mock
        factory.side_effect = make_mock
        yield factory


@pytest.fixture
def cheap_backend():
    return BackendConfig(
        model="gpt-4o-mini", provider="openai",
        cost_input=0.15, cost_output=0.60,
        capabilities={"chat", "code", "json_mode"},
        max_tokens=4096,
    )


@pytest.fixture
def fast_backend():
    return BackendConfig(
        model="claude-3-haiku-20240307", provider="anthropic",
        cost_input=0.25, cost_output=1.25,
        capabilities={"chat", "vision", "code"},
        max_tokens=4096,
    )


@pytest.fixture
def capable_backend():
    return BackendConfig(
        model="gpt-4o", provider="openai",
        cost_input=2.50, cost_output=10.00,
        capabilities={"chat", "code", "vision", "json_mode", "function_calling"},
        max_tokens=16384,
    )


@pytest.fixture
def router(mock_backend_factory, cheap_backend, fast_backend, capable_backend):
    return ModelRouter([cheap_backend, fast_backend, capable_backend])


# ═══════════════════════════════════════════════════════════════════════
# BackendConfig Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBackendConfig:
    def test_defaults(self):
        cfg = BackendConfig(model="test", provider="openai",
                            cost_input=1.0, cost_output=2.0)
        assert cfg.model == "test"
        assert cfg.provider == "openai"
        assert cfg.cost_input == 1.0
        assert cfg.cost_output == 2.0
        assert cfg.capabilities == {"chat"}
        assert cfg.weight == 1.0
        assert cfg.enabled is True

    def test_custom_capabilities(self):
        cfg = BackendConfig(model="test", provider="openai",
                            cost_input=1.0, cost_output=2.0,
                            capabilities={"code", "vision"})
        assert cfg.capabilities == {"code", "vision"}

    def test_cost_properties(self):
        cfg = BackendConfig(model="test", provider="openai",
                            cost_input=0.15, cost_output=0.60)
        assert cfg.cost_per_1m_input == 0.15
        assert cfg.cost_per_1m_output == 0.60


# ═══════════════════════════════════════════════════════════════════════
# RouterStats Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRouterStats:
    def test_defaults(self):
        stats = RouterStats()
        assert stats.calls == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_cost_usd == 0.0
        assert stats.errors == 0
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency(self):
        stats = RouterStats()
        stats.latencies_ms = [100, 200, 300]
        assert stats.avg_latency_ms == 200.0

    def test_p50_latency(self):
        stats = RouterStats()
        stats.latencies_ms = [100, 200, 300, 400, 500]
        assert stats.p50_latency_ms == 300.0

    def test_p95_latency(self):
        stats = RouterStats()
        stats.latencies_ms = list(range(1, 101))  # 1..100
        assert stats.p95_latency_ms == 96.0  # index = floor(100 * 0.95) = 95, value = 96

    def test_p50_empty(self):
        stats = RouterStats()
        assert stats.p50_latency_ms == 0.0

    def test_to_dict(self):
        stats = RouterStats()
        stats.calls = 5
        stats.total_cost_usd = 0.05
        d = stats.to_dict()
        assert d["calls"] == 5
        assert d["total_cost_usd"] == 0.05


# ═══════════════════════════════════════════════════════════════════════
# ModelRouter Initialization
# ═══════════════════════════════════════════════════════════════════════

class TestModelRouterInit:
    def test_empty_backends_raises(self):
        with pytest.raises(ValueError, match="At least one enabled backend"):
            ModelRouter([])

    def test_default_strategy(self, cheap_backend):
        router = ModelRouter([cheap_backend])
        assert router.strategy == RoutingStrategy.CHEAPEST

    def test_custom_strategy(self, cheap_backend, fast_backend):
        router = ModelRouter([cheap_backend, fast_backend],
                             strategy=RoutingStrategy.FASTEST)
        assert router.strategy == RoutingStrategy.FASTEST

    def test_no_budget_by_default(self, cheap_backend):
        router = ModelRouter([cheap_backend])
        assert router.budget_limit_usd is None

    def test_budget_limit(self, cheap_backend):
        router = ModelRouter([cheap_backend], budget_limit_usd=5.00)
        assert router.budget_limit_usd == 5.00

    def test_disabled_backend_excluded(self, cheap_backend, fast_backend):
        fast_backend.enabled = False
        router = ModelRouter([cheap_backend, fast_backend])
        # Only 1 enabled backend
        assert len(router.backends) == 1

    def test_all_disabled_raises(self, cheap_backend):
        cheap_backend.enabled = False
        with pytest.raises(ValueError, match="At least one enabled backend"):
            ModelRouter([cheap_backend])

    def test_backend_configs_stored(self, cheap_backend, fast_backend):
        router = ModelRouter([cheap_backend, fast_backend])
        assert len(router.backends) == 2
        assert router.backends[0].model == "gpt-4o-mini"


# ═══════════════════════════════════════════════════════════════════════
# ModelRouter.chat() Tests
# ═══════════════════════════════════════════════════════════════════════

class TestModelRouterChat:
    def test_basic_chat(self, router):
        response = router.chat("Hello")
        assert response.content == "test response"
        assert response.model in {"gpt-4o-mini", "claude-3-haiku-20240307", "gpt-4o"}

    def test_chat_with_messages_list(self, router):
        response = router.chat([{"role": "user", "content": "Hi"}])
        assert response.content == "test response"

    def test_chat_with_system(self, router):
        response = router.chat("Hello", system="You are helpful")
        assert response.content == "test response"

    def test_chat_with_strategy_override(self, router):
        response = router.chat("Hello", strategy=RoutingStrategy.FASTEST)
        assert response.content == "test response"

    def test_chat_with_capabilities_override(self, router):
        response = router.chat("Describe this image",
                               capabilities={"vision"})
        assert response.content == "test response"

    def test_chat_updates_stats(self, router):
        router.chat("Hello")
        # After chat, at least one model should have been used
        stats = router.stats()
        assert len(stats) >= 1
        for model, s in stats.items():
            assert s.calls >= 1

    def test_chat_updates_total_cost(self, router):
        initial = router._total_cost_usd
        router.chat("Hello")
        assert router._total_cost_usd > initial

    def test_multiple_chats_increment_stats(self, router):
        for i in range(5):
            router.chat(f"Prompt {i}")
        total_calls = sum(s.calls for s in router.stats().values())
        assert total_calls == 5


# ═══════════════════════════════════════════════════════════════════════
# Routing Strategy Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRoutingCheapest:
    def test_selects_cheapest(self, router):
        router.strategy = RoutingStrategy.CHEAPEST
        selected = router.route()
        assert selected == "gpt-4o-mini"  # 0.15/0.60 is cheapest

    def test_all_same_cost(self, mock_backend_factory):
        a = BackendConfig(model="a", provider="test", cost_input=1.0, cost_output=1.0)
        b = BackendConfig(model="b", provider="test", cost_input=1.0, cost_output=1.0)
        r = ModelRouter([a, b])
        selected = r.route()
        assert selected in {"a", "b"}


class TestRoutingFastest:
    def test_selects_lowest_latency(self, router):
        router.strategy = RoutingStrategy.FASTEST
        # No history yet — first call uses cheapest
        router._latency_history["gpt-4o-mini"] = [200]
        router._latency_history["claude-3-haiku-20240307"] = [50]
        router._latency_history["gpt-4o"] = [500]
        selected = router.route()
        assert selected == "claude-3-haiku-20240307"


class TestRoutingCapability:
    def test_selects_capable_backend(self, router):
        router.strategy = RoutingStrategy.CAPABILITY
        selected = router.route(capabilities={"vision"})
        # gpt-4o-mini doesn't have vision, but claude-haiku and gpt-4o do
        assert selected != "gpt-4o-mini"

    def test_no_capable_backend_raises(self, router):
        router.strategy = RoutingStrategy.CAPABILITY
        with pytest.raises(ValueError, match="No backend supports"):
            router.route(capabilities={"quantum_computing", "time_travel"})


class TestRoutingRoundRobin:
    def test_cycles_through_backends(self, mock_backend_factory):
        a = BackendConfig(model="a", provider="test", cost_input=1.0, cost_output=1.0)
        b = BackendConfig(model="b", provider="test", cost_input=1.0, cost_output=1.0)
        c = BackendConfig(model="c", provider="test", cost_input=1.0, cost_output=1.0)
        r = ModelRouter([a, b, c], strategy=RoutingStrategy.ROUND_ROBIN)
        selected = []
        for _ in range(6):
            selected.append(r.route())
        assert selected == ["a", "b", "c", "a", "b", "c"]


class TestRoutingFailover:
    def test_uses_primary(self, router):
        router.strategy = RoutingStrategy.FAILOVER
        selected = router.route()
        assert selected == "gpt-4o-mini"  # First in list

    def test_fallback_on_failure(self, router):
        router.strategy = RoutingStrategy.FAILOVER
        # Prime backends so _backend_instances is populated
        router.chat("prime")
        # Make the primary fail
        router._backend_instances["gpt-4o-mini"].chat.side_effect = Exception("fail")
        # Reset stats so we don't count the priming call
        router.reset_stats()
        response = router.chat("Hello")
        # Should fall back to second backend
        assert response.content == "test response"
        assert response.model != "gpt-4o-mini"

    def test_all_failover_raises(self, router):
        router.strategy = RoutingStrategy.FAILOVER
        # Prime ALL backends by calling chat with different strategies
        for backend_cfg in router.backends:
            router._get_backend(backend_cfg.model)
        for inst in router._backend_instances.values():
            inst.chat.side_effect = Exception("fail")
        with pytest.raises(RuntimeError, match="All .* backends failed"):
            router.chat("Hello")


# ═══════════════════════════════════════════════════════════════════════
# Budget Enforcement Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBudget:
    def test_within_budget(self, cheap_backend, mock_backend_factory):
        router = ModelRouter([cheap_backend], budget_limit_usd=1.00)
        response = router.chat("Hello")
        assert response.content == "test response"

    def test_exceeds_budget(self, cheap_backend, mock_backend_factory):
        router = ModelRouter([cheap_backend], budget_limit_usd=0.000001)
        with pytest.raises(BudgetExceededError):
            router.chat("Hello")

    def test_budget_warning_threshold(self, cheap_backend, mock_backend_factory):
        router = ModelRouter([cheap_backend], budget_limit_usd=0.01,
                             budget_alert_threshold=0.01)  # Alert at 1%
        # Cost per call is ~$0.0001, should alert after enough calls
        for _ in range(5):
            try:
                router.chat("Hello")
            except BudgetExceededError:
                break

    def test_budget_none_unlimited(self, cheap_backend, mock_backend_factory):
        router = ModelRouter([cheap_backend], budget_limit_usd=None)
        for _ in range(10):
            router.chat("Hello")
        # No exception raised


# ═══════════════════════════════════════════════════════════════════════
# Stats Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_after_chat(self, router):
        router.chat("Hello")
        s = router.stats()
        assert len(s) >= 1
        for model, stats in s.items():
            assert isinstance(stats, RouterStats)
            assert stats.calls >= 1

    def test_reset_stats(self, router):
        router.chat("Hello")
        router.reset_stats()
        s = router.stats()
        for stats in s.values():
            assert stats.calls == 0
        assert router._total_cost_usd == 0.0

    def test_stats_summary(self, router):
        router.chat("Hello")
        summary = router.stats()
        assert isinstance(summary, dict)

    def test_total_cost_property(self, router):
        initial = router.total_cost_usd
        router.chat("Hello")
        assert router.total_cost_usd > initial


# ═══════════════════════════════════════════════════════════════════════
# Strategy String Tests
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyFromString:
    def test_valid_strings(self):
        assert ModelRouter._strategy_from_string("cheapest") == RoutingStrategy.CHEAPEST
        assert ModelRouter._strategy_from_string("fastest") == RoutingStrategy.FASTEST
        assert ModelRouter._strategy_from_string("capability") == RoutingStrategy.CAPABILITY
        assert ModelRouter._strategy_from_string("round_robin") == RoutingStrategy.ROUND_ROBIN
        assert ModelRouter._strategy_from_string("failover") == RoutingStrategy.FAILOVER

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ModelRouter._strategy_from_string("invalid")

    def test_enum_passed_through(self):
        result = ModelRouter._strategy_from_string(RoutingStrategy.FASTEST)
        assert result == RoutingStrategy.FASTEST


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_workflow(self, router):
        # Chat multiple times with different strategies
        responses = []
        for strat in RoutingStrategy:
            if strat == RoutingStrategy.CAPABILITY:
                resp = router.chat("Hello", strategy=strat,
                                   capabilities={"chat"})
            else:
                resp = router.chat("Hello", strategy=strat)
            responses.append(resp)

        assert all(r.content == "test response" for r in responses)
        assert len(responses) == len(RoutingStrategy)

    def test_convenience_string_strategy(self, cheap_backend):
        router = ModelRouter([cheap_backend], strategy="fastest")  # type: ignore
        assert router.strategy == RoutingStrategy.FASTEST

    def test_chat_returns_llm_response(self, router):
        response = router.chat("Hello")
        assert isinstance(response, LLMResponse)
        assert response.content == "test response"
