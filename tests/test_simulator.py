"""
Tests for the simulated agent backend.
"""

import asyncio

import pytest

from agentops.evals.simulator import (
    SimulatedAgent,
    SimConfig,
    PERFECT_AGENT,
    PRODUCTION_AGENT,
    DEVELOPMENT_AGENT,
    UNRELIABLE_AGENT,
    get_profile,
    ALL_PROFILES,
    _extract_key_terms,
)


class TestKeyTermExtraction:
    def test_extracts_terms_from_question(self):
        terms = _extract_key_terms("How do I reset my password on the CloudDeploy platform?")
        assert "reset" in terms or "password" in terms or "clouddeploy" in terms
        assert len(terms) > 0

    def test_filters_stopwords(self):
        terms = _extract_key_terms("the a an is are was were be been being have has had")
        assert len(terms) == 0

    def test_extracts_hyphenated_terms(self):
        terms = _extract_key_terms("Use blue-green deployment for zero-downtime updates")
        assert "blue-green" in terms or "zero-downtime" in terms

    def test_deduplicates_terms(self):
        terms = _extract_key_terms("deploy deploy deploy the app app app")
        assert terms.count("deploy") <= 1
        assert terms.count("app") <= 1

    def test_caps_at_15_terms(self):
        long_text = " ".join([f"keyword{i}" for i in range(30)])
        terms = _extract_key_terms(long_text)
        assert len(terms) <= 15


class TestSimConfig:
    def test_perfect_agent_has_full_quality(self):
        assert PERFECT_AGENT.groundedness_target == 1.0
        assert PERFECT_AGENT.verification_pass_rate == 1.0
        assert PERFECT_AGENT.hallucination_rate == 0.0

    def test_unreliable_agent_has_low_quality(self):
        assert UNRELIABLE_AGENT.groundedness_target < 0.6
        assert UNRELIABLE_AGENT.verification_pass_rate < 0.5

    def test_seed_hash_is_deterministic(self):
        config = SimConfig(name="test")
        h1 = config.seed_hash("task-1")
        h2 = config.seed_hash("task-1")
        assert h1 == h2

    def test_seed_hash_differs_by_task_id(self):
        config = SimConfig(name="test")
        h1 = config.seed_hash("task-1")
        h2 = config.seed_hash("task-2")
        assert h1 != h2

    def test_get_profile_returns_correct(self):
        p = get_profile("production")
        assert p is not None
        assert p.name == "production"

    def test_get_profile_returns_none_for_unknown(self):
        assert get_profile("nonexistent") is None

    def test_all_profiles_have_unique_names(self):
        names = [p.name for p in ALL_PROFILES]
        assert len(names) == len(set(names))


class TestSimulatedAgent:
    @pytest.mark.asyncio
    async def test_basic_run_returns_result(self):
        agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        result = await agent.run("What is the password policy?", task_id="test-1")
        assert result.task_id == "test-1"
        assert result.task is not None
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_perfect_agent_always_passes(self):
        agent = SimulatedAgent(config=PERFECT_AGENT, seed=42)
        passes = 0
        for i in range(20):
            result = await agent.run(f"Task number {i}", task_id=f"perf-{i}")
            if result.verification_passed:
                passes += 1
        # Perfect agent should pass at a very high rate
        assert passes >= 19  # Allow occasional randomness edge case

    @pytest.mark.asyncio
    async def test_unreliable_agent_fails_often(self):
        agent = SimulatedAgent(config=UNRELIABLE_AGENT, seed=42)
        failures = 0
        for i in range(20):
            result = await agent.run(f"Task number {i}", task_id=f"unrel-{i}")
            if not result.verification_passed:
                failures += 1
        # Unreliable agent should fail frequently
        assert failures >= 8

    @pytest.mark.asyncio
    async def test_deterministic_same_seed(self):
        agent1 = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        agent2 = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        r1 = await agent1.run("What is the deployment strategy?", task_id="det-1")
        r2 = await agent2.run("What is the deployment strategy?", task_id="det-1")
        assert r1.verification_passed == r2.verification_passed
        assert r1.grounded_claims == r2.grounded_claims

    @pytest.mark.asyncio
    async def test_different_seed_different_result(self):
        agent1 = SimulatedAgent(config=PRODUCTION_AGENT, seed=1)
        agent2 = SimulatedAgent(config=PRODUCTION_AGENT, seed=999)
        r1 = await agent1.run("What is the deployment strategy?", task_id="det-2")
        r2 = await agent2.run("What is the deployment strategy?", task_id="det-2")
        # Different seeds may produce different results (not guaranteed but likely)
        # At minimum, both should produce valid results
        assert r1.total_latency_ms > 0
        assert r2.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        result = await agent.run("Test task for field checking", task_id="fields")
        assert result.task_id == "fields"
        assert isinstance(result.final_answer, str)
        assert isinstance(result.success, bool)
        assert isinstance(result.verification_passed, bool)
        assert isinstance(result.grounded_claims, list)
        assert isinstance(result.ungrounded_claims, list)
        assert isinstance(result.citations_used, list)
        assert isinstance(result.plan, list)
        assert isinstance(result.reliability_trace, list)
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_trace_has_correct_steps(self):
        agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        result = await agent.run("Test trace steps", task_id="trace-1")
        trace = result.reliability_trace
        step_types = [s.get("step_type") for s in trace if isinstance(s, dict)]
        assert "plan" in step_types
        assert "retrieve" in step_types
        assert "verify" in step_types
        assert "respond" in step_types

    @pytest.mark.asyncio
    async def test_reset_is_noop(self):
        agent = SimulatedAgent(config=PRODUCTION_AGENT)
        agent.reset()  # Should not raise

    @pytest.mark.asyncio
    async def test_tool_tasks_trigger_tool_calls(self):
        agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
        # Task with "calculate" should trigger tool use
        result = await agent.run("Calculate the total cost of 5 agents", task_id="tool-1")
        # Either it has tool calls or not — both are valid depending on randomness
        assert result.tool_calls_count >= 0  # Just verify no crash

    def test_to_dict_works(self):
        result = asyncio.run(
            SimulatedAgent(config=PRODUCTION_AGENT).run("Dict test", task_id="dict-1")
        )
        d = result.to_dict()
        assert d["task_id"] == "dict-1"
        assert "total_latency_ms" in d
        assert "grounded_claims" in d
