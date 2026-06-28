"""
Tests for the agentops.llm module — backend abstraction, agent integration,
and multi-provider support.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agentops.llm.backend import (
    AnthropicBackend,
    LLMResponse,
    OpenAIBackend,
    available_backends,
    check_availability,
    create_backend,
)
from agentops.llm.agent import (
    LLMAgent,
    LLMAgentConfig,
    _parse_plan,
    _parse_tool_call,
    _extract_citations,
)


# ── Backend: LLMResponse ──────────────────────────────────────────────

class TestLLMResponse:
    def test_basic_response(self):
        r = LLMResponse(content="Hello", model="gpt-4o", provider="openai")
        assert r.content == "Hello"
        assert r.model == "gpt-4o"
        assert r.provider == "openai"
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_response_with_tokens(self):
        r = LLMResponse(
            content="Hi", model="gpt-4o", provider="openai",
            input_tokens=10, output_tokens=5, total_tokens=15,
            cost_usd=0.0005, finish_reason="stop",
        )
        assert r.input_tokens == 10
        assert r.output_tokens == 5
        assert r.total_tokens == 15
        assert r.cost_usd == 0.0005
        assert r.total_cost == 0.0005
        assert r.finish_reason == "stop"


# ── Backend: OpenAIBackend ────────────────────────────────────────────

class TestOpenAIBackend:
    def test_init_default(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend()
            assert backend.model == "gpt-4o"
            assert backend.provider == "openai"
            assert backend.temperature == 0.0

    def test_init_with_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend(model="gpt-4o-mini")
            assert backend.model == "gpt-4o-mini"

    def test_init_deepseek(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend(
                model="deepseek-chat",
                base_url="https://api.deepseek.com/v1",
            )
            assert backend.model == "deepseek-chat"
            assert backend.provider == "deepseek"

    def test_custom_api_key(self):
        backend = OpenAIBackend(api_key="custom-key")
        assert backend._api_key == "custom-key"

    def test_chat_mocked(self):
        """Test that chat() correctly wraps the OpenAI client."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Hello, World!"
            mock_choice.finish_reason = "stop"
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 5
            mock_usage.completion_tokens = 3
            mock_usage.total_tokens = 8
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_client.chat.completions.create.return_value = mock_response
            backend._client = mock_client

            response = backend.chat("Hi")
            assert response.content == "Hello, World!"
            assert response.model == "gpt-4o"
            assert response.input_tokens == 5
            assert response.output_tokens == 3
            assert response.total_tokens == 8
            assert response.cost_usd > 0
            assert response.latency_ms >= 0
            assert backend.call_count == 1

    def test_chat_list_messages(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_choice.finish_reason = "stop"
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 2
            mock_usage.total_tokens = 12
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_client.chat.completions.create.return_value = mock_response
            backend._client = mock_client

            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            response = backend.chat(messages)
            assert response.content == "Response"

    def test_stats_accumulation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "OK"
            mock_choice.finish_reason = "stop"
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 3
            mock_usage.completion_tokens = 1
            mock_usage.total_tokens = 4
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_client.chat.completions.create.return_value = mock_response
            backend._client = mock_client

            backend.chat("a")
            backend.chat("b")
            backend.chat("c")
            assert backend.call_count == 3
            assert backend.total_input_tokens == 9
            assert backend.total_output_tokens == 3
            assert backend.total_cost > 0

            backend.reset_stats()
            assert backend.call_count == 0
            assert backend.total_input_tokens == 0

    def test_health_check(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend()
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "pong"
            mock_choice.finish_reason = "stop"
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 1
            mock_usage.completion_tokens = 1
            mock_usage.total_tokens = 2
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage = mock_usage
            mock_client.chat.completions.create.return_value = mock_response
            backend._client = mock_client

            assert backend.health_check() is True

    def test_health_check_failure(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = OpenAIBackend()
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API down")
            backend._client = mock_client

            assert backend.health_check() is False


# ── Backend: AnthropicBackend ─────────────────────────────────────────

class TestAnthropicBackend:
    def test_init_default(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            backend = AnthropicBackend()
            assert backend.model == "claude-3-5-sonnet-20241022"
            assert backend.provider == "anthropic"

    def test_init_with_model(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            backend = AnthropicBackend(model="claude-3-opus-20240229")
            assert backend.model == "claude-3-opus-20240229"

    def test_chat_mocked(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            backend = AnthropicBackend()
            mock_client = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Hello from Claude"
            mock_usage = MagicMock()
            mock_usage.input_tokens = 10
            mock_usage.output_tokens = 5
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_response.usage = mock_usage
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            backend._client = mock_client

            response = backend.chat("Hello")
            assert "Hello from Claude" in response.content
            assert response.provider == "anthropic"
            assert response.input_tokens == 10
            assert response.output_tokens == 5

    def test_chat_with_system_message(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            backend = AnthropicBackend()
            mock_client = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_usage = MagicMock()
            mock_usage.input_tokens = 5
            mock_usage.output_tokens = 3
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_response.usage = mock_usage
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            backend._client = mock_client

            response = backend.chat("Question", system="You are helpful.")
            assert response.content == "Response"
            # Verify system message was passed to Anthropic
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "You are helpful."

    def test_stats(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            backend = AnthropicBackend()
            assert backend.total_cost == 0.0
            assert backend.call_count == 0


# ── Backend: Factory ──────────────────────────────────────────────────

class TestCreateBackend:
    def test_explicit_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = create_backend(provider="openai", model="gpt-4o-mini")
            assert isinstance(backend, OpenAIBackend)
            assert backend.model == "gpt-4o-mini"

    def test_explicit_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            backend = create_backend(provider="anthropic")
            assert isinstance(backend, AnthropicBackend)

    def test_explicit_deepseek(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-ds"}):
            backend = create_backend(provider="deepseek", model="deepseek-chat")
            assert isinstance(backend, OpenAIBackend)
            assert backend.provider == "deepseek"
            assert backend._base_url == "https://api.deepseek.com/v1"

    def test_auto_detect_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            backend = create_backend()
            assert isinstance(backend, OpenAIBackend)

    def test_auto_detect_deepseek(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-ds",
            "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        }, clear=True):
            backend = create_backend()
            assert isinstance(backend, OpenAIBackend)
            assert backend.provider == "deepseek"

    def test_auto_detect_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            backend = create_backend()
            assert isinstance(backend, AnthropicBackend)

    def test_no_keys_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No LLM API key found"):
                create_backend()


# ── Availability helpers ──────────────────────────────────────────────

class TestAvailability:
    def test_available_backends_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            result = available_backends()
            assert "openai" in result

    def test_available_backends_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            result = available_backends()
            assert "anthropic" in result

    def test_available_backends_deepseek(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-ds",
            "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        }, clear=True):
            result = available_backends()
            assert "deepseek" in result

    def test_available_backends_none(self):
        with patch.dict(os.environ, {}, clear=True):
            result = available_backends()
            assert result == []

    def test_check_availability(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
        }, clear=True):
            avail = check_availability()
            assert avail["openai"] is True
            assert avail["anthropic"] is False
            assert avail["deepseek"] is False


# ── Agent: LLMAgentConfig ─────────────────────────────────────────────

class TestLLMAgentConfig:
    def test_defaults(self):
        config = LLMAgentConfig()
        assert config.temperature == 0.0
        assert config.max_tokens == 2048
        assert config.enable_plan is True
        assert config.enable_verify is True

    def test_custom(self):
        config = LLMAgentConfig(
            temperature=0.5, max_tokens=512,
            enable_plan=False, enable_verify=True,
        )
        assert config.temperature == 0.5
        assert config.enable_plan is False


# ── Agent: LLMAgent ───────────────────────────────────────────────────

class TestLLMAgent:
    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend that returns deterministic responses."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            backend = OpenAIBackend()
            mock_client = MagicMock()

            def make_response(content):
                mock_choice = MagicMock()
                mock_choice.message.content = content
                mock_choice.finish_reason = "stop"
                mock_usage = MagicMock()
                mock_usage.prompt_tokens = 10
                mock_usage.completion_tokens = len(content) // 4
                mock_usage.total_tokens = 10 + len(content) // 4
                mock_resp = MagicMock()
                mock_resp.choices = [mock_choice]
                mock_resp.usage = mock_usage
                return mock_resp

            # 3 calls: plan, verify, respond
            # (retrieve skipped — no retrieval_fn; execute skipped — no tools registered)
            mock_client.chat.completions.create.side_effect = [
                make_response("1. Analyze the question\n2. Retrieve information\n3. Formulate answer"),
                make_response('{"verification_passed": true, "grounded_claims": ["Docker is a runtime"], "ungrounded_claims": [], "notes": "All claims grounded"}'),
                make_response("Docker is a container runtime used for packaging applications. Kubernetes orchestrates Docker containers in production."),
            ]
            backend._client = mock_client
            return backend

    @pytest.mark.asyncio
    async def test_run_basic(self, mock_backend):
        agent = LLMAgent(backend=mock_backend)
        result = await agent.run("What is Docker?", task_id="t1")
        assert result.task_id == "t1"
        assert "Docker" in result.final_answer
        assert result.verification_passed is True
        assert result.success is True
        assert len(result.plan) >= 2
        assert result.total_latency_ms > 0
        assert len(result.reliability_trace) == 5  # plan, retrieve, execute, verify, respond

    @pytest.mark.asyncio
    async def test_run_trace_structure(self, mock_backend):
        agent = LLMAgent(backend=mock_backend)
        result = await agent.run("Explain Kubernetes", task_id="t2")
        trace = result.reliability_trace
        step_names = [s["step_name"] for s in trace]
        assert step_names == ["plan", "retrieve", "execute", "verify", "respond"]

    @pytest.mark.asyncio
    async def test_config_disables_steps(self, mock_backend):
        config = LLMAgentConfig(enable_plan=False, enable_verify=False)
        agent = LLMAgent(backend=mock_backend, config=config)
        result = await agent.run("task", task_id="t3")
        assert result.plan == []
        assert result.verification_passed is True  # Default when verify disabled

    @pytest.mark.asyncio
    async def test_reset_stats(self, mock_backend):
        agent = LLMAgent(backend=mock_backend)
        await agent.run("task 1")
        assert mock_backend.call_count >= 3  # plan, retrieve, verify, respond

        agent.reset()
        assert mock_backend.call_count == 0

    @pytest.mark.asyncio
    async def test_to_dict(self, mock_backend):
        agent = LLMAgent(backend=mock_backend)
        result = await agent.run("What is CI/CD?")
        d = result.to_dict()
        assert d["task_id"] is not None
        assert "final_answer" in d
        assert "total_latency_ms" in d
        assert "reliability_trace_length" in d


# ── Agent: Parse helpers ──────────────────────────────────────────────

class TestParsePlan:
    def test_numbered_plan(self):
        text = "1. Analyze the question\n2. Retrieve docs\n3. Formulate answer"
        result = _parse_plan(text)
        assert len(result) == 3
        assert result[0] == "Analyze the question"

    def test_dashed_plan(self):
        text = "- Step one\n- Step two"
        result = _parse_plan(text)
        assert len(result) == 2

    def test_fallback(self):
        text = "Just do it"
        result = _parse_plan(text)
        assert len(result) == 1
        assert "Just do it" in result[0]


class TestParseToolCall:
    def test_valid_tool(self):
        text = "TOOL: calculator\nARGS: {\"expression\": \"2+2\"}"
        result = _parse_tool_call(text)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1] == {"expression": "2+2"}

    def test_no_tools_needed(self):
        text = "NO_TOOLS_NEEDED"
        result = _parse_tool_call(text)
        assert result is None

    def test_no_tool_section(self):
        text = "Just some text"
        result = _parse_tool_call(text)
        assert result is None


class TestExtractCitations:
    def test_extract(self):
        text = "Docker is a runtime [source: chunk0]. Kubernetes [source: chunk1] orchestrates."
        citations = _extract_citations(text)
        assert len(citations) == 2
        assert "[source: chunk0]" in citations

    def test_no_citations(self):
        text = "No citations here."
        assert _extract_citations(text) == []


# ── Integration: LLMAgent with no backend ─────────────────────────────

class TestLLMAgentNoBackend:
    """Tests that LLMAgent handles missing retrieval gracefully."""

    @pytest.mark.asyncio
    async def test_run_without_retrieval(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            backend = OpenAIBackend()
            mock_client = MagicMock()

            def make_resp(content):
                mc = MagicMock()
                mc.message.content = content
                mc.finish_reason = "stop"
                mu = MagicMock()
                mu.prompt_tokens = 5
                mu.completion_tokens = 3
                mu.total_tokens = 8
                mr = MagicMock()
                mr.choices = [mc]
                mr.usage = mu
                return mr

            # 3 calls: plan, verify, respond
            mock_client.chat.completions.create.side_effect = [
                make_resp("1. Plan step"),
                make_resp('{"verification_passed": true, "grounded_claims": [], "ungrounded_claims": [], "notes": "ok"}'),
                make_resp("Final answer."),
            ]
            backend._client = mock_client

            agent = LLMAgent(backend=backend)  # No retrieval_fn
            result = await agent.run("Test task", task_id="t0")
            assert result.retrieved_chunks_count == 0
            assert result.success is True
