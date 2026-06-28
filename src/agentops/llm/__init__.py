"""
AgentOps LLM Integration — pluggable real LLM backends for benchmarks.

Provides multi-provider LLM backends (OpenAI, Anthropic, DeepSeek) and
a real LLM-backed agent that drops into the existing evaluation pipeline.

Usage:
    from agentops.llm import create_backend, LLMAgent

    backend = create_backend()  # Auto-detect from env
    agent = LLMAgent(backend=backend)
    result = await agent.run("What is the capital of France?")
"""

from .backend import (
    AnthropicBackend,
    LLMBackend,
    LLMResponse,
    OpenAIBackend,
    available_backends,
    check_availability,
    create_backend,
)
from .agent import LLMAgent, LLMAgentConfig

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "OpenAIBackend",
    "AnthropicBackend",
    "create_backend",
    "available_backends",
    "check_availability",
    "LLMAgent",
    "LLMAgentConfig",
]
