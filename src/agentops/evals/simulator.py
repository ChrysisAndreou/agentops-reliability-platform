"""
Simulated agent backend for demonstration and evaluation without LLM API keys.

Provides deterministic, configurable agent behavior that produces realistic
AgentRunResult objects — enabling the full evaluation pipeline to run and
produce publishable reports without requiring paid LLM API access.

The simulator is designed for:
- CI/CD testing of the evaluation framework
- Demo and documentation generation
- Benchmark development and validation
- Comparative evaluation methodology demonstration

For production use, swap in a real ReliabilityAgent backed by an LLM.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..agent.implementations import AgentRunResult
from ..agent.tool_registry import ToolRegistry


@dataclass
class SimConfig:
    """Configuration for the simulated agent's behavior profile.

    Each parameter controls a dimension of agent quality. Set all to 1.0
    for a "perfect" agent, or tune individually to simulate realistic
    failure modes.
    """

    # Quality targets (0.0 = worst, 1.0 = best)
    groundedness_target: float = 0.85
    verification_pass_rate: float = 0.80
    answer_completeness_target: float = 0.75
    tool_success_rate: float = 0.90

    # Performance
    base_latency_ms: float = 2000
    latency_jitter_ms: float = 800

    # Failure modes (probabilities)
    hallucination_rate: float = 0.05
    tool_error_rate: float = 0.05
    missed_retrieval_rate: float = 0.05

    # Metadata
    name: str = "sim-default"
    description: str = "Default simulation profile"

    def seed_hash(self, task_id: str, salt: str = "") -> int:
        """Deterministic seed from task_id for reproducible results."""
        h = hashlib.sha256(f"{task_id}:{salt}:{self.name}".encode()).digest()
        return int.from_bytes(h[:4], "big")


# ── Pre-built simulation profiles ────────────────────────────────────

PERFECT_AGENT = SimConfig(
    name="perfect",
    description="Ideal agent: 100% grounded, always verified, zero latency jitter",
    groundedness_target=1.0,
    verification_pass_rate=1.0,
    answer_completeness_target=1.0,
    tool_success_rate=1.0,
    base_latency_ms=1000,
    latency_jitter_ms=0,
    hallucination_rate=0.0,
    tool_error_rate=0.0,
    missed_retrieval_rate=0.0,
)

PRODUCTION_AGENT = SimConfig(
    name="production",
    description="Realistic production agent: strong but imperfect",
    groundedness_target=0.90,
    verification_pass_rate=0.85,
    answer_completeness_target=0.80,
    tool_success_rate=0.95,
    base_latency_ms=2500,
    latency_jitter_ms=1000,
    hallucination_rate=0.03,
    tool_error_rate=0.02,
    missed_retrieval_rate=0.02,
)

DEVELOPMENT_AGENT = SimConfig(
    name="development",
    description="Agent under active development: moderate quality, higher variance",
    groundedness_target=0.75,
    verification_pass_rate=0.65,
    answer_completeness_target=0.65,
    tool_success_rate=0.85,
    base_latency_ms=3500,
    latency_jitter_ms=2000,
    hallucination_rate=0.08,
    tool_error_rate=0.05,
    missed_retrieval_rate=0.05,
)

UNRELIABLE_AGENT = SimConfig(
    name="unreliable",
    description="Poorly configured agent: frequent failures, hallucinations",
    groundedness_target=0.50,
    verification_pass_rate=0.35,
    answer_completeness_target=0.40,
    tool_success_rate=0.60,
    base_latency_ms=5000,
    latency_jitter_ms=3000,
    hallucination_rate=0.15,
    tool_error_rate=0.12,
    missed_retrieval_rate=0.10,
)

ALL_PROFILES = [PERFECT_AGENT, PRODUCTION_AGENT, DEVELOPMENT_AGENT, UNRELIABLE_AGENT]


def get_profile(name: str) -> SimConfig | None:
    """Look up a simulation profile by name."""
    for p in ALL_PROFILES:
        if p.name == name:
            return p
    return None


class SimulatedAgent:
    """A drop-in replacement for ReliabilityAgent that produces realistic
    AgentRunResult objects without LLM API calls.

    The simulation is deterministic given the same (task_id, config, seed)
    combination, making it suitable for CI and reproducible benchmarks.

    Usage:
        sim = SimulatedAgent(config=PRODUCTION_AGENT, retrieval_fn=my_retrieval)
        result = await sim.run("How do I reset my password?", task_id="task-1")
        print(f"Verified: {result.verification_passed}")
    """

    def __init__(
        self,
        config: SimConfig = PRODUCTION_AGENT,
        retrieval_fn: Callable | None = None,
        tool_registry: ToolRegistry | None = None,
        model: str = "simulated",
        seed: int | None = None,
    ):
        self.config = config
        self._retrieval_fn = retrieval_fn or _default_retrieval
        self.tool_registry = tool_registry or ToolRegistry()
        self.model = model
        self._seed = seed

    async def run(
        self, task: str, task_id: str = "default", context: str = ""
    ) -> AgentRunResult:
        """Simulate an agent run with deterministic, configurable behavior."""
        rng = random.Random(self.config.seed_hash(task_id, salt=task))

        # Extract key terms from task for simulated grounding
        key_terms = _extract_key_terms(task)

        # Simulate latency
        jitter = rng.uniform(-self.config.latency_jitter_ms, self.config.latency_jitter_ms)
        latency = max(500, self.config.base_latency_ms + jitter)

        # Simulate retrieval
        should_retrieve = rng.random() > self.config.missed_retrieval_rate
        if should_retrieve and self._retrieval_fn:
            try:
                retrieved = self._retrieval_fn(task, k=5)
                retrieved_chunks = len(retrieved) if isinstance(retrieved, list) else 0
            except Exception:
                retrieved_chunks = 0
        else:
            retrieved_chunks = 0

        # Simulate grounded vs ungrounded claims
        n_terms = max(1, len(key_terms))
        n_grounded = max(0, int(n_terms * self.config.groundedness_target))
        n_grounded = min(n_grounded, n_terms)
        rng.shuffle(key_terms)
        grounded_claims = key_terms[:n_grounded]
        ungrounded_claims = key_terms[n_grounded:]

        # Simulate hallucinations (fabricated claims not in key terms)
        if rng.random() < self.config.hallucination_rate:
            hallucination_templates = [
                "Based on internal benchmarks",
                "According to the latest release notes",
                "As documented in the changelog",
                "Per the configuration reference",
            ]
            ungrounded_claims.append(rng.choice(hallucination_templates))

        # Simulate citations
        n_cite = max(0, int(n_grounded * 0.8))
        citations_used = [
            f"chunk:{i}:{hashlib.md5(c.encode()).hexdigest()[:8]}"  # nosec B324 — non-crypto chunk ID
            for i, c in enumerate(grounded_claims[:n_cite])
        ]

        # Simulate tool calls
        has_tool = "calculate" in task.lower() or "compute" in task.lower()
        tool_count = 1 if (has_tool and rng.random() < 0.8) else 0
        tool_error = rng.random() < self.config.tool_error_rate

        # Simulate verification
        verification_pass = rng.random() < self.config.verification_pass_rate

        # Simulate plan
        plan = [
            f"Analyze task: {task[:60]}...",
            "Retrieve relevant documentation",
            "Evaluate retrieved evidence",
            "Formulate grounded response",
        ]
        if has_tool:
            plan.insert(2, "Execute required calculations")

        # Simulate final answer
        completeness = max(0.3, self.config.answer_completeness_target + rng.uniform(-0.1, 0.1))
        n_answer_terms = max(1, int(n_terms * completeness))
        answer_parts = key_terms[:n_answer_terms]
        final_answer = (
            "Based on the available documentation, the key aspects are: "
            + ", ".join(answer_parts)
            + "."
        )

        # Build trace
        trace = [
            {
                "step_name": "plan",
                "step_type": "plan",
                "input_summary": task[:80],
                "output_summary": f"Generated {len(plan)}-step plan",
                "tool_calls": [],
                "retrieved_chunks": [],
                "verification_passed": None,
                "latency_ms": latency * 0.05,
                "error": None,
            },
            {
                "step_name": "retrieve",
                "step_type": "retrieve",
                "input_summary": "Search documentation",
                "output_summary": f"Retrieved {retrieved_chunks} chunks",
                "tool_calls": [],
                "retrieved_chunks": citations_used[:5],
                "verification_passed": None,
                "latency_ms": latency * 0.30,
                "error": None,
            },
        ]

        if tool_count > 0:
            trace.append({
                "step_name": "execute",
                "step_type": "tool_call",
                "input_summary": "Run calculator tool",
                "output_summary": "Error: division by zero" if tool_error else "Result: computed successfully",
                "tool_calls": [{"tool": "calculator", "result": "error" if tool_error else "ok"}],
                "retrieved_chunks": [],
                "verification_passed": None,
                "latency_ms": latency * 0.15,
                "error": "Tool error" if tool_error else None,
            })

        trace.append({
            "step_name": "verify",
            "step_type": "verify",
            "input_summary": f"Check {n_grounded} claims against sources",
            "output_summary": (
                "PASS: all claims grounded" if verification_pass
                else f"FAIL: {len(ungrounded_claims)} ungrounded claims"
            ),
            "tool_calls": [],
            "retrieved_chunks": [],
            "verification_passed": verification_pass,
            "latency_ms": latency * 0.25,
            "error": None,
        })

        trace.append({
            "step_name": "respond",
            "step_type": "respond",
            "input_summary": "Generate final answer",
            "output_summary": final_answer[:80],
            "tool_calls": [],
            "retrieved_chunks": [],
            "verification_passed": None,
            "latency_ms": latency * 0.25,
            "error": None,
        })

        return AgentRunResult(
            task_id=task_id,
            task=task,
            final_answer=final_answer,
            success=verification_pass and not tool_error,
            error="Simulated tool error" if tool_error else None,
            total_latency_ms=latency,
            verification_passed=verification_pass,
            verification_notes=(
                "All claims supported by retrieved evidence"
                if verification_pass
                else f"{len(ungrounded_claims)} claims could not be verified against sources"
            ),
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            citations_used=citations_used,
            plan=plan,
            tool_calls_count=tool_count,
            retrieved_chunks_count=retrieved_chunks,
            reliability_trace=trace,
        )

    def reset(self) -> None:
        """No-op for simulated agent."""
        pass


def _extract_key_terms(text: str) -> list[str]:
    """Extract potential key terms from a task/question string."""
    import re

    # Remove common stopwords and extract meaningful terms
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all",
        "any", "few", "more", "most", "other", "some", "such", "no",
        "only", "own", "same", "than", "too", "very", "just", "what",
        "which", "who", "whom", "this", "that", "these", "those",
        "how", "when", "where", "why", "about", "also", "if", "then",
        "there", "their", "its", "it", "i", "my", "me", "we", "our",
        "you", "your", "he", "she", "they", "them", "him", "her",
    }

    # Extract words, filter stopwords, keep meaningful terms
    words = re.findall(r"[A-Za-z0-9\-\.\+]+", text.lower())
    terms = [w for w in words if w not in stopwords and len(w) > 1]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[:15]  # Cap at 15 terms


def _default_retrieval(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Default no-op retrieval function."""
    return []
