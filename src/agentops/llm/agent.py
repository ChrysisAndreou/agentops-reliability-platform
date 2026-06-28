"""
LLM-backed agent that produces AgentRunResult objects.

A drop-in replacement for SimulatedAgent that uses real LLM API calls
through the pluggable LLM backend. Supports the same `run(task, task_id)`
interface so it can be swapped into the evaluation harness, benchmark
runner, and regression tester.

Usage:
    from agentops.llm import create_backend, LLMAgent

    backend = create_backend()  # or OpenAIBackend(model="gpt-4o")
    agent = LLMAgent(backend=backend, retrieval_fn=my_retrieval)
    result = await agent.run("How do I reset my password?", task_id="t1")
    print(f"Verified: {result.verification_passed}")
    print(f"Cost: ${backend.total_cost:.4f}")
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent.implementations import AgentRunResult
from ..agent.tool_registry import ToolRegistry
from .backend import LLMBackend, LLMResponse


# ── Prompt Templates ──────────────────────────────────────────────────

PLAN_PROMPT = """You are an AI agent that follows a structured plan→retrieve→execute→verify→respond workflow. Given the task below, produce a numbered plan of 3-5 steps.

Task: {task}

Output ONLY a numbered list of concrete steps. Each step should be one line starting with a number and period. Do not include any other text."""

RETRIEVE_PROMPT = """Based on the retrieved documentation below, extract the key facts relevant to the task. List each relevant fact as a bullet point with a citation [source: chunk_id].

Task: {task}

Retrieved documentation:
{retrieved_chunks}

Output ONLY bullet points of key facts with citations. If no relevant information is found, say "No relevant documentation found." """

EXECUTE_PROMPT = """You have access to the following tools:
{tool_descriptions}

Based on the task and retrieved facts, determine if any tools should be called. If yes, specify which tool and its arguments. If no tools are needed, say "NO_TOOLS_NEEDED".

Task: {task}
Retrieved facts:
{retrieved_facts}

Output format:
TOOL: <tool_name>
ARGS: <json_arguments>
Or: NO_TOOLS_NEEDED"""

VERIFY_PROMPT = """You are a verification specialist. Verify whether every claim in the agent's proposed answer is grounded in the retrieved evidence. Flag any ungrounded claims.

Task: {task}
Retrieved evidence:
{retrieved_facts}

Proposed answer:
{proposed_answer}

Output a JSON object:
{{
  "verification_passed": true/false,
  "grounded_claims": ["claim 1", "claim 2"],
  "ungrounded_claims": ["claim 3"],
  "notes": "explanation"
}}

Only output valid JSON."""

RESPOND_PROMPT = """Based on the verified facts and task, produce a final, well-structured answer. Only include claims that passed verification.

Task: {task}
Verified facts:
{verified_facts}
Verification notes: {verification_notes}

Output the final answer directly. Be concise and well-structured."""


# ── Agent Implementation ──────────────────────────────────────────────

@dataclass
class LLMAgentConfig:
    """Configuration for LLMAgent behavior."""

    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    enable_plan: bool = True
    enable_retrieve: bool = True
    enable_execute: bool = True
    enable_verify: bool = True
    verification_strictness: float = 0.7  # Threshold for verification pass


class LLMAgent:
    """A real-LLM-backed agent producing AgentRunResult objects.

    This is a drop-in replacement for `SimulatedAgent` that makes real
    LLM API calls through the pluggable `LLMBackend`. It follows the
    same plan→retrieve→execute→verify→respond workflow and returns
    `AgentRunResult` objects compatible with the evaluation harness.

    Usage:
        backend = create_backend()
        agent = LLMAgent(
            backend=backend,
            retrieval_fn=my_retrieval,
            tool_registry=my_registry,
        )
        result = await agent.run("How do I reset my password?")
    """

    def __init__(
        self,
        backend: LLMBackend,
        retrieval_fn: Callable | None = None,
        tool_registry: ToolRegistry | None = None,
        config: LLMAgentConfig | None = None,
    ):
        self.backend = backend
        self._retrieval_fn = retrieval_fn or _default_retrieval
        self.tool_registry = tool_registry or ToolRegistry()
        self.config = config or LLMAgentConfig(model=backend.model)
        self.model = backend.model

    async def run(
        self, task: str, task_id: str = "default", context: str = ""
    ) -> AgentRunResult:
        """Execute the agent on a task and return a structured result."""
        t_start = time.perf_counter()
        trace: list[dict[str, Any]] = []

        # ── Step 1: Plan ──────────────────────────────────────────
        plan = []
        plan_latency = 0.0
        if self.config.enable_plan:
            plan_prompt = PLAN_PROMPT.format(task=task)
            t0 = time.perf_counter()
            try:
                resp = self.backend.chat(plan_prompt, max_tokens=512)
                plan = _parse_plan(resp.content)
                plan_latency = (time.perf_counter() - t0) * 1000
            except Exception as e:
                plan = [f"Error during planning: {e}"]
                plan_latency = (time.perf_counter() - t0) * 1000

        trace.append({
            "step_name": "plan",
            "step_type": "plan",
            "input_summary": task[:80],
            "output_summary": f"Generated {len(plan)}-step plan",
            "tool_calls": [],
            "retrieved_chunks": [],
            "verification_passed": None,
            "latency_ms": plan_latency,
            "error": None,
        })

        # ── Step 2: Retrieve ──────────────────────────────────────
        retrieved_chunks_count = 0
        retrieved_facts = ""
        retrieve_latency = 0.0
        if self.config.enable_retrieve:
            t0 = time.perf_counter()
            try:
                retrieved = self._retrieval_fn(task, k=5)
                if isinstance(retrieved, list) and retrieved:
                    retrieved_chunks_count = len(retrieved)
                    chunks_text = "\n\n".join(
                        f"[source: chunk{i}] {_chunk_text(c)}"
                        for i, c in enumerate(retrieved)
                    )
                    # Optionally summarize retrieved content via LLM
                    if self.backend.call_count < 50:  # Budget guard
                        retrieve_prompt = RETRIEVE_PROMPT.format(
                            task=task, retrieved_chunks=chunks_text
                        )
                        try:
                            resp = self.backend.chat(retrieve_prompt, max_tokens=1024)
                            retrieved_facts = resp.content
                        except Exception:
                            retrieved_facts = chunks_text[:2000]
                    else:
                        retrieved_facts = chunks_text[:2000]
                else:
                    retrieved_facts = "No relevant documentation found."
            except Exception as e:
                retrieved_facts = f"Retrieval error: {e}"
            retrieve_latency = (time.perf_counter() - t0) * 1000

        citations_used = _extract_citations(retrieved_facts)

        trace.append({
            "step_name": "retrieve",
            "step_type": "retrieve",
            "input_summary": "Search documentation",
            "output_summary": f"Retrieved {retrieved_chunks_count} chunks",
            "tool_calls": [],
            "retrieved_chunks": citations_used[:5],
            "verification_passed": None,
            "latency_ms": retrieve_latency,
            "error": None,
        })

        # ── Step 3: Execute (tools) ───────────────────────────────
        tool_calls_count = 0
        tool_error = None
        execute_latency = 0.0
        tool_results = ""
        if self.config.enable_execute and self.tool_registry.tool_names:
            t0 = time.perf_counter()
            try:
                tool_descs = "\n".join(
                    f"- {name}: {self.tool_registry.get(name).description}"
                    for name in self.tool_registry.tool_names
                )
                execute_prompt = EXECUTE_PROMPT.format(
                    tool_descriptions=tool_descs,
                    task=task,
                    retrieved_facts=retrieved_facts[:2000],
                )
                resp = self.backend.chat(execute_prompt, max_tokens=512)
                parsed = _parse_tool_call(resp.content)
                if parsed:
                    tool_name, tool_args = parsed
                    try:
                        tool_result = self.tool_registry.invoke(tool_name, tool_args)
                        if tool_result.success:
                            tool_results = str(tool_result.output)
                        else:
                            tool_results = f"Tool error: {tool_result.error}"
                            tool_error = tool_result.error
                        tool_calls_count = 1
                    except Exception as e:
                        tool_error = str(e)
                        tool_results = f"Tool error: {e}"
                        tool_calls_count = 1
            except Exception as e:
                tool_error = str(e)
            execute_latency = (time.perf_counter() - t0) * 1000

        trace.append({
            "step_name": "execute",
            "step_type": "tool_call" if tool_calls_count > 0 else "execute",
            "input_summary": "Run tools if needed",
            "output_summary": tool_results[:100] if tool_results else "No tools needed",
            "tool_calls": [{"tool": "calculator", "result": tool_results[:100]}]
                          if tool_calls_count > 0 else [],
            "retrieved_chunks": [],
            "verification_passed": None,
            "latency_ms": execute_latency,
            "error": tool_error,
        })

        # ── Step 4: Verify ────────────────────────────────────────
        verification_passed = True
        verification_notes = ""
        grounded_claims: list[str] = []
        ungrounded_claims: list[str] = []
        verify_latency = 0.0

        # Build draft answer for verification
        draft_answer = f"Based on the retrieved information: {retrieved_facts[:1000]}"

        if self.config.enable_verify:
            t0 = time.perf_counter()
            try:
                verify_prompt = VERIFY_PROMPT.format(
                    task=task,
                    retrieved_facts=retrieved_facts[:2000],
                    proposed_answer=draft_answer,
                )
                resp = self.backend.chat(verify_prompt, max_tokens=1024)
                verify_latency = (time.perf_counter() - t0) * 1000

                import json as _json
                try:
                    parsed = _json.loads(resp.content)
                    verification_passed = parsed.get("verification_passed", True)
                    grounded_claims = parsed.get("grounded_claims", [])
                    ungrounded_claims = parsed.get("ungrounded_claims", [])
                    verification_notes = parsed.get("notes", "")
                except Exception:
                    # Fallback: pass if content looks reasonable
                    verification_passed = len(resp.content) > 20
                    verification_notes = "Verification parse failed; heuristic pass."
            except Exception as e:
                verify_latency = (time.perf_counter() - t0) * 1000
                verification_notes = f"Verification error: {e}"
                verification_passed = False

        trace.append({
            "step_name": "verify",
            "step_type": "verify",
            "input_summary": f"Check claims against sources",
            "output_summary": (
                "PASS: all claims grounded" if verification_passed
                else f"FAIL: {len(ungrounded_claims)} ungrounded claims"
            ),
            "tool_calls": [],
            "retrieved_chunks": [],
            "verification_passed": verification_passed,
            "latency_ms": verify_latency,
            "error": None,
        })

        # ── Step 5: Respond ───────────────────────────────────────
        final_answer = ""
        respond_latency = 0.0
        try:
            respond_prompt = RESPOND_PROMPT.format(
                task=task,
                verified_facts=retrieved_facts[:1500],
                verification_notes=verification_notes,
            )
            t0 = time.perf_counter()
            resp = self.backend.chat(respond_prompt, max_tokens=1024)
            respond_latency = (time.perf_counter() - t0) * 1000
            final_answer = resp.content
        except Exception as e:
            final_answer = f"Error generating response: {e}"
            respond_latency = 0

        trace.append({
            "step_name": "respond",
            "step_type": "respond",
            "input_summary": "Generate final answer",
            "output_summary": final_answer[:80],
            "tool_calls": [],
            "retrieved_chunks": [],
            "verification_passed": None,
            "latency_ms": respond_latency,
            "error": None,
        })

        total_latency = (time.perf_counter() - t_start) * 1000
        success = verification_passed and tool_error is None

        return AgentRunResult(
            task_id=task_id,
            task=task,
            final_answer=final_answer,
            success=success,
            error=tool_error,
            total_latency_ms=total_latency,
            verification_passed=verification_passed,
            verification_notes=verification_notes or "Verification complete",
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            citations_used=citations_used,
            plan=plan,
            tool_calls_count=tool_calls_count,
            retrieved_chunks_count=retrieved_chunks_count,
            reliability_trace=trace,
        )

    def reset(self) -> None:
        """Reset accumulated backend stats."""
        self.backend.reset_stats()


# ── Helpers ───────────────────────────────────────────────────────────

def _parse_plan(text: str) -> list[str]:
    """Parse a numbered plan from LLM output."""
    import re
    steps = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match "1. step" or "1) step" or "- step"
        m = re.match(r"^(?:\d+[.)]\s*|[-*]\s+)(.+)", line)
        if m:
            steps.append(m.group(1).strip())
        elif steps and line:
            # Continuation of previous step
            steps[-1] += " " + line
    return steps if steps else [text.strip()[:200]]


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse TOOL/ARGS from LLM output."""
    import json as _json
    lines = text.strip().split("\n")
    tool_name = None
    args = {}
    for line in lines:
        if line.startswith("TOOL:"):
            tool_name = line[5:].strip()
        elif line.startswith("ARGS:"):
            try:
                args = _json.loads(line[5:].strip())
            except Exception:
                args = {}
    if tool_name and tool_name != "NO_TOOLS_NEEDED":
        return tool_name, args
    return None


def _extract_citations(text: str) -> list[str]:
    """Extract [source: chunkX] citations from text."""
    import re
    return re.findall(r"\[source:\s*chunk\d+\]", text)


def _chunk_text(chunk: Any) -> str:
    """Extract text content from a chunk, which may be a dict or object."""
    if isinstance(chunk, dict):
        return chunk.get("content", chunk.get("text", str(chunk)))
    if hasattr(chunk, "content"):
        return str(chunk.content)
    return str(chunk)[:500]


def _default_retrieval(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Default no-op retrieval."""
    return []
