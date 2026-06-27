"""
Supervisor-worker multi-agent topology built on LangGraph.

The graph implements a coordinator pattern where a supervisor agent:
1. Decomposes complex tasks into subtasks
2. Routes subtasks to specialized worker agents
3. Aggregates worker results into a coherent answer
4. Verifies the aggregated result before responding

Each worker runs the existing single-agent reliability graph internally
(plan → retrieve → execute → verify → respond), making them independently
testable and traceable.

Topology:
    decompose → assign → [worker_1, worker_2, ..., worker_n] → aggregate → verify → respond
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .state import (
    DEFAULT_WORKER_ROLES,
    InterAgentMessage,
    MultiAgentState,
    WorkerAssignment,
    WorkerResult,
)

# ── Supervisor prompts ─────────────────────────────────────────────

DECOMPOSER_SYSTEM = """You are a task decomposition specialist. Given a complex task, break it
into 2-4 concrete, independent subtasks that can be solved by specialized AI agents.

Available worker roles:
{worker_descriptions}

Rules:
1. Each subtask should be solvable by a single worker
2. Map each subtask to the most appropriate worker role
3. Subtasks should be independent (no circular dependencies)
4. Include context the worker needs to solve the subtask

Output JSON:
{{
  "rationale": "why you decomposed this way",
  "subtasks": [
    {{
      "worker_role": "role_name",
      "subtask": "clear description of what to do",
      "context": "relevant background information",
      "priority": 0
    }}
  ]
}}"""

AGGREGATOR_SYSTEM = """You synthesize results from multiple specialized agents into one coherent answer.

Task: {task}

Worker results:
{worker_results}

Rules:
1. Resolve any contradictions between workers
2. Combine complementary information
3. Cite which worker contributed each claim
4. Note if any subtask was incomplete or errored

Output JSON:
{{
  "aggregated_answer": "complete synthesized answer",
  "notes": "how results were combined, any conflicts resolved",
  "worker_contributions": {{"worker_role": "what they contributed"}},
  "unresolved_conflicts": ["any remaining disagreements"]
}}"""


# ── Helper functions ─────────────────────────────────────────────────

def _build_chat_model(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    """Build a chat model from a name string."""
    if model_name.startswith("gpt") or model_name.startswith("o"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)


def _gen_id(prefix: str = "") -> str:
    """Generate a short unique ID."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _now_ms() -> float:
    return time.time() * 1000


# ── Simulated supervisor decomposition ──────────────────────────────

def _simulated_decompose(
    task: str, context: str, roles: list
) -> tuple[list, str]:
    """Rule-based task decomposition without LLM.

    Splits tasks by domain keywords and maps to worker roles.
    Used when no LLM API key is available, enabling CI-reproducible
    multi-agent evaluation.
    """
    # Domain keyword → worker role mapping
    DOMAIN_KEYWORDS = {
        "retrieval_specialist": [
            "document", "doc", "find", "search", "retrieve", "information",
            "what is", "how do", "how to", "explain", "describe", "list",
            "define", "summarize", "policy", "password", "2fa", "authentication",
            "encryption", "tls", "ssl", "certificate",
        ],
        "tool_executor": [
            "calculate", "compute", "cost", "price", "budget", "math",
            "expression", "formula", "break-even", "compound", "annual",
            "monthly", "total", "savings", "comparison of cost",
        ],
        "code_analyst": [
            "code", "config", "configure", "debug", "error", "exception",
            "stack trace", "log", "crash", "failed", "failure", "bug",
            "deployment fails", "build fails", "pipeline",
        ],
        "verifier": [
            "verify", "validate", "audit", "check", "compliance",
            "security audit", "review", "assess", "risk", "incident",
        ],
    }

    # Determine which subtasks are present by keyword matching
    task_lower = task.lower()
    matched_roles = {}

    for role_name, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > 0:
            matched_roles[role_name] = score

    # If nothing matched, default to retrieval specialist
    if not matched_roles:
        matched_roles = {"retrieval_specialist": 1}

    # Sort by score descending, take top 4
    sorted_roles = sorted(matched_roles.items(), key=lambda x: -x[1])[:4]

    subtask_templates = {
        "retrieval_specialist": "Find and synthesize relevant documentation for: {task}",
        "tool_executor": "Perform calculations and data analysis for: {task}",
        "code_analyst": "Analyze configuration, logs, and code issues for: {task}",
        "verifier": "Verify compliance, security posture, and risks for: {task}",
    }

    subtasks = []
    for role_name, score in sorted_roles:
        template = subtask_templates.get(role_name, "Analyze: {task}")
        subtasks.append({
            "worker_role": role_name,
            "subtask": template.format(task=task[:200]),
            "context": context,
            "priority": len(subtasks),
        })

    rationale = (
        f"Keyword-based decomposition: detected {len(subtasks)} domain(s) "
        f"({', '.join(r for r, _ in sorted_roles)})"
    )

    return subtasks, rationale


# ── Graph builder ────────────────────────────────────────────────────


def build_multi_agent_graph(
    worker_fn: Callable[[str, str, str], WorkerResult],
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    worker_roles: list | None = None,
) -> StateGraph:
    """Build the supervisor-worker multi-agent graph.

    Args:
        worker_fn: Async function (worker_role, subtask, context) -> WorkerResult.
                   Called for each worker assignment. Should run the appropriate
                   worker subgraph internally.
        model_name: LLM model for supervisor (decomposition + aggregation).
        temperature: LLM temperature.
        worker_roles: Optional custom worker role definitions.

    Returns:
        Compiled LangGraph StateGraph for the multi-agent workflow.
    """
    llm = _build_chat_model(model_name, temperature)
    roles = worker_roles or DEFAULT_WORKER_ROLES

    def _add_coord_trace(state: MultiAgentState, phase: str, detail: str, latency_ms: float) -> dict:
        trace_entry = {
            "phase": phase,
            "detail": detail,
            "latency_ms": round(latency_ms, 1),
            "timestamp": _now_ms(),
        }
        state.get("coordination_trace", []).append(trace_entry)
        return {"coordination_trace": [trace_entry]}

    def _record_message(state: MultiAgentState, msg: InterAgentMessage) -> dict:
        return {"inter_agent_messages": [msg]}

    # ── Node: Decompose ──────────────────────────────────────────

    def decompose_node(state: MultiAgentState) -> dict:
        t0 = time.time()
        task = state["task"]
        context = state.get("task_context", "")

        worker_descriptions = "\n".join(
            f"- {r['name']}: {r['description']} (capabilities: {', '.join(r['capabilities'])})"
            for r in roles
        )

        prompt = DECOMPOSER_SYSTEM.format(worker_descriptions=worker_descriptions)
        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Task to decompose:\n{task}"),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        # Parse decomposition
        try:
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
        except (json.JSONDecodeError, AttributeError):
            parsed = {}

        subtask_list = parsed.get("subtasks", [])
        rationale = parsed.get("rationale", "Default decomposition")

        # Fallback: if parsing failed, create single subtask
        if not subtask_list:
            subtask_list = [{
                "worker_role": roles[0]["name"] if roles else "retrieval_specialist",
                "subtask": task,
                "context": context,
                "priority": 0,
            }]
            rationale = "Fallback: single-worker execution"

        # Create assignments
        assignments: list[WorkerAssignment] = []
        for i, st in enumerate(subtask_list):
            assignments.append(WorkerAssignment(
                assignment_id=_gen_id("asgn-"),
                worker_role=st.get("worker_role", roles[0]["name"]),
                subtask=st.get("subtask", f"Subtask {i+1}"),
                context=st.get("context", ""),
                priority=st.get("priority", i),
            ))

        latency = (time.time() - t0) * 1000
        coord_trace = {
            "coordination_trace": [{
                "phase": "decompose",
                "detail": f"Decomposed into {len(assignments)} subtask(s): {rationale[:200]}",
                "latency_ms": round(latency, 1),
                "timestamp": _now_ms(),
            }],
        }

        # Record supervisor→worker assignment messages
        msgs = []
        for a in assignments:
            msgs.append(InterAgentMessage(
                msg_id=_gen_id("msg-"),
                from_agent="supervisor",
                to_agent=a["worker_role"],
                msg_type="assignment",
                content=a["subtask"],
                timestamp_ms=_now_ms(),
                metadata={"assignment_id": a["assignment_id"], "priority": a["priority"]},
            ))

        return {
            **coord_trace,
            "inter_agent_messages": msgs,
            "subtasks": [a["subtask"] for a in assignments],
            "decomposition_rationale": rationale,
            "assignments": assignments,
            "current_phase": "execute",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Execute Workers ────────────────────────────────────

    def execute_workers_node(state: MultiAgentState) -> dict:
        t0 = time.time()
        assignments = state.get("assignments", [])

        if not assignments:
            return {
                "current_phase": "aggregate",
                "error": "No assignments to execute",
                "step_count": state.get("step_count", 0) + 1,
            }

        results: list[WorkerResult] = []
        msgs: list[InterAgentMessage] = []
        coord_entries: list[dict] = []

        for assignment in assignments:
            wt0 = time.time()
            try:
                result = worker_fn(
                    assignment["worker_role"],
                    assignment["subtask"],
                    assignment.get("context", ""),
                )
            except Exception as e:
                result = WorkerResult(
                    assignment_id=assignment["assignment_id"],
                    worker_role=assignment["worker_role"],
                    subtask=assignment["subtask"],
                    answer="",
                    grounded_claims=[],
                    ungrounded_claims=[],
                    citations_used=[],
                    verification_passed=False,
                    tool_calls_count=0,
                    retrieved_chunks_count=0,
                    latency_ms=(time.time() - wt0) * 1000,
                    error=str(e),
                )

            results.append(result)

            # Record worker→supervisor result message
            msgs.append(InterAgentMessage(
                msg_id=_gen_id("msg-"),
                from_agent=assignment["worker_role"],
                to_agent="supervisor",
                msg_type="result",
                content=result.get("answer", "")[:500],
                timestamp_ms=_now_ms(),
                metadata={
                    "assignment_id": assignment["assignment_id"],
                    "verification_passed": result.get("verification_passed", False),
                    "error": result.get("error"),
                },
            ))

            coord_entries.append({
                "phase": "worker_execution",
                "detail": (
                    f"Worker '{assignment['worker_role']}' completed '{assignment['subtask'][:100]}': "
                    f"verified={result.get('verification_passed')}, "
                    f"tool_calls={result.get('tool_calls_count', 0)}, "
                    f"latency={result.get('latency_ms', 0):.0f}ms"
                ),
                "latency_ms": round(result.get("latency_ms", 0), 1),
                "timestamp": _now_ms(),
                "worker_role": assignment["worker_role"],
            })

        (time.time() - t0) * 1000

        return {
            "worker_results": results,
            "inter_agent_messages": msgs,
            "coordination_trace": coord_entries,
            "current_phase": "aggregate",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Aggregate ──────────────────────────────────────────

    def aggregate_node(state: MultiAgentState) -> dict:
        t0 = time.time()
        task = state["task"]
        results = state.get("worker_results", [])

        # Format worker results for the aggregator
        results_text = "\n\n".join(
            f"Worker [{r['worker_role']}]:\n"
            f"  Subtask: {r['subtask']}\n"
            f"  Answer: {r.get('answer', 'No answer')[:500]}\n"
            f"  Verified: {r.get('verification_passed', False)}\n"
            f"  Grounded claims: {len(r.get('grounded_claims', []))}\n"
            f"  Error: {r.get('error') or 'None'}"
            for r in results
        )

        prompt = AGGREGATOR_SYSTEM.format(task=task, worker_results=results_text)

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Synthesize the worker results into one answer."),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        try:
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
        except (json.JSONDecodeError, AttributeError):
            parsed = {}

        aggregated = parsed.get("aggregated_answer", content)
        notes = parsed.get("notes", "")

        latency = (time.time() - t0) * 1000

        return {
            "aggregated_answer": aggregated,
            "aggregation_notes": notes,
            "coordination_trace": [{
                "phase": "aggregate",
                "detail": f"Aggregated {len(results)} worker result(s): {notes[:200]}",
                "latency_ms": round(latency, 1),
                "timestamp": _now_ms(),
            }],
            "current_phase": "verify",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Verify ─────────────────────────────────────────────

    def verify_node(state: MultiAgentState) -> dict:
        t0 = time.time()
        state["task"]
        aggregated = state.get("aggregated_answer", "")

        # Collect all grounded claims from workers
        all_grounded = []
        all_ungrounded = []
        all_citations = []
        for r in state.get("worker_results", []):
            all_grounded.extend(r.get("grounded_claims", []))
            all_ungrounded.extend(r.get("ungrounded_claims", []))
            all_citations.extend(r.get("citations_used", []))

        # Verification passes if all workers passed
        all_verified = all(
            r.get("verification_passed", False)
            for r in state.get("worker_results", [])
        ) if state.get("worker_results") else False

        latency = (time.time() - t0) * 1000

        return {
            "verification_passed": all_verified,
            "verification_notes": f"All {len(state.get('worker_results', []))} workers passed verification" if all_verified else "Some workers failed verification",
            "grounded_claims": all_grounded,
            "ungrounded_claims": all_ungrounded,
            "citations_used": list(set(all_citations)),
            "final_answer": aggregated,
            "coordination_trace": [{
                "phase": "verify",
                "detail": f"Verification: {'PASSED' if all_verified else 'FAILED'} "
                          f"(grounded={len(all_grounded)}, ungrounded={len(all_ungrounded)})",
                "latency_ms": round(latency, 1),
                "timestamp": _now_ms(),
            }],
            "current_phase": "respond",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Respond ────────────────────────────────────────────

    def respond_node(state: MultiAgentState) -> dict:
        return {
            "done": True,
            "current_phase": "done",
            "step_count": state.get("step_count", 0) + 1,
            "coordination_trace": [{
                "phase": "respond",
                "detail": "Multi-agent workflow complete",
                "latency_ms": 0,
                "timestamp": _now_ms(),
            }],
        }

    # ── Build graph ──────────────────────────────────────────────

    graph = StateGraph(MultiAgentState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("execute_workers", execute_workers_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("decompose")

    graph.add_edge("decompose", "execute_workers")
    graph.add_edge("execute_workers", "aggregate")
    graph.add_edge("aggregate", "verify")
    graph.add_edge("verify", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=MemorySaver())


# ── Simulated multi-agent graph (no LLM needed) ─────────────────────


def build_simulated_multi_agent_graph(
    worker_fn: Callable[[str, str, str], Any],
    worker_roles: list | None = None,
) -> StateGraph:
    """Build a multi-agent graph using keyword-based decomposition.

    No LLM is required — the supervisor uses rule-based keyword matching
    for task decomposition and concatenates worker answers for aggregation.
    This enables fully CI-reproducible multi-agent evaluation without API keys.

    Args:
        worker_fn: Function (role, subtask, context) -> WorkerResult.
        worker_roles: Optional custom worker role definitions.

    Returns:
        Compiled LangGraph StateGraph.
    """
    roles = worker_roles or DEFAULT_WORKER_ROLES

    # ── Node: Decompose (simulated) ──────────────────────────────

    def decompose_node(state: MultiAgentState) -> dict:
        t0 = time.time()
        task = state["task"]
        context = state.get("task_context", "")

        subtask_list, rationale = _simulated_decompose(task, context, roles)

        assignments: list[WorkerAssignment] = []
        for i, st in enumerate(subtask_list):
            assignments.append(WorkerAssignment(
                assignment_id=_gen_id("asgn-"),
                worker_role=st.get("worker_role", roles[0]["name"]),
                subtask=st.get("subtask", f"Subtask {i+1}"),
                context=st.get("context", ""),
                priority=st.get("priority", i),
            ))

        latency = (time.time() - t0) * 1000
        msgs = []
        for a in assignments:
            msgs.append(InterAgentMessage(
                msg_id=_gen_id("msg-"),
                from_agent="supervisor",
                to_agent=a["worker_role"],
                msg_type="assignment",
                content=a["subtask"],
                timestamp_ms=_now_ms(),
                metadata={"assignment_id": a["assignment_id"], "priority": a["priority"]},
            ))

        return {
            "coordination_trace": [{
                "phase": "decompose",
                "detail": f"Simulated decomposition: {rationale}",
                "latency_ms": round(latency, 1),
                "timestamp": _now_ms(),
            }],
            "inter_agent_messages": msgs,
            "subtasks": [a["subtask"] for a in assignments],
            "decomposition_rationale": rationale,
            "assignments": assignments,
            "current_phase": "execute",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Execute Workers ────────────────────────────────────

    def execute_workers_node(state: MultiAgentState) -> dict:
        time.time()
        assignments = state.get("assignments", [])

        if not assignments:
            return {
                "current_phase": "aggregate",
                "error": "No assignments to execute",
                "step_count": state.get("step_count", 0) + 1,
            }

        results: list[WorkerResult] = []
        msgs: list[InterAgentMessage] = []
        coord_entries: list[dict] = []

        for assignment in assignments:
            wt0 = time.time()
            try:
                result = worker_fn(
                    assignment["worker_role"],
                    assignment["subtask"],
                    assignment.get("context", ""),
                )
            except Exception as e:
                result = WorkerResult(
                    assignment_id=assignment["assignment_id"],
                    worker_role=assignment["worker_role"],
                    subtask=assignment["subtask"],
                    answer="",
                    grounded_claims=[],
                    ungrounded_claims=[],
                    citations_used=[],
                    verification_passed=False,
                    tool_calls_count=0,
                    retrieved_chunks_count=0,
                    latency_ms=(time.time() - wt0) * 1000,
                    error=str(e),
                )

            results.append(result)
            msgs.append(InterAgentMessage(
                msg_id=_gen_id("msg-"),
                from_agent=assignment["worker_role"],
                to_agent="supervisor",
                msg_type="result",
                content=result.get("answer", "")[:500],
                timestamp_ms=_now_ms(),
                metadata={
                    "assignment_id": assignment["assignment_id"],
                    "verification_passed": result.get("verification_passed", False),
                    "error": result.get("error"),
                },
            ))
            coord_entries.append({
                "phase": "worker_execution",
                "detail": (
                    f"Worker '{assignment['worker_role']}': "
                    f"verified={result.get('verification_passed')}, "
                    f"latency={result.get('latency_ms', 0):.0f}ms"
                ),
                "latency_ms": round(result.get("latency_ms", 0), 1),
                "timestamp": _now_ms(),
                "worker_role": assignment["worker_role"],
            })

        return {
            "worker_results": results,
            "inter_agent_messages": msgs,
            "coordination_trace": coord_entries,
            "current_phase": "aggregate",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Aggregate (simulated) ──────────────────────────────

    def aggregate_node(state: MultiAgentState) -> dict:
        t0 = time.time()
        results = state.get("worker_results", [])

        # Simple concatenation of worker answers
        parts = []
        for r in results:
            role = r.get("worker_role", "unknown")
            answer = r.get("answer", "")
            if answer:
                parts.append(f"[{role}]\n{answer}")

        aggregated = "\n\n".join(parts) if parts else "No results from workers."

        latency = (time.time() - t0) * 1000

        return {
            "aggregated_answer": aggregated,
            "aggregation_notes": f"Concatenated {len(parts)} worker result(s)",
            "coordination_trace": [{
                "phase": "aggregate",
                "detail": f"Simulated aggregation: {len(parts)} worker result(s) combined",
                "latency_ms": round(latency, 1),
                "timestamp": _now_ms(),
            }],
            "current_phase": "verify",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Verify ─────────────────────────────────────────────

    def verify_node(state: MultiAgentState) -> dict:
        t0 = time.time()

        all_grounded = []
        all_ungrounded = []
        all_citations = []
        for r in state.get("worker_results", []):
            all_grounded.extend(r.get("grounded_claims", []))
            all_ungrounded.extend(r.get("ungrounded_claims", []))
            all_citations.extend(r.get("citations_used", []))

        all_verified = all(
            r.get("verification_passed", False)
            for r in state.get("worker_results", [])
        ) if state.get("worker_results") else False

        latency = (time.time() - t0) * 1000

        return {
            "verification_passed": all_verified,
            "verification_notes": (
                f"All {len(state.get('worker_results', []))} workers passed verification"
                if all_verified else "Some workers failed verification"
            ),
            "grounded_claims": all_grounded,
            "ungrounded_claims": all_ungrounded,
            "citations_used": list(set(all_citations)),
            "final_answer": state.get("aggregated_answer", ""),
            "coordination_trace": [{
                "phase": "verify",
                "detail": f"Verification: {'PASSED' if all_verified else 'FAILED'}",
                "latency_ms": round(latency, 1),
                "timestamp": _now_ms(),
            }],
            "current_phase": "respond",
            "step_count": state.get("step_count", 0) + 1,
        }

    # ── Node: Respond ────────────────────────────────────────────

    def respond_node(state: MultiAgentState) -> dict:
        return {
            "done": True,
            "current_phase": "done",
            "step_count": state.get("step_count", 0) + 1,
            "coordination_trace": [{
                "phase": "respond",
                "detail": "Multi-agent workflow complete (simulated supervisor)",
                "latency_ms": 0,
                "timestamp": _now_ms(),
            }],
        }

    # ── Build graph ──────────────────────────────────────────────

    graph = StateGraph(MultiAgentState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("execute_workers", execute_workers_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "execute_workers")
    graph.add_edge("execute_workers", "aggregate")
    graph.add_edge("aggregate", "verify")
    graph.add_edge("verify", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=MemorySaver())
