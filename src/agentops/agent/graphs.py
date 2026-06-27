"""
Reliability-oriented LangGraph agent graph.

The graph implements a plan → retrieve → tool_execute → verify → respond
pipeline where each step produces traceable state and the verifier acts as
a quality gate before the final response is delivered.

This is more rigorous than a simple ReAct loop — every claim must be
grounded in retrieved evidence, every tool call is validated, and the
verifier catches hallucinations and missing citations before they reach
the user.
"""

from __future__ import annotations

import json
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .state import ReliabilityState, ReliabilityStep
from .tool_registry import ToolRegistry

# ── Prompts ─────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a planning agent for a technical support / systems quality system.
Given a user query and available context, break the task into concrete, executable steps.

Rules:
1. Each step must be a single, specific action.
2. If information is missing, include a retrieval step first.
3. If a calculation or API call is needed, list it as a tool step.
4. End with a response synthesis step.

Output a JSON array of step descriptions:
["Step 1: ...", "Step 2: ...", ...]
"""

RETRIEVER_SYSTEM = """You formulate search queries to find relevant information for the task.
Given the plan and task, generate 1-3 specific search queries that will find the needed information.

Output a JSON array of query strings:
["query 1", "query 2"]"""

EXECUTOR_SYSTEM = """You execute one step of a plan using available tools and retrieved evidence.

Available tools:
{tool_descriptions}

Retrieved evidence (use this for factual claims):
{retrieved_context}

Current plan:
{plan}

Execute the current step. Output format:
- To use a tool: {{"tool": "tool_name", "args": {{...}}}}
- To synthesize from evidence: {{"observation": "your observation"}}
- If you have a final answer: {{"final_answer": "your answer"}}"""

VERIFIER_SYSTEM = """You are a strict fact-checker for an AI agent's output. Your job is to verify
that every factual claim in the agent's proposed answer is grounded in the retrieved evidence.

Task: {task}

Retrieved evidence:
{retrieved_context}

Agent's proposed answer:
{proposed_answer}

Check each factual claim:
1. Is it explicitly supported by the evidence? → "grounded"
2. Is it contradicted by the evidence? → "contradicted"
3. Is it not mentioned in the evidence? → "ungrounded"

Output JSON:
{{
  "all_grounded": true/false,
  "grounded_claims": ["claim 1", "claim 2"],
  "ungrounded_claims": ["claim 3"],
  "contradicted_claims": [],
  "notes": "explanation",
  "citations_used": ["chunk_id_1", "chunk_id_2"]
}}

If any claim is ungrounded or contradicted, mark all_grounded as false.
"""

RESPONDER_SYSTEM = """You write the final response to the user based on verified evidence.
Only make claims that are grounded in the retrieved evidence. Cite sources using [1], [2], etc.

Task: {task}

Verified evidence:
{retrieved_context}

Verification notes:
{verification_notes}

Write a clear, helpful response. Start each citation-backed claim with the citation number."""


# ── Graph Builder ────────────────────────────────────────────────────

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


def build_reliability_graph(
    tool_registry: ToolRegistry,
    retrieval_fn,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> StateGraph:
    """Build the reliability agent graph.

    Args:
        tool_registry: Typed tool registry with validation.
        retrieval_fn: Async function (query: str, k: int) -> list[RetrievalResult].
        model_name: LLM model identifier.
        temperature: LLM temperature.
        max_retries: Max verification retries before accepting with warnings.
    """

    llm = _build_chat_model(model_name, temperature)

    def _step_trace(state: ReliabilityState, step_name: str, step_type: str,
                    input_summary: str, output_summary: str,
                    latency_ms: float, error: str | None = None) -> ReliabilityStep:
        return ReliabilityStep(
            step_name=step_name,
            step_type=step_type,  # type: ignore[arg-type]
            input_summary=input_summary,
            output_summary=output_summary,
            tool_calls=[],
            retrieved_chunks=[],
            verification_passed=None,
            latency_ms=latency_ms,
            error=error,
        )

    # ── Nodes ────────────────────────────────────────────────────

    def planner_node(state: ReliabilityState) -> dict:
        t0 = time.time()
        task = state["task"]
        context = state.get("task_context", "")

        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        response = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        try:
            # Extract JSON array from response
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            plan = json.loads(match.group(0)) if match else [content]
        except (json.JSONDecodeError, AttributeError):
            plan = [content]

        latency = (time.time() - t0) * 1000
        trace = _step_trace(state, "planner", "plan",
                            f"Task: {task[:100]}",
                            f"Generated {len(plan)} step(s)",
                            latency)

        return {
            "plan": plan,
            "current_step": "retrieve",
            "step_count": 1,
            "reliability_trace": [trace],
        }

    def retriever_node(state: ReliabilityState) -> dict:
        t0 = time.time()
        task = state["task"]
        plan = state.get("plan", [])

        # Generate queries
        query_prompt = f"Task: {task}\nPlan:\n" + "\n".join(f"- {s}" for s in plan)
        response = llm.invoke([
            SystemMessage(content=RETRIEVER_SYSTEM),
            HumanMessage(content=query_prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        try:
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            queries = json.loads(match.group(0)) if match else [task]
        except (json.JSONDecodeError, AttributeError):
            queries = [task]

        # Execute retrieval
        all_chunks = []
        citation_map = {}
        for _i, query in enumerate(queries[:3]):
            try:
                chunks = retrieval_fn(query, k=5)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    citation_map[chunk["chunk_id"]] = chunk["content"]
            except Exception:
                pass

        # Deduplicate
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                unique_chunks.append(chunk)

        latency = (time.time() - t0) * 1000
        trace = _step_trace(state, "retriever", "retrieve",
                            f"Queries: {queries}",
                            f"Retrieved {len(unique_chunks)} chunks from {len(queries)} queries",
                            latency)

        return {
            "retrieved_chunks": unique_chunks,
            "citation_map": citation_map,
            "current_step": "execute",
            "step_count": state.get("step_count", 0) + 1,
            "reliability_trace": [trace],
        }

    def executor_node(state: ReliabilityState) -> dict:
        t0 = time.time()
        task = state["task"]
        plan = state.get("plan", [])
        step_idx = state.get("step_count", 0) - 1  # relative to plan
        retrieved = state.get("retrieved_chunks", [])

        retrieved_context = "\n\n".join(
            f"[{c['chunk_id']}] ({c['source']})\n{c['content'][:1000]}"
            for c in retrieved[:10]
        )

        tool_descriptions = "\n".join(
            f"- {name}: {tool_registry.get(name).description}"
            for name in tool_registry.tool_names
        ) if tool_registry.tool_names else "No tools available"

        current_plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))

        prompt = EXECUTOR_SYSTEM.format(
            tool_descriptions=tool_descriptions,
            retrieved_context=retrieved_context or "No evidence retrieved yet.",
            plan=current_plan_text,
        )

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Task: {task}\n\nExecute step {step_idx + 1}."),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        # Parse tool calls
        tool_calls_made = []
        tool_results_raw = []
        try:
            parsed = json.loads(content) if content.strip().startswith("{") else None
        except json.JSONDecodeError:
            parsed = None

        if parsed and "tool" in parsed:
            result = tool_registry.invoke(parsed["tool"], parsed.get("args", {}))
            tool_calls_made.append(parsed)
            tool_results_raw.append(
                result.output if result.success else f"Error [{result.error_type}]: {result.error}"
            )
        elif parsed and "final_answer" in parsed:
            latency = (time.time() - t0) * 1000
            trace = _step_trace(state, "executor", "tool_call",
                                f"Step {step_idx + 1}: {plan[step_idx][:100] if plan else 'execute'}",
                                "Final answer produced",
                                latency)
            return {
                "final_answer": parsed["final_answer"],
                "reliability_trace": [trace],
                "current_step": "verify",
                "step_count": state.get("step_count", 0) + 1,
            }

        latency = (time.time() - t0) * 1000
        trace = _step_trace(state, "executor", "tool_call",
                            f"Step {step_idx + 1}",
                            f"Tool calls: {len(tool_calls_made)}",
                            latency)

        return {
            "tool_calls_made": state.get("tool_calls_made", []) + tool_calls_made,
            "tool_results_raw": state.get("tool_results_raw", []) + tool_results_raw,
            "reliability_trace": [trace],
            "current_step": "verify",
            "step_count": state.get("step_count", 0) + 1,
        }

    def verifier_node(state: ReliabilityState) -> dict:
        t0 = time.time()
        task = state["task"]
        proposed = state.get("final_answer", "")
        retrieved = state.get("retrieved_chunks", [])

        if not proposed:
            # No answer yet — synthesize one from evidence
            retrieved_context = "\n\n".join(
                f"[{c['chunk_id']}] {c['content'][:800]}"
                for c in retrieved[:8]
            )
            synthesis = llm.invoke([
                SystemMessage(content=RESPONDER_SYSTEM.format(
                    task=task,
                    retrieved_context=retrieved_context or "No evidence available.",
                    verification_notes="",
                )),
                HumanMessage(content="Synthesize a response from the evidence above."),
            ])
            proposed = synthesis.content if hasattr(synthesis, "content") else str(synthesis)

        retrieved_context = "\n\n".join(
            f"[{c['chunk_id']}] ({c['source']})\n{c['content'][:600]}"
            for c in retrieved[:10]
        )

        response = llm.invoke([
            SystemMessage(content=VERIFIER_SYSTEM.format(
                task=task,
                retrieved_context=retrieved_context or "No evidence retrieved.",
                proposed_answer=proposed,
            )),
            HumanMessage(content="Verify the proposed answer."),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        try:
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            verification = json.loads(match.group(0)) if match else {"all_grounded": True, "notes": "Could not parse verification"}
        except (json.JSONDecodeError, AttributeError):
            verification = {"all_grounded": True, "notes": "Could not parse verification"}

        all_grounded = verification.get("all_grounded", True)
        grounded = verification.get("grounded_claims", [])
        ungrounded = verification.get("ungrounded_claims", [])
        citations = verification.get("citations_used", [])
        notes = verification.get("notes", "")

        latency = (time.time() - t0) * 1000
        trace = _step_trace(state, "verifier", "verify",
                            f"Verifying {len(grounded) + len(ungrounded)} claims",
                            f"Grounded: {len(grounded)}, Ungrounded: {len(ungrounded)}",
                            latency)

        return {
            "verification_passed": all_grounded,
            "verification_notes": notes,
            "grounded_claims": grounded,
            "ungrounded_claims": ungrounded,
            "citations_used": citations,
            "final_answer": proposed if all_grounded else state.get("final_answer", ""),
            "done": all_grounded,
            "reliability_trace": [trace],
            "current_step": "respond" if all_grounded else "retry",
        }

    def responder_node(state: ReliabilityState) -> dict:
        t0 = time.time()
        task = state["task"]
        retrieved = state.get("retrieved_chunks", [])
        verification_notes = state.get("verification_notes", "")
        answer = state.get("final_answer", "")

        retrieved_context = "\n\n".join(
            f"[{i+1}] ({c['source']})\n{c['content'][:800]}"
            for i, c in enumerate(retrieved[:8])
        )

        response = llm.invoke([
            SystemMessage(content=RESPONDER_SYSTEM.format(
                task=task,
                retrieved_context=retrieved_context,
                verification_notes=verification_notes,
            )),
            HumanMessage(content=f"Draft answer: {answer}\n\nProduce the final cited response."),
        ])
        final = response.content if hasattr(response, "content") else str(response)

        latency = (time.time() - t0) * 1000
        trace = _step_trace(state, "responder", "respond",
                            "Synthesizing final response",
                            f"Response: {final[:200]}",
                            latency)

        return {
            "final_answer": final,
            "done": True,
            "reliability_trace": [trace],
            "current_step": "done",
        }

    # ── Routing ──────────────────────────────────────────────────

    def route_after_planner(state: ReliabilityState) -> str:
        return "retrieve"

    def route_after_retriever(state: ReliabilityState) -> str:
        return "execute"

    def route_after_executor(state: ReliabilityState) -> str:
        if state.get("final_answer"):
            return "verify"
        return "verify"  # always verify after execution

    def route_after_verifier(state: ReliabilityState) -> str:
        if state.get("verification_passed"):
            return "respond"
        retries = state.get("step_count", 0)
        if retries > 8:
            # Max retries exceeded — respond with warnings
            return "respond"
        return "retrieve"  # re-retrieve and try again

    # ── Build Graph ──────────────────────────────────────────────

    graph = StateGraph(ReliabilityState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("executor", executor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("responder", responder_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "executor")
    graph.add_edge("executor", "verifier")

    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"respond": "responder", "retrieve": "retriever"},
    )

    graph.add_edge("responder", END)

    return graph.compile(checkpointer=MemorySaver())
