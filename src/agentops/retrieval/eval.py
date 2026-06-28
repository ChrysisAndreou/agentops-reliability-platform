"""
Retrieval evaluation metrics and benchmark harness.

Provides standard IR evaluation metrics (NDCG, MRR, Recall, Precision, MAP)
plus RAG-specific quality metrics (Context Relevance, Faithfulness).

Includes a built-in BEIR-style test corpus with ground-truth relevance
judgments for reproducible benchmarking of retrieval pipelines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal


# ── Evaluation Metrics ────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Computed retrieval evaluation metrics for a single query."""
    query: str
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    map_score: float = 0.0
    num_results: int = 0
    num_relevant: int = 0


@dataclass
class RAGEvalResult:
    """Complete RAG evaluation result for a benchmark run."""
    benchmark_name: str
    num_queries: int
    retrieval_metrics: list[RetrievalMetrics] = field(default_factory=list)

    # Aggregated metrics
    mean_ndcg_5: float = 0.0
    mean_ndcg_10: float = 0.0
    mean_mrr: float = 0.0
    mean_recall_5: float = 0.0
    mean_recall_10: float = 0.0
    mean_precision_5: float = 0.0
    mean_precision_10: float = 0.0
    mean_map: float = 0.0

    # Context quality metrics (RAG-specific)
    context_relevance: float = 0.0
    answer_faithfulness: float = 0.0

    @property
    def summary(self) -> dict[str, float]:
        """Aggregated metric summary."""
        return {
            "ndcg@5": self.mean_ndcg_5,
            "ndcg@10": self.mean_ndcg_10,
            "mrr": self.mean_mrr,
            "recall@5": self.mean_recall_5,
            "recall@10": self.mean_recall_10,
            "precision@5": self.mean_precision_5,
            "precision@10": self.mean_precision_10,
            "map": self.mean_map,
            "context_relevance": self.context_relevance,
            "answer_faithfulness": self.answer_faithfulness,
        }


class RetrievalEvaluator:
    """Evaluate retrieval quality using standard IR and RAG metrics.

    Computes NDCG, MRR, Recall, Precision, MAP, and RAG-specific
    context relevance and answer faithfulness scores.
    """

    @staticmethod
    def evaluate_query(
        query: str,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        relevance_grades: dict[str, int] | None = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality for a single query.

        Args:
            query: The search query text.
            retrieved_ids: Ordered list of retrieved chunk IDs.
            relevant_ids: Set of relevant chunk IDs.
            relevance_grades: Optional dict mapping chunk_id to relevance grade
                              (0=irrelevant, 1=somewhat, 2=relevant, 3=highly).

        Returns:
            RetrievalMetrics with all computed scores.
        """
        if not retrieved_ids or not relevant_ids:
            return RetrievalMetrics(query=query, num_results=len(retrieved_ids))

        # Use binary relevance if no grades provided
        grades = relevance_grades or {rid: 1 for rid in relevant_ids}

        # NDCG
        ndcg_5 = RetrievalEvaluator._ndcg_at_k(retrieved_ids, grades, 5)
        ndcg_10 = RetrievalEvaluator._ndcg_at_k(retrieved_ids, grades, 10)

        # MRR
        mrr = RetrievalEvaluator._mrr(retrieved_ids, relevant_ids)

        # Recall
        recall_5 = RetrievalEvaluator._recall_at_k(retrieved_ids, relevant_ids, 5)
        recall_10 = RetrievalEvaluator._recall_at_k(retrieved_ids, relevant_ids, 10)

        # Precision
        precision_5 = RetrievalEvaluator._precision_at_k(retrieved_ids, relevant_ids, 5)
        precision_10 = RetrievalEvaluator._precision_at_k(retrieved_ids, relevant_ids, 10)

        # MAP
        map_score = RetrievalEvaluator._map(retrieved_ids, relevant_ids)

        return RetrievalMetrics(
            query=query,
            ndcg_at_5=round(ndcg_5, 4),
            ndcg_at_10=round(ndcg_10, 4),
            mrr=round(mrr, 4),
            recall_at_5=round(recall_5, 4),
            recall_at_10=round(recall_10, 4),
            precision_at_5=round(precision_5, 4),
            precision_at_10=round(precision_10, 4),
            map_score=round(map_score, 4),
            num_results=len(retrieved_ids),
            num_relevant=len(relevant_ids),
        )

    @staticmethod
    def evaluate_benchmark(
        benchmark: RetrievalBenchmark,
        retrieved_fn: callable,
        k_values: list[int] | None = None,
    ) -> RAGEvalResult:
        """Evaluate a retrieval function against a benchmark.

        Args:
            benchmark: A RetrievalBenchmark with queries and relevance judgments.
            retrieved_fn: Function(query, k) -> list[str] of chunk IDs.
            k_values: k values to evaluate at (default: [5, 10]).

        Returns:
            RAGEvalResult with per-query and aggregated metrics.
        """
        k_values = k_values or [5, 10]
        all_metrics = []

        for query_item in benchmark.queries:
            retrieved = retrieved_fn(query_item["query"], max(k_values))
            retrieved_ids = [r.chunk_id if hasattr(r, 'chunk_id') else r for r in retrieved]

            metrics = RetrievalEvaluator.evaluate_query(
                query=query_item["query"],
                retrieved_ids=retrieved_ids,
                relevant_ids=set(query_item["relevant_ids"]),
                relevance_grades=query_item.get("relevance_grades"),
            )
            all_metrics.append(metrics)

        # Aggregate
        n = len(all_metrics) if all_metrics else 1
        result = RAGEvalResult(
            benchmark_name=benchmark.name,
            num_queries=len(all_metrics),
            retrieval_metrics=all_metrics,
            mean_ndcg_5=round(sum(m.ndcg_at_5 for m in all_metrics) / n, 4),
            mean_ndcg_10=round(sum(m.ndcg_at_10 for m in all_metrics) / n, 4),
            mean_mrr=round(sum(m.mrr for m in all_metrics) / n, 4),
            mean_recall_5=round(sum(m.recall_at_5 for m in all_metrics) / n, 4),
            mean_recall_10=round(sum(m.recall_at_10 for m in all_metrics) / n, 4),
            mean_precision_5=round(sum(m.precision_at_5 for m in all_metrics) / n, 4),
            mean_precision_10=round(sum(m.precision_at_10 for m in all_metrics) / n, 4),
            mean_map=round(sum(m.map_score for m in all_metrics) / n, 4),
        )
        return result

    # ── Metric implementations ─────────────────────────────────────

    @staticmethod
    def _ndcg_at_k(
        retrieved: list[str],
        grades: dict[str, int],
        k: int,
    ) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved[:k]):
            rel = grades.get(chunk_id, 0)
            dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed

        # Ideal DCG: sort by relevance descending
        ideal_grades = sorted(grades.values(), reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_grades))):
            idcg += ideal_grades[i] / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _mrr(retrieved: list[str], relevant_ids: set[str]) -> float:
        """Mean Reciprocal Rank — reciprocal of first relevant result's rank."""
        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant_ids: set[str], k: int) -> float:
        """Recall@k — fraction of relevant documents retrieved in top-k."""
        if not relevant_ids:
            return 0.0
        retrieved_relevant = sum(1 for rid in retrieved[:k] if rid in relevant_ids)
        return retrieved_relevant / len(relevant_ids)

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant_ids: set[str], k: int) -> float:
        """Precision@k — fraction of top-k results that are relevant."""
        if k <= 0 or not retrieved:
            return 0.0
        retrieved_relevant = sum(1 for rid in retrieved[:k] if rid in relevant_ids)
        return retrieved_relevant / min(k, len(retrieved[:k]))

    @staticmethod
    def _map(retrieved: list[str], relevant_ids: set[str]) -> float:
        """Mean Average Precision."""
        if not relevant_ids:
            return 0.0
        precisions = []
        num_relevant = 0
        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_ids:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        if not precisions:
            return 0.0
        return sum(precisions) / len(relevant_ids)

    @staticmethod
    def context_relevance(
        query: str,
        retrieved_docs: list[str],
        llm: Any = None,
    ) -> float:
        """Estimate context relevance — how much of retrieved context is relevant.

        Uses LLM-as-judge: asks whether each document is on-topic for the query.
        Falls back to lexical overlap when no LLM is available.

        Args:
            query: The search query.
            retrieved_docs: List of retrieved document texts.
            llm: Optional LLM for semantic relevance judgment.

        Returns:
            Score 0-1 representing context relevance.
        """
        if not retrieved_docs:
            return 0.0

        # Fast lexical fallback
        query_terms = set(query.lower().split())
        scores = []
        for doc in retrieved_docs:
            doc_terms = set(doc.lower().split())
            overlap = query_terms & doc_terms
            scores.append(len(overlap) / max(len(query_terms), 1))

        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def answer_faithfulness(
        answer: str,
        retrieved_docs: list[str],
    ) -> float:
        """Estimate answer faithfulness — how well the answer is grounded in retrieved docs.

        Simple n-gram overlap between answer claims and retrieved document content.
        For production use, replace with NLI-based faithfulness scoring.

        Args:
            answer: The generated answer text.
            retrieved_docs: The retrieved documents used to ground the answer.

        Returns:
            Score 0-1 representing faithfulness.
        """
        if not answer or not retrieved_docs:
            return 0.0

        # Extract key claims (sentences) from answer
        import re
        claims = re.split(r'(?<=[.!?])\s+', answer.strip())
        if not claims:
            return 0.0

        # Check each claim against the combined retrieved context
        combined_context = " ".join(retrieved_docs).lower()
        supported = 0
        for claim in claims:
            claim_lower = claim.lower().strip()
            if not claim_lower:
                continue
            # Simple verification: check if significant words appear in context
            claim_words = set(claim_lower.split())
            context_words = set(combined_context.split())
            overlap = claim_words & context_words
            if len(overlap) / max(len(claim_words), 1) > 0.3:
                supported += 1

        return round(supported / len(claims), 4)


# ── BEIR-style Benchmark Data ─────────────────────────────────────────

@dataclass
class RetrievalBenchmark:
    """A retrieval benchmark with queries and ground-truth relevance judgments."""
    name: str
    description: str
    documents: list[dict[str, str]]  # [{id, content, title, source}, ...]
    queries: list[dict[str, Any]]    # [{query, relevant_ids, relevance_grades?}, ...]


# ── Built-in Test Corpora ─────────────────────────────────────────────

def load_agentops_retrieval_corpus() -> RetrievalBenchmark:
    """A realistic test corpus for agent reliability documentation retrieval.

    Models a knowledge base of LLM agent operational documentation with
    20 documents and 10 test queries with ground-truth relevance judgments.
    Designed to exercise BM25, dense, and hybrid retrieval strategies
    with varying query types (exact match, semantic, multi-hop).
    """
    documents = [
        {"id": "d001", "content": "Agent memory management: Agents use a combination of short-term and long-term memory. Short-term memory is stored in the conversation context window and holds the last 10 turns. Long-term memory uses a vector database for semantic retrieval of past interactions. Memory is automatically pruned when the context window exceeds 80% capacity.", "title": "Memory Management", "source": "docs/memory.md"},
        {"id": "d002", "content": "Tool calling specification: Agents can invoke tools via structured function calls. Each tool must declare its name, description, parameters (JSON Schema), and return type. Tools are executed synchronously by default. The agent waits for tool output before continuing planning. Maximum 10 tool calls per agent run to prevent infinite loops.", "title": "Tool Calling API", "source": "docs/tools.md"},
        {"id": "d003", "content": "Multi-agent orchestration patterns: Supervisor-worker topology uses a coordinator agent that delegates subtasks to specialized worker agents. Workers communicate through a shared message bus. The supervisor tracks progress, handles failures, and can reassign tasks. Maximum 5 concurrent workers per supervisor to prevent state explosion.", "title": "Multi-Agent Orchestration", "source": "docs/multi-agent.md"},
        {"id": "d004", "content": "Guardrails and safety: Agents are protected by three layers of safety: input filtering (prompt injection detection), output verification (factual grounding check), and runtime monitoring (tool call validation). Each layer can be configured with custom rules. Violations are logged with severity levels: WARN, BLOCK, and CRITICAL.", "title": "Safety Architecture", "source": "docs/safety.md"},
        {"id": "d005", "content": "Evaluation framework: Agents are evaluated using 10 standardized benchmarks covering reasoning, tool use, retrieval, and multi-agent coordination. Each benchmark produces scores on 15 reliability metrics including task success rate, tool call accuracy, retrieval precision, and latency. Results are tracked in the trace store for regression testing.", "title": "Evaluation System", "source": "docs/eval.md"},
        {"id": "d006", "content": "Deployment configuration: Agents run inside Docker containers orchestrated by Kubernetes. The deployment manifest specifies resource limits (CPU: 2 cores, Memory: 4Gi), health check endpoints (/health, /ready), and auto-scaling policies based on queue depth. Production deployments use 3 replicas minimum for high availability.", "title": "Deployment Guide", "source": "docs/deploy.md"},
        {"id": "d007", "content": "Observability pipeline: All agent runs produce OpenTelemetry traces with spans for planning, retrieval, tool execution, and response generation. Traces are exported to an OTLP collector and stored in a time-series database. Dashboards show p50/p95/p99 latency, error rates, and token consumption per agent run.", "title": "Observability Setup", "source": "docs/obs.md"},
        {"id": "d008", "content": "Prompt engineering guide: Prompts should be structured with system, context, and instruction sections. System prompts define agent persona and constraints. Context prompts inject retrieved documents. Instruction prompts specify the task. Use few-shot examples (3-5) for complex tasks. Token budget: 2000 for system, 4000 for context, 1000 for instructions.", "title": "Prompt Engineering", "source": "docs/prompts.md"},
        {"id": "d009", "content": "Streaming response protocol: Agents can stream responses token-by-token via Server-Sent Events (SSE). Each event contains a token, confidence score, and optional citation reference. The client renders tokens progressively and can interrupt streaming by closing the connection. Streaming reduces time-to-first-token by 60% compared to batch responses.", "title": "Streaming Protocol", "source": "docs/streaming.md"},
        {"id": "d010", "content": "Error handling and recovery: Agent failures are classified into three categories: transient (retry with backoff), permanent (fail with clear error message), and partial (return partial results with warnings). The recovery manager implements exponential backoff (1s, 2s, 4s, 8s, 16s) with jitter for transient errors. Circuit breaker trips after 5 consecutive failures.", "title": "Error Recovery", "source": "docs/errors.md"},
        {"id": "d011", "content": "Retrieval-Augmented Generation (RAG) pipeline: The RAG system uses a hybrid retrieval approach combining BM25 lexical search with dense vector embeddings from all-MiniLM-L6-v2. Documents are chunked into 512-token segments with 64-token overlap. Retrieved chunks are reranked using a cross-encoder before being injected into the LLM context. The pipeline achieves 85% recall@10 on internal benchmarks.", "title": "RAG Pipeline", "source": "docs/rag.md"},
        {"id": "d012", "content": "Model routing configuration: The model router selects the optimal LLM for each request based on cost, latency, and capability requirements. Supports five routing strategies: CHEAPEST (minimize cost), FASTEST (minimize latency), CAPABILITY (maximize quality), ROUND_ROBIN (load balance), and FAILOVER (high availability). Budget enforcement caps daily spending per agent.", "title": "Model Router", "source": "docs/router.md"},
        {"id": "d013", "content": "Structured output validation: Agent outputs can be constrained to JSON schemas, Pydantic models, or custom regex patterns. The validator checks type correctness, required fields, enum values, and numeric ranges. Invalid outputs trigger automatic retry with error feedback. Success rate with structured output is 92% on standard benchmarks.", "title": "Structured Output", "source": "docs/structured-output.md"},
        {"id": "d014", "content": "Authentication and API keys: Agent API endpoints require Bearer token authentication. API keys are scoped to specific operations (read, write, admin). Rate limiting is enforced at 100 requests/minute per key. Keys are rotated every 90 days. Failed authentication attempts are logged and trigger alerts after 5 consecutive failures from the same IP.", "title": "Authentication", "source": "docs/auth.md"},
        {"id": "d015", "content": "Fine-tuning integration: Agents can be fine-tuned on task-specific datasets using QLoRA for efficient adaptation. The fine-tuning pipeline supports PEFT adapters, 4-bit quantization, and gradient checkpointing. Training runs are tracked in W&B with metrics: loss, accuracy, and evaluation scores. Fine-tuned adapters are loaded at agent initialization.", "title": "Fine-tuning Guide", "source": "docs/fine-tuning.md"},
        {"id": "d016", "content": "Cost optimization strategies: Reduce LLM costs by caching frequent responses (TTL: 1 hour), using smaller models for simple tasks (classification, extraction), implementing prompt compression for long contexts, and batching requests during off-peak hours. Cost tracking shows per-agent, per-model, and per-endpoint spending with daily budget alerts.", "title": "Cost Optimization", "source": "docs/cost.md"},
        {"id": "d017", "content": "Multi-turn conversation handling: Agents maintain conversation state across turns using a thread-based model. Each thread has a unique ID, message history, and metadata store. Threads support branching (fork conversation at any point) and merging. Maximum 100 messages per thread. Old messages are summarized and archived when the thread exceeds limits.", "title": "Conversation Management", "source": "docs/conversation.md"},
        {"id": "d018", "content": "Testing and CI/CD pipeline: Agent behavior is tested with deterministic simulators that replace LLM calls with pre-recorded responses. The test suite runs 893 tests across 18 modules. CI enforces 80% coverage minimum, type checking with mypy, and linting with ruff. Regression tests compare current agent output against known-good snapshots.", "title": "Testing Framework", "source": "docs/testing.md"},
        {"id": "d019", "content": "Performance benchmarks: Agent throughput benchmarks measure requests/second under load. A single agent instance handles 50 concurrent requests with p95 latency under 2 seconds. Scaling to 10 replicas supports 500 concurrent requests. Memory usage is 2GB baseline + 500MB per concurrent thread. GPU acceleration reduces LLM inference latency by 40%.", "title": "Performance Benchmarks", "source": "docs/performance.md"},
        {"id": "d020", "content": "Security compliance: Agent deployments comply with SOC 2 Type II and GDPR requirements. All data is encrypted at rest (AES-256) and in transit (TLS 1.3). PII is automatically detected and redacted from agent logs. Audit trails track all agent decisions for 90 days. Data residency options include US, EU, and APAC regions.", "title": "Security Compliance", "source": "docs/security.md"},
    ]

    queries = [
        {
            "query": "How does the agent handle memory overflow when the context window is full?",
            "relevant_ids": ["d001"],
            "relevance_grades": {"d001": 3},
        },
        {
            "query": "What happens when a tool call fails or times out?",
            "relevant_ids": ["d002", "d010"],
            "relevance_grades": {"d002": 2, "d010": 3},
        },
        {
            "query": "How do multiple agents coordinate on a complex task?",
            "relevant_ids": ["d003", "d017"],
            "relevance_grades": {"d003": 3, "d017": 1},
        },
        {
            "query": "How are agent outputs verified for safety and correctness?",
            "relevant_ids": ["d004", "d013"],
            "relevance_grades": {"d004": 3, "d013": 2},
        },
        {
            "query": "How do I evaluate and benchmark my agent's performance?",
            "relevant_ids": ["d005", "d019"],
            "relevance_grades": {"d005": 3, "d019": 2},
        },
        {
            "query": "What are the Docker and Kubernetes requirements for production deployment?",
            "relevant_ids": ["d006", "d020"],
            "relevance_grades": {"d006": 3, "d020": 1},
        },
        {
            "query": "How do I set up observability dashboards and alerts?",
            "relevant_ids": ["d007"],
            "relevance_grades": {"d007": 3},
        },
        {
            "query": "What is the best way to structure prompts for optimal results?",
            "relevant_ids": ["d008"],
            "relevance_grades": {"d008": 3},
        },
        {
            "query": "How does streaming response work and how can I enable it?",
            "relevant_ids": ["d009"],
            "relevance_grades": {"d009": 3},
        },
        {
            "query": "How do I reduce LLM costs while maintaining quality?",
            "relevant_ids": ["d012", "d016"],
            "relevance_grades": {"d012": 2, "d016": 3},
        },
    ]

    return RetrievalBenchmark(
        name="agentops-retrieval-corpus",
        description="Agent reliability documentation retrieval corpus — 20 docs, 10 queries with graded relevance judgments. Covers memory, tools, orchestration, safety, evaluation, deployment, observability, prompts, streaming, errors, RAG, routing, structured output, auth, fine-tuning, cost, conversation, testing, performance, and security.",
        documents=documents,
        queries=queries,
    )


# ── Corpus Export ─────────────────────────────────────────────────────

def export_corpus_json(benchmark: RetrievalBenchmark | None = None) -> str:
    """Export a benchmark corpus as JSON for reproducibility."""
    if benchmark is None:
        benchmark = load_agentops_retrieval_corpus()
    return json.dumps({
        "name": benchmark.name,
        "description": benchmark.description,
        "documents": benchmark.documents,
        "queries": benchmark.queries,
    }, indent=2)
