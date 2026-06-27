# AgentOps: A Production Readiness Framework for Tool-Using AI Agents

**Chrysis Andreou**  
MSc Artificial Intelligence, University of Cyprus  
June 2026

---

## Abstract

Deploying tool-using AI agents in production requires systematic evaluation beyond
single-turn benchmarks. Current evaluation frameworks focus on model capabilities in
isolation, but production agent systems face compound failure modes across
orchestration, retrieval, tool execution, verification, and multi-agent coordination.
We present AgentOps, an open-source reliability platform that defines, measures, and
enforces production readiness for LangGraph-based tool-using agents. AgentOps provides
a 10-benchmark evaluation suite (50 tasks), an 8-dimension weighted scoring engine, a
4-tier readiness classification system with CI gate enforcement, and full-stack
observability including OpenTelemetry trace export, live dashboards, and multi-channel
alerting. The platform ships with a deterministic simulated-agent backend enabling
CI-reproducible evaluation without API keys, Docker Compose one-command setup, and
production-grade Kubernetes/Terraform deployment manifests. We present the
architecture, evaluation methodology, benchmark results across 3 simulated model
profiles, and discuss how the framework addresses the emerging "AI Evaluation
Engineer" role identified in job market analysis of 260+ AI/agent engineering
positions.

---

## 1. Introduction

The rapid adoption of language-model-based agents in production systems has outpaced
the development of evaluation infrastructure. While benchmarks like MMLU, GSM8K, and
HumanEval measure isolated model capabilities, they fail to capture the compound
failure modes that emerge when agents compose retrieval, tool execution, planning, and
verification in multi-step workflows. A model that achieves 90% on a reasoning
benchmark may still hallucinate tool calls, retrieve irrelevant documents, or produce
ungrounded claims when deployed in an agent loop.

Recent job market analysis of 260+ AI and agent engineering roles reveals a distinct
emerging role: the **AI Evaluation Engineer**. Companies including Mistral AI, Cohere,
and LangChain are hiring dedicated engineers to "define what 'ready for production'
means" for AI agents, build evaluation infrastructure, and design comprehensive
benchmarking frameworks. This validates the need for systematic agent evaluation
tooling.

AgentOps addresses this gap by providing:

1. **A production-grade agent pipeline** with LangGraph orchestration, hybrid
   retrieval, typed tool execution, and verification gating.
2. **A comprehensive evaluation framework** spanning 10 benchmark suites, 15
   evaluation metrics, and cross-model comparison.
3. **A production readiness scoring engine** that synthesizes evaluation results
   into an 8-dimension composite score with tiered deployment recommendations.
4. **Full-stack observability** including OpenTelemetry trace export, real-time
   dashboards, and multi-channel production alerting.
5. **CI-integrated quality gates** enforcing deployment decisions with structured
   exit codes.

The platform is implemented in ~14,000 lines of Python (50 modules), backed by 791
passing pytest tests, and deployed via Docker Compose with Kubernetes and Terraform
production manifests.

---

## 2. System Architecture

### 2.1 Agent Execution Pipeline

The core agent runs as a LangGraph StateGraph with five nodes forming a sequential
pipeline through typed state transitions:

```
Planner → Retriever → Tool Executor → Verifier → Response
```

**Planner.** Decomposes user queries into retrieval plans and tool execution sequences.
Maintains a typed state object tracking the plan, retrieval context, tool results, and
verification decisions.

**Retriever.** Implements hybrid search combining BM25 lexical matching with dense
sentence-transformer embeddings. Documents are chunked with configurable overlap and
indexed in-memory. Retrieval results carry citation metadata (document ID, chunk
offset, relevance score) forwarded to the verifier for groundedness checking.

**Tool Executor.** Maintains a typed tool registry with Pydantic schema validation.
Each tool call is validated against its schema before execution. Failed validations
produce structured errors with categorization (type mismatch, missing required field,
hallucinated tool name, etc.). All tool outputs are cached for deterministic replay.

**Verifier.** The verification gate inspects agent outputs before they reach the user.
It cross-references claims against retrieved evidence, checks citation accuracy, and
flags unsupported assertions. This is the primary hallucination defense — claims that
fail verification are blocked and the agent is prompted to revise.

**Response.** Formats the verified output as structured JSON with inline citations.
The response node is the final quality gate; if verification fails, the state machine
loops back to the Planner for revision.

### 2.2 Observability Stack

The observability layer captures every agent step for debugging, monitoring, and
evaluation:

**Trace Store (SQLite).** Records per-step traces including: node type, input/output
state, tool calls with schema validation results, retrieved documents with citation
metadata, verification decisions, latency, and token cost estimates. An 8-pattern
failure classifier (HALLUCINATED_TOOL, SCHEMA_MISMATCH, RETRIEVAL_MISS,
VERIFICATION_FAIL, TOOL_TIMEOUT, PARSE_ERROR, GROUNDING_FAIL, UNKNOWN) categorizes
every failure for pattern analysis.

**OpenTelemetry Integration.** Agent runs are exported as OTLP spans with per-step
child spans. Seven production metrics (run count, success/failure rate, latency
histograms, verification pass rate, groundedness score, tool call count, token usage)
are emitted to any OTLP-compatible collector (Jaeger, Grafana, Honeycomb).

**Alerting Engine.** Eleven threshold-based alert rules monitor verification quality,
hallucination rate, groundedness, latency, failure rate, tool reliability, and
multi-dimensional degradation. Four configurable sensitivity profiles (strict,
production, permissive, silent) with cooldown-based alert storm prevention. Three
output channels: Slack Block Kit, Discord embeds, and SMTP email — all zero external
dependencies.

**Live Dashboard.** WebSocket-powered HTML dashboard with Chart.js visualizations
showing real-time trace streams, failure pattern breakdowns, latency distributions,
and verification scores. Dark-themed professional UI, 32 dedicated tests.

### 2.3 SDK and Developer Experience

A zero-dependency agent instrumentation SDK provides `@agentops.trace()` decorators
for sync/async functions, `agentops.start_run()` context managers, and helper
functions for logging tool calls, retrieval operations, and verification results. The
stdlib-only HTTP client submits traces to the AgentOps server without requiring
langchain or other heavy dependencies.

### 2.4 Infrastructure

- **Docker Compose** with Jaeger collector included — one-command setup.
- **Kubernetes** manifests: Deployment, Service, HPA, Ingress, NetworkPolicy, PDB with
  Kustomize.
- **Terraform** cloud-agnostic module for GKE/EKS/AKS provisioning.

---

## 3. Evaluation Methodology

### 3.1 Benchmark Suite

AgentOps ships with 10 benchmark suites covering 50 tasks across 6 capability
dimensions:

| # | Benchmark | Dimension | Tasks | Description |
|---|-----------|-----------|-------|-------------|
| 1 | Support Tickets | Retrieval + QA | 5 | Enterprise Q&A over product documentation |
| 2 | Systems Quality | Diagnostic | 5 | Infrastructure troubleshooting scenarios |
| 3 | Tool Use | Tool Reliability | 5 | Multi-tool chains with schema requirements |
| 4 | Multi-Step Reasoning | Planning | 5 | Sequential reasoning with dependencies |
| 5 | Edge Cases | Robustness | 5 | Ambiguous, underspecified, or contradictory inputs |
| 6 | Hallucination Resistance | Safety | 5 | Adversarial prompts with no grounding data |
| 7 | AI Safety Guardrails | Safety | 5 | Prompt injection, content policy, tool misuse |
| 8 | LLM-as-Judge | Quality | 5 | Multi-dimensional quality assessment |
| 9 | Multi-Agent Coordination | Architecture | 5 | Supervisor-worker topology tasks |
| 10 | Prompt Engineering | Optimization | 5 | A/B variant comparison with iterative refinement |

Each benchmark includes ground-truth answers, expected tool sequences, citation
requirements, and scoring rubrics.

### 3.2 Simulated Agent Backend

For CI reproducibility, AgentOps provides a deterministic simulated agent backend that
produces consistent, predictable outputs without API keys. The simulator uses
keyword-based heuristics to emulate agent behavior: it retrieves documents matching
query keywords, executes tool calls from a predefined library, and generates responses
by template-filling retrieved content. While the simulated backend does not reflect
real LLM performance, it enables:

- CI-reproducible evaluation (same inputs → same scores every run)
- Regression detection (any code change that degrades scores is caught)
- Zero-cost benchmarking (no API calls)
- Rapid iteration on evaluation methodology independent of model availability

### 3.3 Evaluation Metrics

Fifteen metrics are computed across benchmarks:

| Metric | Category | Description |
|--------|----------|-------------|
| Verification Pass Rate | Reliability | Fraction of outputs passing verification gate |
| Groundedness Mean | Quality | Average groundedness score across tasks |
| Citation Precision | Retrieval | Fraction of citations verified against source |
| MRR | Retrieval | Mean Reciprocal Rank of correct documents |
| Tool Call Success Rate | Tools | Fraction of tool calls returning valid results |
| Schema Compliance Rate | Tools | Fraction of tool calls with valid argument schemas |
| Hallucinated Tool Rate | Tools | Fraction of calls to non-existent tools |
| Accuracy | Quality | LLM-as-Judge accuracy dimension |
| Completeness | Quality | Coverage of required information |
| Relevance | Quality | Relevance of response to query |
| Clarity | Quality | Readability and structure of response |
| Guardrail Block Rate | Safety | Fraction of unsafe inputs correctly blocked |
| False Negative Rate | Safety | Unsafe inputs that passed guardrails |
| Recall Precision | Memory | Precision of multi-turn memory recall |
| Recall Rate | Memory | Recall rate of multi-turn memory |

### 3.4 Model Comparison Framework

The ModelBenchmark module compares multiple model profiles on identical benchmarks.
Each profile specifies: model name, provider, cost per 1K input/output tokens,
average latency, and behavioral notes. Rankings are computed by mean composite score.
A Pareto frontier identifies models that are non-dominated in the cost-performance
space.

Seven pre-configured model profiles are included (gpt-4o, claude-3-sonnet,
deepseek-v4, gpt-4o-mini, claude-3-haiku, llama-3.1-8b, gemini-1.5-flash).

### 3.5 Regression Testing

Versioned baseline persistence enables cross-version regression detection. Each
baseline stores per-benchmark composite scores. The regression runner compares current
results against a selected baseline and flags degradations exceeding configurable
thresholds. CI-friendly exit codes enforce quality gates: exit 0 (no regressions),
exit 1 (warnings), exit 2 (critical regressions blocking deployment).

---

## 4. Production Readiness Assessment

### 4.1 Dimension Scoring

The Readiness Assessor synthesizes evaluation data into 8 dimension scores, each with
configurable weight, pass/warn thresholds, and evidence:

| Dimension | Weight | Pass ≥ | Warn ≥ | Evidence Sources |
|-----------|--------|--------|--------|-----------------|
| Verification Quality | 20% | 85 | 60 | Verification pass rate, groundedness mean |
| Safety Compliance | 18% | 85 | 70 | Guardrail block rate, false negative rate, active patterns |
| Tool Reliability | 15% | 85 | 60 | Tool success rate, schema compliance, hallucinated tool rate |
| Response Quality | 15% | 80 | 60 | Accuracy, completeness, relevance, clarity |
| Retrieval Quality | 12% | 80 | 60 | Citation precision, relevance score, MRR |
| Latency & Efficiency | 8% | 85 | 60 | Avg latency, P95 latency, budget compliance |
| Memory Consistency | 7% | 80 | 60 | Recall precision, recall rate, F1, hallucination rate |
| Multi-Agent Coordination | 5% | 80 | 60 | Coordination score, message efficiency, task completion |

Each dimension produces a [0-100] score with evidence annotations and targeted
recommendations. The composite score is a weighted average.

### 4.2 Readiness Tiers

| Tier | Condition | Meaning | CI Exit |
|------|-----------|---------|---------|
| PRODUCTION READY | Composite ≥ 90, all PASS | Safe to deploy. Monitor with alerting. | 0 |
| CONDITIONAL | Composite ≥ 75, 1-2 WARN | Deploy with monitoring. Address warnings. | 0 |
| NEEDS WORK | Composite < 75 or 3+ WARN | Fix failures before deployment. | 1 |
| CRITICAL ISSUES | Composite < 50 or any FAIL | DO NOT DEPLOY. Fix immediately. | 2 |

### 4.3 CI Gate Enforcement

The `agentops readiness gate` command runs the full assessment and returns the
corresponding exit code. This integrates directly into GitHub Actions or any CI
pipeline:

```yaml
- name: AgentOps Readiness Gate
  run: agentops readiness gate
```

If the gate fails (exit 1 or 2), the CI pipeline blocks deployment. The `agentops
readiness scenarios` command runs pre-configured assessment scenarios (healthy,
degraded, critical, edge-threshold, multi-dimension-degrade) for testing alert
pipeline behavior.

---

## 5. Benchmark Results

### 5.1 Simulated Agent Performance

The following results are from the deterministic simulated-agent backend (CI profile),
included to demonstrate the evaluation pipeline. These are NOT real LLM benchmark
scores and should not be compared to model leaderboards.

| Benchmark | Composite | Verification Pass Rate | Groundedness | Tasks |
|-----------|-----------|----------------------|--------------|-------|
| Support Tickets | 0.715 | 100% | 0.859 | 5 |
| Systems Quality | 0.614 | 60% | 0.860 | 5 |
| Tool Use | 0.676 | 80% | 0.867 | 5 |
| Multi-Step | 0.673 | 80% | 0.867 | 5 |
| Edge Cases | 0.715 | 100% | 0.855 | 5 |
| Hallucination Resistance | 0.713 | 100% | 0.841 | 5 |

Systems Quality shows the lowest composite (0.614) and verification pass rate (60%),
reflecting the simulated agent's difficulty with infrastructure diagnostic scenarios
requiring multi-step causal reasoning.

### 5.2 Model Comparison (Simulated)

| Model | Composite | Cost (USD) | Latency (ms) | Pareto? |
|-------|-----------|------------|--------------|---------|
| gpt-4o | 0.416 | $0.0018 | 4000 | No |
| claude-3-sonnet | 0.416 | $0.0027 | 3500 | No |
| deepseek-v4 | 0.416 | $0.0004 | 3000 | Yes |

DeepSeek V4 achieves identical composite scores at 78% lower cost than gpt-4o and 85%
lower than Claude 3 Sonnet, placing it on the Pareto frontier. These results reflect
the simulated backend's uniform scoring across models — real LLM evaluation would show
differentiation.

### 5.3 Production Readiness Assessment

Running `agentops readiness assess` with the CI-simulated profile yields:

- **Verdict:** CONDITIONAL — Deploy with Monitoring
- **Composite Score:** 83.6/100
- **7 PASS, 1 WARN (Memory Consistency), 0 FAIL**

The Memory Consistency dimension scores lowest (77.5) due to the simulated agent's
simplified memory model. All other dimensions, including Verification Quality (91.0)
and Safety Compliance (90.5), exceed production thresholds.

### 5.4 Failure Pattern Analysis

The dominant failure patterns across benchmarks are:
- **Tool schema mismatches** (tool-use benchmark): incorrect argument types
- **Retrieval misses** (systems-quality): documents not found for specialized queries
- **Incomplete multi-step chains** (multi-step): agent stops before full sequence

These patterns inform targeted improvements: schema validation hardening, retrieval
index expansion, and multi-step completion enforcement.

---

## 6. Related Work

AgentOps occupies a distinct position in the evaluation ecosystem:

**LM Evaluation Harness (EleutherAI).** Benchmarks model capabilities (MMLU, GSM8K,
etc.) but does not evaluate agent systems — no pipeline orchestration, tool use, or
retrieval evaluation.

**OpenAI Evals.** Framework for writing custom LLM evaluations. Focused on
single-turn model outputs, not compound agent workflows.

**LangSmith (LangChain).** Production monitoring and tracing for LLM applications.
Provides observability but not systematic evaluation frameworks, benchmark suites, or
readiness scoring.

**Arize Phoenix.** Open-source observability for LLM applications with span tracing
and evaluation. Focuses on tracing and drift monitoring rather than comprehensive
multi-dimensional readiness assessment.

**Braintrust.** Evaluation platform with dataset management and experiment tracking.
Commercial SaaS; does not provide agent-specific evaluation or CI gate enforcement.

**Galileo.** LLM observability and evaluation platform. Commercial; focuses on
monitoring rather than pre-deployment readiness assessment.

AgentOps differentiates by: (1) evaluating the entire agent pipeline, not just model
outputs; (2) providing a systematic readiness framework with tiered deployment
decisions; (3) shipping CI-native, deterministic evaluation without API dependencies;
(4) including production infrastructure (K8s, Terraform, Docker) as first-class
concerns.

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

**Simulated Backend Only.** All benchmark results in this report use the deterministic
simulated agent. Real LLM evaluation requires API keys and would produce
differentiated, publishable results. The framework supports real-model evaluation but
the results are not yet collected.

**No Human Evaluation.** The LLM-as-Judge module provides automated quality scoring,
but correlation with human judgment has not been measured. Future work should include
human evaluation studies.

**Single-Agent Focus.** Multi-agent coordination is the lowest-weighted dimension (5%)
and receives a neutral score when not applicable. As multi-agent systems become more
common in production, this dimension should be expanded.

**Static Benchmarks.** The 50 tasks are manually curated and do not evolve
automatically. Future work should explore synthetic benchmark generation to prevent
overfitting.

### 7.2 Future Directions

1. **Real LLM Benchmarking.** Run the full 10-benchmark suite against gpt-4o,
   claude-3-sonnet, deepseek-v4, and open-weight models (Llama 3.1, Qwen 2.5,
   Mistral) to produce publishable comparative results with statistical significance.

2. **Human Correlation Study.** Measure the alignment between automated LLM-as-Judge
   scores and human evaluator ratings on agent outputs.

3. **Dynamic Benchmark Generation.** Use LLMs to generate new benchmark tasks,
   preventing overfitting to the static 50-task set.

4. **Production Deployment Study.** Deploy AgentOps as the reliability layer for a
   real production agent system and measure its impact on incident rates, mean time
   to detection, and deployment confidence.

5. **Multi-Agent Expansion.** Deepen multi-agent evaluation with additional topologies
   (hierarchical, peer-to-peer, auction-based) and coordination metrics.

---

## 8. Conclusion

AgentOps demonstrates that systematic production readiness assessment for AI agents is
feasible with current tooling. The platform provides a unified framework spanning
evaluation, observability, alerting, and deployment enforcement — addressing the gap
between model benchmarks and production agent reliability.

The emergence of dedicated "AI Evaluation Engineer" roles validates the problem space.
AgentOps is positioned as both a practical tool for teams deploying agents and a
research platform for studying agent failure modes at scale.

The platform is open-source (MIT), fully tested (791 pytest tests), and deployable
with a single Docker command. All code, benchmarks, and deployment manifests are
available at: [github.com/ChrysisAndreou/agentops-reliability-platform](https://github.com/ChrysisAndreou/agentops-reliability-platform)

---

## References

1. AgentOps Reliability Platform v0.18. github.com/ChrysisAndreou/agentops-reliability-platform
2. LangGraph: Building Stateful, Multi-Actor Applications with LLMs. langchain.com/langgraph
3. OpenTelemetry: High-Quality, Ubiquitous, and Portable Telemetry. opentelemetry.io
4. EleutherAI LM Evaluation Harness. github.com/EleutherAI/lm-evaluation-harness
5. OpenAI Evals: A Framework for Evaluating LLMs. github.com/openai/evals
6. LangSmith: LLM Application Observability. smith.langchain.com
7. Arize Phoenix: AI Observability & Evaluation. github.com/Arize-AI/phoenix
8. Agency: Agent Architecture Evaluation Framework. github.com/ChrysisAndreou/agency
9. Function Calling Fine-Tuning Benchmark. github.com/ChrysisAndreou/fine-tuning-benchmark
10. Sentinel: Agentic Code & System Quality Guardian. github.com/ChrysisAndreou/sentinel

---

*AgentOps is an open-source project (MIT License). No external funding was received.
The author declares no competing interests. All benchmarks and code are publicly
available for reproduction and verification.*

---

## Appendix: Source Code Metrics

| Metric | Value |
|--------|-------|
| Python Source Modules | 50 |
| Lines of Code | ~14,000 |
| Test Files | 30+ |
| Passing Tests | 791 |
| Test Coverage | CI-tracked |
| Benchmark Suites | 10 |
| Benchmark Tasks | 50 |
| Evaluation Metrics | 15 |
| Readiness Dimensions | 8 |
| Alert Rules | 11 |
| Git History | 18 versions |
| Docker Containers | 3 (API + Dashboard + Jaeger) |
| Kubernetes Objects | 6 |
| Terraform Resources | 12+ |
