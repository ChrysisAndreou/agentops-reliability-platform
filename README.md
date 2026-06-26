# AgentOps Reliability Platform

**Production-oriented observability, tracing, and evaluation for tool-using AI agents.**

[![CI](https://github.com/ChrysisAndreou/agentops-reliability-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrysisAndreou/agentops-reliability-platform/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What Problem Does This Solve?

LLM agents that use tools and retrieval are powerful but unreliable in production. They hallucinate, miss citations, call tools incorrectly, and produce unverifiable claims. Without systematic observability, tracing, and evaluation, teams ship agents that fail silently.

**AgentOps Reliability Platform** provides a complete reliability control plane:

- **Reliability-first agent workflow**: plan → retrieve → execute → verify → respond, with a verification gate that catches hallucinations before they reach users.
- **Hybrid retrieval with citations**: BM25 + dense vector search with per-claim citation tracking.
- **Persistent trace store**: SQLite-backed trace persistence with query, replay, and failure analysis.
- **Systematic evaluation**: 7 benchmarks (35 tasks) with groundedness, citation precision, verification pass rate, and latency metrics. Simulated agent backend enables demo/eval without API keys.
- **Comparative evaluation**: A/B testing between agent configurations with statistical significance detection and regression monitoring.
- **Multi-agent coordination**: Supervisor-worker topology with inter-agent message tracing, coordination metrics, and a 5-task multi-agent benchmark.
- **Failure classification**: automatic pattern detection for 8 failure modes.
- **Cost & latency budget gates**: configurable per-run and per-step cost/latency limits with graceful degradation.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Agent Graph (LangGraph)               │
│  ┌────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐   │
│  │Planner │→ │Retriever │→ │Executor│→ │Verifier  │   │
│  └────────┘  └──────────┘  └────────┘  └────┬─────┘   │
│                                       ┌─────┴──────┐   │
│                              ┌────────┤✓ Grounded  │   │
│                              │        │  Responder │   │
│                              │        └────────────┘   │
│                              │                         │
│                              │        ┌────────────┐   │
│                              └────────┤✗ Ungrounded│   │
│                                       │  Re-retrieve│  │
│                                       └────────────┘   │
└────────────────────────────────────────────────────────┘
         │                    │                   │
         ▼                    ▼                   ▼
┌─────────────┐  ┌────────────────────┐  ┌─────────────────┐
│  Retrieval  │  │   Tool Registry    │  │   Trace Store   │
│  (BM25 +    │  │  (typed, validated,│  │   (SQLite)       │
│   Dense)    │  │   replayable)      │  │                 │
└─────────────┘  └────────────────────┘  └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Failure Analysis│
                                              │ + Eval Reports  │
                                              └─────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Verification gate before response** | Catches hallucinations before the user sees them — unlike post-hoc evaluation |
| **Hybrid retrieval (BM25 + dense)** | Lexical match for precise queries, semantic for conceptual ones; avoids pure-dense blind spots |
| **SQLite trace store** | Zero-infra persistence; queryable by verification status, task, and failure mode |
| **Typed tool registry with replay** | Schema validation prevents bad tool calls; replay enables deterministic evaluation |
| **Failure pattern taxonomy** | 8 classified failure modes for systematic improvement, not just "did it work?" |
| **OpenTelemetry (optional)** | Production-grade observability via OTLP spans + metrics; disabled by default for zero overhead |

---

## Quick Start

### Installation

```bash
git clone https://github.com/ChrysisAndreou/agentops-reliability-platform.git
cd agentops-reliability-platform
pip install -e ".[dev]"
```

### Run the Agent on a Task

```bash
agentops run "How do I enable two-factor authentication on CloudDeploy?"
```

Uses the sample CloudDeploy documentation as the retrieval corpus. The agent will plan, retrieve relevant docs, verify claims, and produce a cited response.

### Run Evaluation Benchmarks

```bash
agentops eval --benchmark support-tickets --output eval_results/
```

Runs 5 support-ticket resolution tasks and produces:
- `eval_results/support-tickets_report.md` — human-readable report
- `eval_results/support-tickets_report.json` — structured metrics
- `eval_results/traces.db` — full trace database for replay

### Inspect Traces

```bash
agentops traces           # list recent runs
agentops trace st-001     # inspect a specific trace
agentops stats            # aggregate statistics
```

### Start the API Server

```bash
agentops serve --port 8000
```

Endpoints:
- `GET /health` — health check
- `POST /api/run` — run an agent on a task
- `GET /api/traces` — list traces
- `GET /api/traces/{id}` — trace detail
- `GET /api/traces/{id}/replay` — replay data
- `GET /api/evals` — evaluation runs
- `GET /api/stats` — aggregate statistics

### Docker

```bash
docker compose -f docker/docker-compose.yml up
```

This starts both the AgentOps API server and a Jaeger collector for OpenTelemetry traces and metrics. Access the Jaeger UI at http://localhost:16686.

### OpenTelemetry Observability

AgentOps exports agent execution traces and metrics to any OTLP-compatible collector (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc.).

**Enable OTEL export:**

```bash
# Via CLI flag
agentops run "How do I enable 2FA?" --otel

# Via environment variable
export AGENTOPS_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
agentops serve --otel
```

**What gets exported:**

| Signal | Contents |
|--------|----------|
| **Traces** | Parent span per agent run, child spans per step (plan, retrieve, execute, verify, respond). Span attributes carry task, verification status, grounded/ungrounded claims, tool counts, and latency. |
| **Metrics** | `agentops.runs.total` (counter), `agentops.runs.verification_pass` (counter), `agentops.runs.failed` (counter), `agentops.latency_ms` (histogram), `agentops.step.latency_ms` (histogram), `agentops.retrieved_chunks` (histogram), `agentops.tool_calls` (histogram) |

**Programmatic usage:**

```python
from agentops.tracing import OTelObserver
from agentops.agent.implementations import ReliabilityAgent

observer = OTelObserver(service_name="my-agentops")
observer.start()

agent = ReliabilityAgent(
    tool_registry=registry,
    retrieval_fn=my_retrieval,
    otel_observer=observer,
)
result = await agent.run("What is the deployment strategy?")
# Traces + metrics automatically exported to collector

observer.shutdown()
```

OTEL is **disabled by default** — no overhead, no collector dependency. When `AGENTOPS_OTEL_ENABLED` is not set or `--otel` is not passed, the observer is a zero-cost no-op.

### Kubernetes

Production-grade K8s deployment with HPA, TLS ingress, network policies, and Terraform support.

```bash
# Apply with Kustomize
kubectl apply -k k8s/

# Or provision with Terraform
cd terraform && terraform init && terraform apply
```

See [`k8s/README.md`](k8s/README.md) for full deployment guide, architecture diagram, and troubleshooting.

---

## Metrics Measured

| Metric | Description | Range |
|--------|-------------|-------|
| **Groundedness** | Fraction of claims grounded in retrieved evidence | 0–1 |
| **Citation Precision** | Citations per available chunk | 0–1 |
| **Verification Pass Rate** | Binary: did the verifier accept the output? | 0 or 1 |
| **Tool Success Rate** | Fraction of tool calls that succeeded | 0–1 |
| **Answer Completeness** | Key terms present in the final answer | 0–1 |
| **Latency Score** | Normalized latency (linear decay from 0 to 120s) | 0–1 |
| **Composite** | Weighted aggregate of all above | 0–1 |

**Composite weights**: groundedness (0.30), citation precision (0.20), verification (0.25), tool success (0.10), latency (0.10), completeness (0.05). Verification is weighted heavily because an unverified answer is a production risk regardless of other metrics.

---

## Failure Modes Detected

The failure classifier automatically identifies 8 failure patterns:

| Pattern | Severity | Description |
|---------|----------|-------------|
| Hallucination | Critical | Claims not grounded in retrieved evidence |
| Verification Failure | High | Verifier explicitly rejected the output |
| Retrieval Gap | High | No relevant chunks retrieved |
| Timeout/Abort | High | Agent run didn't complete |
| Tool Error | Medium | A tool call failed during execution |
| Planning Failure | Medium | Empty or nonsensical plan |
| No Citations | Medium | Response provided without evidence citations |
| Low Retrieval Quality | Medium | Chunks retrieved but relevance was poor |

---

## Project Structure

```
agentops-reliability-platform/
├── src/agentops/
│   ├── agent/             # LangGraph graphs, tool registry, agent impls
│   │   ├── state.py       # Typed state schemas (ReliabilityState)
│   │   ├── graphs.py      # plan→retrieve→execute→verify→respond graph
│   │   ├── tool_registry.py  # Typed tools with validation and replay
│   │   └── implementations.py # ReliabilityAgent class
│   ├── retrieval/         # BM25 + dense hybrid retrieval
│   │   ├── ingest.py      # Document chunking pipeline
│   │   └── engine.py      # Hybrid search engine
│   ├── evals/             # Evaluation framework
│   │   ├── metrics.py     # Groundedness, citation, verification metrics
│   │   ├── benchmarks.py  # 6 benchmark suites (30 tasks: retrieval, tool-use, multi-step, edge-cases, hallucination)
│   │   ├── harness.py     # Evaluation runner + report generator
│   │   ├── simulator.py   # Configurable simulated agent (4 profiles, deterministic, no API keys)
│   │   ├── comparator.py  # A/B testing, regression detection, multi-profile comparison
│   │   └── budget.py      # Cost and latency budget gates with graceful enforcement
│   ├── multi_agent/        # Multi-agent coordination (v0.5)\n│   │   ├── state.py        # MultiAgentState, WorkerAssignment, InterAgentMessage\n│   │   ├── topology.py     # Supervisor-worker LangGraph topology\n│   │   └── coordinator.py  # Orchestrator + trace store extensions\n│   ├── tracing/            # Trace persistence and failure analysis
│   │   ├── store.py       # SQLite trace store
│   │   ├── classifier.py  # 8-pattern failure taxonomy
│   │   └── opentelemetry.py  # OTLP span/metric export (optional)
│   ├── api/               # FastAPI server
│   │   └── app.py         # REST endpoints
│   └── cli/               # CLI (Typer)
│       └── main.py        # agentops run/eval/serve/traces/stats/index
├── sample_data/
│   ├── docs/              # CloudDeploy product docs (3 files, 7 chunks)
│   └── tickets/           # 10 realistic support/quality tickets
├── tests/                 # 89 pytest tests (core, OTEL, simulator, comparator, budget gates)
├── docker/                # Dockerfile + docker-compose
├── k8s/                   # Kubernetes manifests (Deployment, HPA, Ingress, etc.)
├── terraform/             # Terraform module for GKE/EKS/AKS provisioning
└── .github/workflows/     # CI (lint, type-check, test, build)
```

---

## Tradeoffs

- **LLM-dependent verification**: The verifier uses another LLM call, adding latency and cost. This is intentional — a human-in-the-loop equivalent is impractical for every agent query, and the verifier catches hallucinations the executor misses.
- **SQLite, not distributed traces**: SQLite keeps the system zero-infra and portable. For production deployment, swap to OpenTelemetry + a columnar store.
- **Single-agent + multi-agent**: The platform supports both individual tool-using agents and supervisor-worker multi-agent coordination with full tracing.
- **Synthetic documentation**: The sample data is fictional (CloudDeploy) to avoid licensing issues. Replace with real docs for production use.

---

### Evaluation Benchmarks & Simulator (v0.4)

Run the full evaluation pipeline without API keys using the configurable simulated agent:

```bash
# List all 6 benchmarks
agentops benchmarks

# Run all benchmarks with production-quality simulated agent
agentops simulate --profile production --output eval_results/

# Run a specific benchmark
agentops simulate --benchmark tool-use --profile development

# A/B compare two agent configurations
agentops compare --benchmark tool-use --profile-a production --profile-b development
```

**4 simulation profiles** (perfect, production, development, unreliable) with tunable groundedness, verification rate, hallucination rate, and latency. All runs are deterministic by task ID — same input produces same result.

```python
from agentops.evals.simulator import SimulatedAgent, PRODUCTION_AGENT
from agentops.evals.harness import EvalHarness

agent = SimulatedAgent(config=PRODUCTION_AGENT, seed=42)
harness = EvalHarness(agent=agent, model="sim-production")
report = await harness.run_benchmark(benchmark)
print(report.to_markdown())
```

**35 evaluation tasks** across 7 benchmarks:
- `support-tickets` (5 tasks) — Resolve CloudDeploy support tickets
- `systems-quality` (5 tasks) — Evaluate reliability/quality characteristics
- `tool-use` (5 tasks) — Test tool-calling correctness and error handling
- `multi-step` (5 tasks) — Chain multiple retrieval and reasoning steps
- `edge-cases` (5 tasks) — Robustness against ambiguous/adversarial inputs
- `hallucination-resistance` (5 tasks) — Test tendency to fabricate when docs are silent
- `multi-agent` (5 tasks) — Supervisor-worker coordination, task decomposition, and result aggregation

### Multi-Agent Coordination (v0.5)

Supervisor-worker topology for complex tasks requiring decomposition across specialized agents:

```bash
# Run a complex task with multi-agent decomposition
agentops run-multi "Diagnose three simultaneous deployment failures and produce an incident response plan"

# Evaluate multi-agent coordination benchmarks
agentops eval-multi --benchmark multi-agent --profile production

# A/B compare coordinator profiles
agentops eval-multi --profile production --output eval_results/
```

**Worker roles**: retrieval specialist, tool executor, code analyst, verifier — each runs the existing reliability graph independently. The supervisor decomposes, routes, aggregates, and verifies across workers. Full inter-agent message tracing and coordination metrics.

```python
from agentops.multi_agent import MultiAgentCoordinator, MultiAgentConfig

worker_fn = MultiAgentCoordinator.make_simulated_worker_fn("production")
config = MultiAgentConfig(model="gpt-4o")
coordinator = MultiAgentCoordinator(worker_fn=worker_fn, config=config)

result = await coordinator.run(
    "Analyze security posture, compute deployment costs, and recommend architecture"
)
print(f"Workers: {result.worker_count}, Verified: {result.verification_passed}")
```

### Budget Gates

```python
from agentops.evals.budget import BudgetState, BudgetGate, NORMAL_BUDGET

state = BudgetState(cost_budget=NORMAL_BUDGET)
state.start()

for step in workflow:
    state.record_step(input_tokens=500, output_tokens=200, latency_ms=5000)
    result = gate.check(state)
    if not result.allowed:
        raise BudgetExceededError(result.reason)
```

---

## Roadmap

- [x] Kubernetes Deployment — production-grade manifests with HPA, TLS, network policies, Terraform
- [x] OpenTelemetry trace export for production observability — OTLP spans + metrics to Jaeger/Grafana/Honeycomb
- [x] Evaluation benchmark suite — 6 benchmarks, 30 tasks, simulated agent, comparative A/B testing, regression detection
- [x] Cost and latency budget gates — configurable per-run and per-step budgets with graceful enforcement
- [x] Multi-agent coordination tracing — supervisor-worker topology, inter-agent message tracing, coordination metrics, 5-task benchmark
- [ ] Streaming verification (partial response checking)
- [ ] Web dashboard for trace exploration
- [ ] Integration tests with local LLM (Ollama) for CI reproducibility

---

## Contributing

```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
