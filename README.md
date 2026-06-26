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
- **Systematic evaluation**: benchmark runners with groundedness, citation precision, verification pass rate, and latency metrics.
- **Failure classification**: automatic pattern detection for hallucination, retrieval gaps, tool errors, and verification failures.

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
│   │   ├── benchmarks.py  # Benchmark task definitions (10 tasks)
│   │   └── harness.py     # Evaluation runner + report generator
│   ├── tracing/           # Trace persistence and failure analysis
│   │   ├── store.py       # SQLite trace store
│   │   └── classifier.py  # 8-pattern failure taxonomy
│   ├── api/               # FastAPI server
│   │   └── app.py         # REST endpoints
│   └── cli/               # CLI (Typer)
│       └── main.py        # agentops run/eval/serve/traces/stats/index
├── sample_data/
│   ├── docs/              # CloudDeploy product docs (3 files, 7 chunks)
│   └── tickets/           # 10 realistic support/quality tickets
├── tests/                 # 19 pytest tests
├── docker/                # Dockerfile + docker-compose
└── .github/workflows/     # CI (lint, type-check, test, build)
```

---

## Tradeoffs

- **LLM-dependent verification**: The verifier uses another LLM call, adding latency and cost. This is intentional — a human-in-the-loop equivalent is impractical for every agent query, and the verifier catches hallucinations the executor misses.
- **SQLite, not distributed traces**: SQLite keeps the system zero-infra and portable. For production deployment, swap to OpenTelemetry + a columnar store.
- **Single-agent focus**: The platform targets individual tool-using agents. Multi-agent coordination tracing is a future direction.
- **Synthetic documentation**: The sample data is fictional (CloudDeploy) to avoid licensing issues. Replace with real docs for production use.

---

## Roadmap

- [ ] OpenTelemetry trace export for production observability
- [ ] Multi-agent coordination tracing
- [ ] Streaming verification (partial response checking)
- [ ] Cost and latency budget gates
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
