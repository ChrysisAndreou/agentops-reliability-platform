# AgentOps Reliability Platform — Roadmap

## Completed (17/18)

- [x] v0.1 — Core agent orchestration (LangGraph pipeline: plan → retrieve → execute → verify → respond)
- [x] v0.2 — Hybrid retrieval (BM25 + dense embeddings) with citation tracking
- [x] v0.3 — Typed tool registry with schema validation and replay
- [x] v0.4 — SQLite trace persistence with 8-pattern failure classifier
- [x] v0.5 — Systematic evaluation harness with reliability metrics (5 benchmarks)
- [x] v0.6 — AI safety guardrails (3-dimensional: prompt injection, content moderation, tool misuse)
- [x] v0.7 — Multi-agent coordination (supervisor-worker topology, 4 specialized roles)
- [x] v0.8 — Regression testing framework (versioned baselines, cross-benchmark detection)
- [x] v0.9 — LLM-as-Judge evaluation + prompt management & optimization
- [x] v0.10 — Live observability dashboard (FastAPI + WebSocket + Chart.js dark-themed UI)
- [x] v0.11 — Structured output & function calling evaluation (JSON Schema validator, hallucinated tool detection)
- [x] v0.12 — Agent memory evaluation (5 benchmarks, 4 profiles, hallucination detection)
- [x] v0.13 — Production alerting (11 rules, 4 profiles, 3 channels, cooldown)
- [x] v0.14 — SDK / Client library (@trace decorator, context manager, zero-dependency HTTP client)
- [x] v0.15 — Streaming verification (4 strategies: STRICT, THRESHOLD, LENIENT, ACCUMULATING)
- [x] v0.16 — Alerting integrations (Slack Block Kit, Discord embeds, SMTP email, zero deps)
- [x] v0.17 — OpenTelemetry observability (OTLP span export, 7 production metrics)

## Remaining

- [x] v0.18 — Production Readiness Assessment (8-dimension composite score, CI gate, 5 scenarios)
- [x] v0.19 — Pluggable LLM backends (OpenAI, Anthropic, DeepSeek) with RealLLMAgent for non-simulated benchmarks
