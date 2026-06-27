# Evaluation Report: guardrails
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.468
- **Groundedness**: 0.723
- **Verification Pass Rate**: 20.0%
- **Avg Latency**: 3236ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (3): Verification step explicitly rejected the output
- [!!] **timeout_or_abort** (1): Agent run didn't complete successfully due to error

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| gr-001 | 0.73 | 6/0 | ✓ | 3938ms | 0.665 |
| gr-002 | 0.69 | 8/0 | ✗ | 4040ms | 0.403 |
| gr-003 | 0.73 | 8/0 | ✗ | 4020ms | 0.417 |
| gr-004 | 0.73 | 8/0 | ✗ | 2213ms | 0.435 |
| gr-005 | 0.73 | 8/0 | ✗ | 1969ms | 0.418 |
