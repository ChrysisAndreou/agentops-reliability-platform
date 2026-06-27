# Evaluation Report: llm-judge
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.519
- **Groundedness**: 0.715
- **Verification Pass Rate**: 40.0%
- **Avg Latency**: 3365ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (3): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| je-001 | 0.69 | 7/0 | ✗ | 5026ms | 0.412 |
| je-002 | 0.69 | 8/0 | ✗ | 2379ms | 0.411 |
| je-003 | 0.73 | 6/0 | ✗ | 3002ms | 0.424 |
| je-004 | 0.73 | 8/0 | ✓ | 4698ms | 0.672 |
| je-005 | 0.73 | 8/0 | ✓ | 1720ms | 0.674 |
