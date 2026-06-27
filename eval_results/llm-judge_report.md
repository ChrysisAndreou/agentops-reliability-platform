# Evaluation Report: llm-judge
Model: sim-production | Tasks: 5 | 2026-06-27 01:24:12

## Summary
- **Composite Score**: 0.666
- **Groundedness**: 0.853
- **Verification Pass Rate**: 80.0%
- **Avg Latency**: 2310ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (1): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| je-001 | 0.85 | 8/0 | ✓ | 3093ms | 0.710 |
| je-002 | 0.87 | 10/0 | ✓ | 2122ms | 0.715 |
| je-003 | 0.82 | 7/0 | ✗ | 2726ms | 0.443 |
| je-004 | 0.87 | 10/0 | ✓ | 1884ms | 0.733 |
| je-005 | 0.87 | 10/0 | ✓ | 1723ms | 0.731 |
