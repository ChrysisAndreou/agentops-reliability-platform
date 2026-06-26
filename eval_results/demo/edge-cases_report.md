# Evaluation Report: edge-cases
Model: sim-production | Tasks: 5 | 2026-06-26 15:04:14

## Summary
- **Composite Score**: 0.715
- **Groundedness**: 0.855
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 2494ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| ec-001 | 0.88 | 5/0 | ✓ | 2036ms | 0.721 |
| ec-002 | 0.85 | 8/0 | ✓ | 2817ms | 0.701 |
| ec-003 | 0.87 | 10/0 | ✓ | 2288ms | 0.728 |
| ec-004 | 0.82 | 7/0 | ✓ | 3419ms | 0.718 |
| ec-005 | 0.87 | 10/0 | ✓ | 1912ms | 0.708 |
