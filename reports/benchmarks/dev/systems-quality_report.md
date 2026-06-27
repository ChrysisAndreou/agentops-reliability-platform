# Evaluation Report: systems-quality
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.618
- **Groundedness**: 0.715
- **Verification Pass Rate**: 80.0%
- **Avg Latency**: 3859ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (1): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| sq-001 | 0.69 | 7/0 | ✗ | 4570ms | 0.421 |
| sq-002 | 0.67 | 4/0 | ✓ | 4300ms | 0.663 |
| sq-003 | 0.75 | 4/0 | ✓ | 4301ms | 0.671 |
| sq-004 | 0.75 | 7/0 | ✓ | 2337ms | 0.673 |
| sq-005 | 0.71 | 4/0 | ✓ | 3784ms | 0.661 |
