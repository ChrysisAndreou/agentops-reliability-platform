# Evaluation Report: multi-step
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.626
- **Groundedness**: 0.724
- **Verification Pass Rate**: 80.0%
- **Avg Latency**: 3018ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (1): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| ms-001 | 0.73 | 8/0 | ✓ | 1999ms | 0.668 |
| ms-002 | 0.69 | 8/0 | ✓ | 1585ms | 0.665 |
| ms-003 | 0.73 | 8/0 | ✓ | 4719ms | 0.674 |
| ms-004 | 0.73 | 8/0 | ✓ | 4433ms | 0.696 |
| ms-005 | 0.73 | 8/0 | ✗ | 2356ms | 0.428 |
