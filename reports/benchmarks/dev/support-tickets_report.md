# Evaluation Report: support-tickets
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.623
- **Groundedness**: 0.718
- **Verification Pass Rate**: 80.0%
- **Avg Latency**: 2648ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **timeout_or_abort** (1): Agent run didn't complete successfully due to error

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| st-001 | 0.69 | 7/0 | ✓ | 1944ms | 0.681 |
| st-002 | 0.71 | 4/0 | ✗ | 3044ms | 0.412 |
| st-003 | 0.73 | 8/0 | ✓ | 2981ms | 0.668 |
| st-004 | 0.73 | 8/0 | ✓ | 2378ms | 0.678 |
| st-005 | 0.71 | 4/0 | ✓ | 2891ms | 0.674 |
