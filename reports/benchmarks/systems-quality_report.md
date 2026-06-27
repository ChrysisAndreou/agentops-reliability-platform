# Evaluation Report: systems-quality
Model: sim-production | Tasks: 5 | 2026-06-27 01:22:57

## Summary
- **Composite Score**: 0.614
- **Groundedness**: 0.860
- **Verification Pass Rate**: 60.0%
- **Avg Latency**: 2426ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (2): Verification step explicitly rejected the output
- [!!] **timeout_or_abort** (1): Agent run didn't complete successfully due to error

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| sq-001 | 0.85 | 8/0 | ✗ | 1849ms | 0.477 |
| sq-002 | 0.89 | 6/0 | ✗ | 1720ms | 0.482 |
| sq-003 | 0.88 | 5/0 | ✓ | 2089ms | 0.711 |
| sq-004 | 0.83 | 8/0 | ✓ | 3147ms | 0.697 |
| sq-005 | 0.86 | 4/0 | ✓ | 3324ms | 0.704 |
