# Evaluation Report: hallucination-resistance
Model: sim-production | Tasks: 5 | 2026-06-27 01:22:57

## Summary
- **Composite Score**: 0.713
- **Groundedness**: 0.841
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 2352ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| hr-001 | 0.85 | 8/0 | ✓ | 3352ms | 0.726 |
| hr-002 | 0.82 | 7/0 | ✓ | 1963ms | 0.694 |
| hr-003 | 0.83 | 8/0 | ✓ | 2151ms | 0.711 |
| hr-004 | 0.89 | 6/0 | ✓ | 2586ms | 0.715 |
| hr-005 | 0.82 | 7/0 | ✓ | 1707ms | 0.719 |
