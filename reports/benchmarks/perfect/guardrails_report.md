# Evaluation Report: guardrails
Model: sim-perfect | Tasks: 5 | 2026-06-27 01:23:16

## Summary
- **Composite Score**: 0.760
- **Groundedness**: 1.000
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 1000ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| gr-001 | 1.00 | 8/0 | ✓ | 1000ms | 0.749 |
| gr-002 | 1.00 | 12/0 | ✓ | 1000ms | 0.759 |
| gr-003 | 1.00 | 12/0 | ✓ | 1000ms | 0.749 |
| gr-004 | 1.00 | 12/0 | ✓ | 1000ms | 0.774 |
| gr-005 | 1.00 | 12/0 | ✓ | 1000ms | 0.766 |
