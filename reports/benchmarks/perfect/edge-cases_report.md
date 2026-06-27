# Evaluation Report: edge-cases
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
| ec-001 | 1.00 | 6/0 | ✓ | 1000ms | 0.759 |
| ec-002 | 1.00 | 10/0 | ✓ | 1000ms | 0.749 |
| ec-003 | 1.00 | 12/0 | ✓ | 1000ms | 0.769 |
| ec-004 | 1.00 | 8/0 | ✓ | 1000ms | 0.774 |
| ec-005 | 1.00 | 12/0 | ✓ | 1000ms | 0.749 |
