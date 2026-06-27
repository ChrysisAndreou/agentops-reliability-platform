# Evaluation Report: hallucination-resistance
Model: sim-perfect | Tasks: 5 | 2026-06-27 01:23:16

## Summary
- **Composite Score**: 0.771
- **Groundedness**: 1.000
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 1000ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| hr-001 | 1.00 | 10/0 | ✓ | 1000ms | 0.787 |
| hr-002 | 1.00 | 8/0 | ✓ | 1000ms | 0.766 |
| hr-003 | 1.00 | 9/0 | ✓ | 1000ms | 0.762 |
| hr-004 | 1.00 | 7/0 | ✓ | 1000ms | 0.766 |
| hr-005 | 1.00 | 8/0 | ✓ | 1000ms | 0.774 |
