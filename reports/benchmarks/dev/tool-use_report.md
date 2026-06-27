# Evaluation Report: tool-use
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.575
- **Groundedness**: 0.733
- **Verification Pass Rate**: 60.0%
- **Avg Latency**: 3183ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (2): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| tu-001 | 0.73 | 8/0 | ✗ | 2532ms | 0.428 |
| tu-002 | 0.73 | 8/0 | ✓ | 3699ms | 0.687 |
| tu-003 | 0.73 | 8/0 | ✓ | 2634ms | 0.668 |
| tu-004 | 0.73 | 8/0 | ✗ | 5339ms | 0.426 |
| tu-005 | 0.73 | 8/0 | ✓ | 1709ms | 0.669 |
