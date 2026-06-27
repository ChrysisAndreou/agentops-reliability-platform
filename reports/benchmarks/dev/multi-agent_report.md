# Evaluation Report: multi-agent
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.572
- **Groundedness**: 0.733
- **Verification Pass Rate**: 60.0%
- **Avg Latency**: 3546ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (2): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| ma-001 | 0.73 | 8/0 | ✓ | 4249ms | 0.679 |
| ma-002 | 0.73 | 8/0 | ✗ | 2379ms | 0.418 |
| ma-003 | 0.73 | 8/0 | ✗ | 4565ms | 0.423 |
| ma-004 | 0.73 | 8/0 | ✓ | 3146ms | 0.674 |
| ma-005 | 0.73 | 8/0 | ✓ | 3389ms | 0.667 |
