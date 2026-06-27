# Evaluation Report: edge-cases
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.524
- **Groundedness**: 0.727
- **Verification Pass Rate**: 40.0%
- **Avg Latency**: 3106ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (3): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| ec-001 | 0.75 | 4/0 | ✗ | 2105ms | 0.433 |
| ec-002 | 0.69 | 7/0 | ✗ | 3694ms | 0.405 |
| ec-003 | 0.73 | 8/0 | ✓ | 4608ms | 0.686 |
| ec-004 | 0.73 | 6/0 | ✗ | 1613ms | 0.429 |
| ec-005 | 0.73 | 8/0 | ✓ | 3507ms | 0.667 |
