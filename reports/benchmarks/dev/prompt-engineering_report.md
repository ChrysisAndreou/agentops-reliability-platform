# Evaluation Report: prompt-engineering
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.528
- **Groundedness**: 0.733
- **Verification Pass Rate**: 40.0%
- **Avg Latency**: 4003ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (3): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| pe-001 | 0.73 | 8/0 | ✗ | 4390ms | 0.416 |
| pe-002 | 0.73 | 8/0 | ✗ | 2953ms | 0.443 |
| pe-003 | 0.73 | 8/0 | ✓ | 4696ms | 0.676 |
| pe-004 | 0.73 | 8/0 | ✓ | 4513ms | 0.681 |
| pe-005 | 0.73 | 8/0 | ✗ | 3466ms | 0.425 |
