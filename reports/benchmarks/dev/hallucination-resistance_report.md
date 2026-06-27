# Evaluation Report: hallucination-resistance
Model: sim-development | Tasks: 5 | 2026-06-27 01:23:08

## Summary
- **Composite Score**: 0.519
- **Groundedness**: 0.703
- **Verification Pass Rate**: 40.0%
- **Avg Latency**: 3204ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (3): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| hr-001 | 0.64 | 7/0 | ✗ | 5254ms | 0.413 |
| hr-002 | 0.73 | 6/0 | ✓ | 4043ms | 0.681 |
| hr-003 | 0.75 | 7/0 | ✗ | 3637ms | 0.422 |
| hr-004 | 0.67 | 4/0 | ✗ | 1551ms | 0.399 |
| hr-005 | 0.73 | 6/0 | ✓ | 1532ms | 0.679 |
