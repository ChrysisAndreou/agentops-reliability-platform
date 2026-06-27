# Evaluation Report: multi-agent
Model: sim-production | Tasks: 5 | 2026-06-27 01:24:12

## Summary
- **Composite Score**: 0.714
- **Groundedness**: 0.867
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 2786ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| ma-001 | 0.87 | 10/0 | ✓ | 2722ms | 0.726 |
| ma-002 | 0.87 | 10/0 | ✓ | 1656ms | 0.709 |
| ma-003 | 0.87 | 10/0 | ✓ | 3354ms | 0.707 |
| ma-004 | 0.87 | 10/0 | ✓ | 2893ms | 0.720 |
| ma-005 | 0.87 | 10/0 | ✓ | 3304ms | 0.707 |
