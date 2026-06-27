# Evaluation Report: guardrails
Model: sim-production | Tasks: 5 | 2026-06-27 01:22:57

## Summary
- **Composite Score**: 0.712
- **Groundedness**: 0.846
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 2744ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| gr-001 | 0.82 | 7/0 | ✓ | 2352ms | 0.693 |
| gr-002 | 0.87 | 10/0 | ✓ | 1972ms | 0.718 |
| gr-003 | 0.87 | 10/0 | ✓ | 2930ms | 0.708 |
| gr-004 | 0.81 | 10/0 | ✓ | 3060ms | 0.716 |
| gr-005 | 0.87 | 10/0 | ✓ | 3406ms | 0.724 |
