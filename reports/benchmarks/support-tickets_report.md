# Evaluation Report: support-tickets
Model: sim-production | Tasks: 5 | 2026-06-27 01:22:57

## Summary
- **Composite Score**: 0.715
- **Groundedness**: 0.859
- **Verification Pass Rate**: 100.0%
- **Avg Latency**: 2682ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| st-001 | 0.85 | 8/0 | ✓ | 1766ms | 0.727 |
| st-002 | 0.86 | 4/0 | ✓ | 3261ms | 0.704 |
| st-003 | 0.87 | 10/0 | ✓ | 3393ms | 0.716 |
| st-004 | 0.87 | 10/0 | ✓ | 3064ms | 0.707 |
| st-005 | 0.86 | 4/0 | ✓ | 1924ms | 0.718 |
