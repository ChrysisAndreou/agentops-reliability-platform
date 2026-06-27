# Evaluation Report: prompt-engineering
Model: sim-production | Tasks: 5 | 2026-06-27 01:24:12

## Summary
- **Composite Score**: 0.675
- **Groundedness**: 0.867
- **Verification Pass Rate**: 80.0%
- **Avg Latency**: 3038ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (1): Verification step explicitly rejected the output

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| pe-001 | 0.87 | 10/0 | ✓ | 2983ms | 0.724 |
| pe-002 | 0.87 | 10/0 | ✓ | 2803ms | 0.749 |
| pe-003 | 0.87 | 10/0 | ✓ | 3281ms | 0.707 |
| pe-004 | 0.87 | 10/0 | ✗ | 3392ms | 0.486 |
| pe-005 | 0.87 | 10/0 | ✓ | 2734ms | 0.708 |
