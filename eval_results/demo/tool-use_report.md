# Evaluation Report: tool-use
Model: sim-production | Tasks: 5 | 2026-06-26 15:04:14

## Summary
- **Composite Score**: 0.676
- **Groundedness**: 0.867
- **Verification Pass Rate**: 80.0%
- **Avg Latency**: 2173ms
- **Citation Precision**: 0.000

## Failure Analysis
- [!!!] **hallucination** (5): Agent produced claims not grounded in retrieved evidence
- [!!] **retrieval_gap** (5): No relevant chunks retrieved for the query
- [!!] **verification_failure** (1): Verification step explicitly rejected the output
- [!] **tool_error** (1): A tool call failed during execution
- [!!] **timeout_or_abort** (1): Agent run didn't complete successfully due to error

## Per-Task Metrics
| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| tu-001 | 0.87 | 10/0 | ✓ | 1805ms | 0.718 |
| tu-002 | 0.87 | 10/0 | ✓ | 2345ms | 0.728 |
| tu-003 | 0.87 | 10/0 | ✓ | 2501ms | 0.728 |
| tu-004 | 0.87 | 10/0 | ✓ | 2368ms | 0.748 |
| tu-005 | 0.87 | 10/0 | ✗ | 1846ms | 0.458 |
