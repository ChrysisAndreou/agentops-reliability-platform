# Sample Evaluation Report

Generated: 2026-06-26 | Model: dry-run validation (real LLM evaluation requires API keys)

## Summary

The AgentOps Reliability Platform ships with two benchmarks (10 tasks total) over the
CloudDeploy sample documentation corpus. This report demonstrates the evaluation report
format that real LLM runs produce.

### Benchmark: support-tickets (5 tasks)
Resolution of CloudDeploy support tickets using product documentation.

| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| st-001 - Docker daemon not available | 0.50 | 1/5 | ✓ | 5000ms | 0.686 |
| st-002 - Enable 2FA | 0.40 | 1/5 | ✓ | 5000ms | 0.656 |
| st-003 - Out of memory on Shared agents | 0.33 | 1/5 | ✓ | 5000ms | 0.636 |
| st-004 - Auto-rollback from health check 503 | 0.40 | 1/5 | ✓ | 5000ms | 0.656 |
| st-005 - Encryption standards | 0.50 | 1/5 | ✓ | 5000ms | 0.686 |

**Aggregate**: composite_mean=0.664, verification_pass_rate=100%, avg_latency=5000ms

### Benchmark: systems-quality (5 tasks)
Evaluation of CloudDeploy reliability/quality characteristics.

| Task | Grounded | Citations | Verified | Latency | Composite |
|------|----------|-----------|----------|---------|-----------|
| sq-001 - Deployment strategies comparison | 0.33 | 1/5 | ✓ | 5000ms | 0.636 |
| sq-002 - Incident response SLA | 0.33 | 1/5 | ✓ | 5000ms | 0.636 |
| sq-003 - Build agent tiers | 0.33 | 1/5 | ✓ | 5000ms | 0.636 |
| sq-004 - API token 401 troubleshooting | 0.40 | 1/5 | ✓ | 5000ms | 0.656 |
| sq-005 - Password policy summary | 0.29 | 1/5 | ✓ | 5000ms | 0.622 |

**Aggregate**: composite_mean=0.637, verification_pass_rate=100%, avg_latency=5000ms

### Failure Analysis
No failure patterns detected in dry-run validation. Real LLM runs would produce actual
failure classification from the 8-pattern taxonomy.

---

**Note**: These are dry-run validation scores using synthetic result data. Real LLM
evaluation requires setting `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and running:

```bash
agentops eval --benchmark support-tickets --model gpt-4o --output eval_results/
```

The evaluation harness produces identical format reports with real groundedness,
citation precision, and verification outcomes from live agent runs.
