# Structured Output Evaluation: structured-output + function-calling

**Generated:** 2026-06-27T03:43:16.103272

## Summary Metrics

| Metric | Value |
|--------|-------|
| Schema Adherence (avg) | 0.975 |
| Valid / Invalid Outputs | 4 / 1 |
| Function Call Correctness (avg) | 1.000 |
| Correct / Incorrect Calls | 12 / 0 |
| Tool Selection Errors | 0 |
| Parameter Errors | 0 |
| **Composite Score** | **0.988** |

## Schema Validation Results

| Task | Schema | Adherence | Valid | Fields (OK/Total) | Errors |
|------|--------|-----------|-------|-------------------|--------|
| ✓ | incident-report | 1.000 | PASS | 6/6 | 0 |
| ✓ | pipeline-config | 1.000 | PASS | 7/7 | 0 |
| ✗ | support-ticket | 0.875 | FAIL | 7/8 | 1 |
| ✓ | metrics-query | 1.000 | PASS | 4/4 | 0 |
| ✓ | audit-report | 1.000 | PASS | 2/2 | 0 |

### Schema: support-ticket
- **resolved_at** [wrong_type]: Field 'resolved_at' is null (expected string)
## Function Call Results

| Call ID | Expected → Actual | Correctness | Params (OK/Total) | Errors |
|---------|-------------------|-------------|--------------------|--------|
| ✓ | check_resource_usage → check_resource_usage | 1.000 | 2/2 | 0 |
| ✓ | get_service_logs → get_service_logs | 1.000 | 3/3 | 0 |
| ✓ | get_service_logs → get_service_logs | 1.000 | 3/3 | 0 |
| ✓ | check_resource_usage → check_resource_usage | 1.000 | 2/2 | 0 |
| ✓ | page_oncall → page_oncall | 1.000 | 3/3 | 0 |
| ✓ | get_db_version → get_db_version | 1.000 | 1/1 | 0 |
| ✓ | create_db_backup → create_db_backup | 1.000 | 2/2 | 0 |
| ✓ | run_migration → run_migration | 1.000 | 3/3 | 0 |
| ✓ | check_migration_status → check_migration_status | 1.000 | 1/1 | 0 |
| ✓ | flag_security_concern → flag_security_concern | 1.000 | 2/2 | 0 |
| ✓ | get_deployment_logs → get_deployment_logs | 1.000 | 2/2 | 0 |
| ✓ | create_support_ticket → create_support_ticket | 1.000 | 3/3 | 0 |
