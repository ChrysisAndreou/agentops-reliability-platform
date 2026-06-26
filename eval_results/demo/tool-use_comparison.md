# Multi-Profile Comparison: tool-use
Profiles: perfect, production, development, unreliable
Tasks: 5
Generated: 2026-06-26 15:04:37

## Summary

| Profile | Groundedness | Citation Prec | Verify Rate | Composite |
|---------|-------------|--------------|-------------|-----------|
| perfect | 1.000 | 0.000 | 1.000 | 0.769 |
| production | 0.867 | 0.000 | 0.800 | 0.676 |
| development | 0.733 | 0.000 | 0.600 | 0.575 |
| unreliable | 0.467 | 0.000 | 0.200 | 0.399 |

## Per-Task Breakdown

### tu-001: A deployment pipeline needs 3 dedicated build agents for a monorepo with 5 servi...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.759 |
| production | 0.867 | ✓ | 0.718 |
| development | 0.733 | ✗ | 0.428 |
| unreliable | 0.467 | ✗ | 0.337 |

### tu-002: A team runs 20 concurrent builds on Shared agents (4 GB limit). Build times aver...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.769 |
| production | 0.867 | ✓ | 0.728 |
| development | 0.733 | ✓ | 0.687 |
| unreliable | 0.467 | ✗ | 0.347 |

### tu-003: Calculate the compound annual cost of CloudDeploy's Enterprise plan at $2,000/mo...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.779 |
| production | 0.867 | ✓ | 0.728 |
| development | 0.733 | ✓ | 0.668 |
| unreliable | 0.467 | ✗ | 0.358 |

### tu-004: What mathematical expression would you use to calculate the break-even point (in...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.789 |
| production | 0.867 | ✓ | 0.748 |
| development | 0.733 | ✗ | 0.426 |
| unreliable | 0.467 | ✓ | 0.616 |

### tu-005: A user's deployment failed with the message 'tool execution error: division by z...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.749 |
| production | 0.867 | ✗ | 0.458 |
| development | 0.733 | ✓ | 0.669 |
| unreliable | 0.467 | ✗ | 0.336 |
