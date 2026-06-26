# Multi-Profile Comparison: multi-step
Profiles: perfect, production, development, unreliable
Tasks: 5
Generated: 2026-06-26 15:04:37

## Summary

| Profile | Groundedness | Citation Prec | Verify Rate | Composite |
|---------|-------------|--------------|-------------|-----------|
| perfect | 1.000 | 0.000 | 1.000 | 0.767 |
| production | 0.867 | 0.000 | 0.800 | 0.673 |
| development | 0.724 | 0.000 | 0.800 | 0.626 |
| unreliable | 0.467 | 0.000 | 0.600 | 0.485 |

## Per-Task Breakdown

### ms-001: A customer on the Starter plan wants to deploy a Python Django app with PostgreS...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.749 |
| production | 0.867 | ✗ | 0.458 |
| development | 0.733 | ✓ | 0.668 |
| unreliable | 0.467 | ✓ | 0.587 |

### ms-002: A security audit flagged three issues: (1) API tokens set to never expire, (2) p...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.759 |
| production | 0.867 | ✓ | 0.718 |
| development | 0.688 | ✓ | 0.665 |
| unreliable | 0.467 | ✗ | 0.334 |

### ms-003: Compare CloudDeploy's three build agent tiers (Shared, Dedicated, Enterprise) ac...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.766 |
| production | 0.867 | ✓ | 0.724 |
| development | 0.733 | ✓ | 0.674 |
| unreliable | 0.467 | ✗ | 0.337 |

### ms-004: A production incident occurred: a blue-green deployment of v3.8.0 caused 503 err...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.779 |
| production | 0.867 | ✓ | 0.728 |
| development | 0.733 | ✓ | 0.696 |
| unreliable | 0.467 | ✓ | 0.583 |

### ms-005: Design a deployment pipeline for a compliance-regulated application that needs: ...

| Profile | Grounded | Verified | Composite |
|---------|----------|----------|-----------|
| perfect | 1.000 | ✓ | 0.779 |
| production | 0.867 | ✓ | 0.737 |
| development | 0.733 | ✗ | 0.428 |
| unreliable | 0.467 | ✓ | 0.584 |
