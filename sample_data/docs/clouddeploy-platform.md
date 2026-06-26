# CloudDeploy Platform — Product Documentation

## Overview

CloudDeploy is a continuous deployment platform for cloud-native applications.
It automates build, test, and release pipelines across AWS, GCP, and Azure.

## Getting Started

### Prerequisites
- A GitHub, GitLab, or Bitbucket repository
- Cloud provider credentials (AWS IAM role, GCP service account, or Azure service principal)
- Docker installed on build agents

### Quick Start
1. Connect your repository in the CloudDeploy dashboard
2. Add a `clouddeploy.yml` configuration file to your repository root
3. Push to your default branch — CloudDeploy automatically triggers the first pipeline

## Pipeline Configuration

### Basic Pipeline
```yaml
name: my-app
runtime: docker
steps:
  - name: build
    image: node:20
    commands:
      - npm ci
      - npm run build
  - name: test
    image: node:20
    commands:
      - npm test
  - name: deploy
    image: clouddeploy/cli:latest
    commands:
      - clouddeploy release --env staging
```

### Environment Variables
Secrets and configuration values are managed through the CloudDeploy Vault:
- Navigate to Settings → Vault in the dashboard
- Add key-value pairs; values are encrypted at rest
- Reference in pipelines as `${{ vault.MY_SECRET }}`

### Build Agents
CloudDeploy offers three agent tiers:
- **Shared** (free): Up to 2 concurrent builds, 30 min timeout
- **Dedicated** ($49/mo): Up to 10 concurrent builds, 2 hour timeout
- **Enterprise** (custom): Unlimited concurrent builds, custom timeout, VPC peering

## Deployment Strategies

### Rolling Deployments
Default strategy — gradually replaces instances. Configure with:
```yaml
deploy:
  strategy: rolling
  max_surge: 25%
  max_unavailable: 0
```

### Blue-Green Deployments
Provision a full new environment before switching traffic:
```yaml
deploy:
  strategy: blue-green
  traffic_shift: 100%
  health_check_path: /health
  health_check_interval: 10s
```

### Canary Deployments
Route a percentage of traffic to the new version:
```yaml
deploy:
  strategy: canary
  steps:
    - weight: 10%
      duration: 5m
    - weight: 50%
      duration: 10m
    - weight: 100%
```

## Monitoring & Alerts

### Metrics Dashboard
Access at `https://app.clouddeploy.io/dashboard/metrics`:
- Build success rate
- Deployment frequency
- Mean time to recovery (MTTR)
- Change failure rate

### Alert Rules
Configure alerts in Settings → Alerts:
- **Build Failure Alert**: Triggers when a build fails on the default branch
- **Deployment Failure Alert**: Triggers when a deployment is rolled back
- **Latency Alert**: Triggers when p95 latency exceeds threshold
- **Error Rate Alert**: Triggers when error rate exceeds 5%

### Notification Channels
- Email (default)
- Slack (connect in Settings → Integrations)
- PagerDuty (API key required)
- Webhooks (custom URL)

## Troubleshooting

### Common Build Failures

#### "Docker daemon not available"
The build agent could not connect to Docker. Verify:
1. The pipeline uses `runtime: docker`
2. You haven't disabled Docker in pipeline settings
3. For shared agents, Docker is always available

#### "Out of memory"
The build exceeded the agent's memory limit:
- Shared: 4 GB
- Dedicated: 16 GB
- Enterprise: configurable up to 64 GB

Solutions:
- Optimise your build steps
- Use multi-stage Docker builds
- Upgrade to a higher agent tier

#### "Authentication failed" with cloud provider
Check that your cloud credentials are correctly configured:
1. AWS: Verify the IAM role ARN in Settings → Cloud Providers
2. GCP: Verify the service account key JSON is valid
3. Azure: Verify the service principal has not expired

### Deployment Rollbacks
If a deployment fails, CloudDeploy automatically rolls back to the last
successful version. Manual rollback is available in the Deployments tab.

Run `clouddeploy rollback --env <environment> --to <version>` from the CLI.

## API Reference

### Authentication
All API requests require a Bearer token:
```bash
curl -H "Authorization: Bearer $CLOUDDEPLOY_TOKEN" \
  https://api.clouddeploy.io/v1/pipelines
```

### Endpoints

#### List Pipelines
`GET /v1/pipelines?status=active&limit=20`

#### Trigger Pipeline
`POST /v1/pipelines/{id}/trigger`
```json
{"branch": "main", "variables": {"DEBUG": "true"}}
```

#### Get Build Status
`GET /v1/builds/{build_id}`

#### List Deployments
`GET /v1/deployments?env=production&limit=10`

#### Rollback Deployment
`POST /v1/deployments/{id}/rollback`
```json
{"version": "v2.4.1", "reason": "performance regression"}
```

## Security

### Authentication Methods
- Username/password with optional 2FA (TOTP or SMS)
- SAML SSO (Okta, Azure AD, Google Workspace)
- Personal Access Tokens for API/CLI usage

### Access Control
Role-based access control (RBAC):
- **Viewer**: Read-only access to pipelines and builds
- **Developer**: Can trigger builds, view logs, manage variables
- **Admin**: Full access including billing, team management, and integrations
- **Owner**: Organisation-level access, can delete the organisation

### Audit Logs
Available under Settings → Audit Logs. Records:
- Pipeline triggers and modifications
- Deployment approvals and rollbacks
- User role changes
- Secret access (Vault reads without value exposure)

### Compliance
CloudDeploy is SOC 2 Type II certified. Data is encrypted in transit (TLS 1.3)
and at rest (AES-256). All customer data is isolated per organisation.

## Pricing FAQ

### How is billing calculated?
Based on build minutes and active deployment targets:
- Shared: Free for up to 500 build minutes/month
- Dedicated: $49/month + $0.01 per additional build minute
- Enterprise: Custom pricing; contact sales@clouddeploy.io

### What counts as a build minute?
Wall-clock time from pipeline start to completion. Queued time is not counted.
Parallel steps count once (not multiplied by concurrency).

### Can I pause my subscription?
Yes. Navigate to Settings → Billing → Pause Subscription. Data is retained
for 90 days. After 90 days, data is permanently deleted.
