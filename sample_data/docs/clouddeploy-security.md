# CloudDeploy — Security & Compliance Guide

## Security Overview

CloudDeploy follows a defense-in-depth security model. This document outlines
the security controls, compliance certifications, and best practices for
securing your CloudDeploy organisation.

## Data Protection

### Encryption
- **In transit**: TLS 1.3 for all API and dashboard traffic. mTLS available for Enterprise plans.
- **At rest**: AES-256 encryption for all stored data including build logs, secrets, and configuration.
- **Secrets**: Vault values are encrypted with per-organisation keys. CloudDeploy employees cannot read your secrets.

### Data Residency
Data is stored in the region selected during organisation creation:
- US (us-east-1, us-west-2)
- EU (eu-west-1, eu-central-1) — GDPR compliant
- APAC (ap-southeast-1)

### Data Retention
- Build logs: 90 days (Shared), 1 year (Dedicated), custom (Enterprise)
- Deployment history: retained indefinitely while organisation is active
- Deleted organisation data: permanently erased within 30 days

## Authentication & Access Control

### Password Policy
- Minimum 12 characters
- Must include uppercase, lowercase, number, and special character
- Password history: last 5 passwords cannot be reused
- Account lockout after 10 failed attempts (30-minute cooldown)

### Session Management
- Sessions expire after 8 hours of inactivity
- Maximum 5 concurrent sessions per user
- Force logout of all sessions: Settings → Security → Logout Everywhere

### IP Allowlisting (Enterprise)
Restrict dashboard and API access to specific IP ranges:
Settings → Security → IP Allowlist

## Audit & Compliance

### SOC 2 Type II
CloudDeploy maintains SOC 2 Type II certification covering Security,
Availability, and Confidentiality. The latest report is available under NDA
from security@clouddeploy.io.

### GDPR
For EU customers: CloudDeploy acts as a data processor. A Data Processing
Agreement (DPA) is included in our Terms of Service. Data export and deletion
requests: privacy@clouddeploy.io.

### Audit Logs
All administrative actions are logged and immutable:
- User invitations, role changes, and removals
- Pipeline configuration changes
- Secret access (reads logged, values not recorded)
- Billing changes
- Organisation settings modifications

Export audit logs via Settings → Audit Logs → Export CSV.

## Vulnerability Management

### Dependency Scanning
CloudDeploy automatically scans dependencies in your pipelines for known
vulnerabilities (CVE database). Results appear in the Build details view.

### Container Image Scanning
For Docker-based pipelines, CloudDeploy scans base images for vulnerabilities
before build execution. High and critical vulnerabilities block the build
by default (configurable in pipeline settings).

### Responsible Disclosure
Security vulnerabilities in CloudDeploy itself: security@clouddeploy.io.
PGP key available at https://clouddeploy.io/security/pgp-key.asc.
Response within 24 hours; bounty programme at https://clouddeploy.io/security/bounty.

## Incident Response

### Reporting an Incident
If you suspect a security incident affecting your organisation:
1. Email security@clouddeploy.io with "INCIDENT" in the subject
2. Include your organisation name and a description
3. Do not include sensitive data in the initial email

### CloudDeploy Status Page
System status and ongoing incidents: https://status.clouddeploy.io

### SLA
- Critical incidents (platform unavailable): 15-minute response, 1-hour resolution target
- High severity (feature broken): 1-hour response, 4-hour resolution target
- Medium severity (degraded performance): 4-hour response, 24-hour resolution target
