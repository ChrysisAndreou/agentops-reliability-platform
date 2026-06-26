# Sample Support & Quality Tickets

This file contains realistic support, quality, and reliability tickets
that an AI agent would handle for the fictional CloudDeploy platform.
Each ticket includes an ID, category, user query, and expected handling.

---

## Ticket SQ-001: Build Failed with "Docker daemon not available"

**Category**: Build Failure
**Priority**: High
**User**: devops@acmecorp.com
**Message**:
> Our pipeline `acme-api` (build #4427) keeps failing with "Docker daemon
> not available". We're on the Dedicated plan. This started happening
> after we updated our clouddeploy.yml yesterday. How do we fix this?

**Expected Resolution**:
Check if `runtime: docker` is set correctly. Verify Docker is enabled
in pipeline settings. Confirm agent tier supports Docker (all do).
Check recent pipeline config changes.

**Ground Truth Answer**:
CloudDeploy Dedicated plan supports Docker. The pipeline config likely
modified or removed the `runtime: docker` directive. Check Settings →
Pipelines → acme-api → Configuration to compare with previous version.

---

## Ticket SQ-002: How to Enable 2FA

**Category**: Account Security
**Priority**: Low
**User**: newuser@startup.io
**Message**:
> I want to enable two-factor authentication on my CloudDeploy account.
> What options are available and how do I set it up?

**Expected Resolution**:
Direct to Settings → Security → Enable 2FA. Explain TOTP and SMS options.
Mention recovery codes.

**Ground Truth Answer**:
CloudDeploy supports TOTP (Google Authenticator, Authy) and SMS-based 2FA.
Navigate to Settings → Security, click "Enable Two-Factor Authentication",
scan the QR code, and verify. Save recovery codes securely.

---

## Ticket SQ-003: Out of Memory During Build

**Category**: Build Failure
**Priority**: High
**User**: eng-lead@bigco.com
**Message**:
> Our Java monolith build consistently runs out of memory on Shared agents.
> Build log says "Out of memory: exceeded 4 GB limit". What are our options
> besides upgrading to Dedicated? We're a small team on a budget.

**Expected Resolution**:
Memory limits by tier: Shared 4GB, Dedicated 16GB, Enterprise up to 64GB.
Optimisation options: multi-stage Docker builds, reducing parallel steps,
splitting the monolith build. If optimisation is insufficient, Dedicated
is required.

**Ground Truth Answer**:
Shared agents are limited to 4 GB. Optimisation steps: use multi-stage
Docker builds, split build into smaller parallel jobs, reduce test parallelisation.
If the monolith genuinely needs >4GB, upgrading to Dedicated ($49/mo, 16GB)
is the only option.

---

## Ticket SQ-004: Deployment Rolled Back Automatically

**Category**: Deployment Failure
**Priority**: Critical
**User**: oncall@sre-team.io
**Message**:
> Our production deployment (v3.7.2) was automatically rolled back to v3.7.1.
> The health check at /health started returning 503 after about 30 seconds.
> Why did this happen and how do we prevent it next time?

**Expected Resolution**:
Check the health check configuration. A health check returning 503 triggers
automatic rollback. Verify the /health endpoint behaviour in v3.7.2.
Suggest blue-green deployment with smoke tests before traffic shift.

**Ground Truth Answer**:
CloudDeploy's blue-green and rolling strategies monitor health checks.
If the configured path (default /health) returns non-2xx responses for
the health_check_interval, the deployment auto-rolls back. In v3.7.2,
the /health endpoint had a regression returning 503 after initialization.

---

## Ticket SQ-005: API Token Not Working

**Category**: API / Integration
**Priority**: Medium
**User**: automation@devops-team.net
**Message**:
> I generated a personal access token yesterday but it's returning 401
> Unauthorized on all API calls. The token was set to "never" expire.
> What could be wrong?

**Expected Resolution**:
Check if the token was accidentally revoked. Verify the token is sent
as `Authorization: Bearer <token>`. Check if the user's role changed
or account was disabled.

**Ground Truth Answer**:
Common causes: token was revoked (check Settings → API Tokens), incorrect
header format (must be `Authorization: Bearer <token>`), user account
disabled or role changed, organisation subscription lapsed.

---

## Ticket SQ-006: Pipeline Not Triggering on Push

**Category**: Pipeline Configuration
**Priority**: High
**User**: dev@startup.io
**Message**:
> We pushed to main branch 20 minutes ago but no pipeline was triggered.
> The clouddeploy.yml file is in the repo root. The dashboard shows the
> repo as "Connected" but the last build was 3 days ago. What's wrong?

**Expected Resolution**:
Check if pipeline was paused. Verify branch filter rules in pipeline settings.
Check if the push was to the correct remote. Verify webhook delivery in
GitHub/GitLab settings.

**Ground Truth Answer**:
Possible causes: pipeline paused (Settings → Pipelines → Resume), branch
filter excludes main, webhook not delivered (check GitHub/GitLab webhook
settings and CloudDeploy audit logs for webhook failures).

---

## Ticket SQ-007: Cannot Reset Password

**Category**: Account Recovery
**Priority**: High
**User**: locked-out@company.org
**Message**:
> I forgot my password and the "Forgot Password" link isn't sending me
> an email. I've checked spam. My account email is correct. How do I
> get back in?

**Expected Resolution**:
Verify email is correct. Check if account uses SAML SSO (password reset
handled by identity provider). If email-based, contact support for manual
verification.

**Ground Truth Answer**:
If your organisation uses SAML SSO (Okta, Azure AD, Google Workspace),
password reset is handled by your identity provider, not CloudDeploy.
Otherwise, contact support@clouddeploy.io from your account email for
manual verification and password reset.

---

## Ticket SQ-008: Understanding Build Minutes Billing

**Category**: Billing
**Priority**: Low
**User**: finance@acmecorp.com
**Message**:
> We received a bill showing 12,450 build minutes this month on our
> Dedicated plan. We thought the first 10,000 minutes were included?
> Also, how are parallel steps counted?

**Expected Resolution**:
Dedicated plan includes 10,000 build minutes/month + $0.01/additional.
Parallel steps count once. Queued time not counted. Breakdown available
in Settings → Billing → Usage.

**Ground Truth Answer**:
Dedicated plan: $49/month includes 10,000 build minutes. Additional minutes
at $0.01 each. 12,450 minutes = 10,000 included + 2,450 overage = $24.50
extra. Parallel steps count once (not multiplied). Queued time is not charged.

---

## Ticket SQ-009: Canary Deployment Reverting Traffic

**Category**: Deployment Strategy
**Priority**: Medium
**User**: platform@scaleup.dev
**Message**:
> We set up a canary deployment with 10%→50%→100% steps. At the 50% step,
> traffic reverted to the old version automatically. No error in the build
> itself. What would cause this?

**Expected Resolution**:
Canary steps auto-advance only if health metrics stay within thresholds.
Check if latency or error rate exceeded configured alert thresholds during
the 50% step. Check the Metrics dashboard for any anomaly during the canary
period.

**Ground Truth Answer**:
CloudDeploy monitors health metrics during canary steps. If p95 latency
or error rate exceeds configured thresholds, the canary automatically
reverts. Check Settings → Alerts for your latency and error rate thresholds,
and the Metrics dashboard for the canary period to identify the trigger.

---

## Ticket SQ-010: Migrating from GitHub to GitLab

**Category**: Repository Management
**Priority**: Medium
**User**: infra@migrating-co.io
**Message**:
> We're moving our repos from GitHub to GitLab. Our CloudDeploy pipelines
> are currently connected to GitHub. What's the process to switch without
> losing our pipeline history and configuration?

**Expected Resolution**:
Disconnect GitHub, connect GitLab, update repository URL in pipeline settings.
Build history is preserved. Pipeline config (clouddeploy.yml) lives in the
repo and transfers automatically. Variables and secrets remain in CloudDeploy
Vault.

**Ground Truth Answer**:
1. Settings → Repositories → Disconnect GitHub repo
2. Settings → Repositories → Connect Repository → GitLab
3. Authorise GitLab and select the migrated repo
4. Pipeline history, variables, and Vault secrets are preserved
5. The clouddeploy.yml in your repo remains unchanged
