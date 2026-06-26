# CloudDeploy — Onboarding Guide

## Welcome to CloudDeploy

CloudDeploy helps teams ship software faster and more reliably. This guide
walks you through setting up your first project.

## Step 1: Create an Account

Visit https://app.clouddeploy.io/signup. You can sign up with:
- GitHub (recommended for automatic repo access)
- GitLab
- Google Workspace
- Email and password

After signup, you'll land on the organisation setup page.

## Step 2: Create an Organisation

Organisations group your projects, team members, and billing:
1. Enter your organisation name (e.g., "Acme Corp")
2. Choose a unique slug (used in URLs: `acme-corp.clouddeploy.io`)
3. Select a plan: Shared (free), Dedicated ($49/mo), or Enterprise

## Step 3: Connect a Repository

1. Click "Connect Repository" on the dashboard
2. Authorise CloudDeploy to access your GitHub/GitLab account
3. Select a repository from the list
4. CloudDeploy scans for a `clouddeploy.yml` file

If no configuration file is found, CloudDeploy offers to generate one based
on your project's language and framework.

## Step 4: Configure Cloud Provider

1. Navigate to Settings → Cloud Providers
2. Choose your provider (AWS, GCP, or Azure)
3. Follow the provider-specific instructions to grant access

For AWS: Create an IAM role with the CloudDeploy trust policy (available in
the dashboard). For GCP: Upload a service account JSON key. For Azure:
Provide the tenant ID, subscription ID, and create a service principal.

## Step 5: First Deployment

1. Push a commit to your default branch
2. CloudDeploy detects the push and starts a pipeline automatically
3. Monitor progress in the Builds tab
4. After a successful build, deployment begins automatically

## Step 6: Invite Team Members

1. Go to Settings → Team
2. Click "Invite Member"
3. Enter their email and select a role (Viewer, Developer, Admin)
4. They receive an invitation email with a link to join

## Next Steps

- Set up Slack notifications in Settings → Integrations
- Configure alert rules for build and deployment failures
- Explore the Metrics dashboard for DORA metrics
- Set up a staging environment for pre-production testing

## Account Security

### Two-Factor Authentication (2FA)

CloudDeploy supports TOTP-based 2FA (Google Authenticator, Authy, 1Password)
and SMS-based 2FA. To enable:

1. Go to Settings → Security
2. Click "Enable Two-Factor Authentication"
3. Scan the QR code with your authenticator app
4. Enter the verification code to confirm

If you lose access to your 2FA device, use one of your recovery codes (shown
during setup — save them securely). If recovery codes are also lost, contact
support@clouddeploy.io from your account email for manual recovery.

### API Tokens

Create personal access tokens for CLI and API access:
1. Settings → API Tokens
2. Click "Generate Token"
3. Give it a descriptive name (e.g., "CI/CD Pipeline")
4. Choose an expiration: 30 days, 90 days, 1 year, or never
5. Copy the token — it won't be shown again

Tokens inherit your RBAC permissions. Revoke tokens at any time from the
same page.

## Billing & Plans

### Upgrading Your Plan

1. Settings → Billing → Change Plan
2. Select Dedicated or Enterprise
3. Enter payment information (credit card or invoice for Enterprise)
4. The upgrade takes effect immediately; unused Shared time is not credited

### Viewing Usage

Settings → Billing → Usage shows:
- Current billing period build minutes
- Active deployment targets
- Invoice history (last 12 months)

### Cancelling

Settings → Billing → Cancel Subscription. Your data is retained for 90 days.
During this period, you can reactivate without data loss.
