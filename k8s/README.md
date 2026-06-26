# ─── AgentOps Reliability Platform — Kubernetes Deployment ───

Production-grade Kubernetes manifests for deploying the AgentOps Reliability
Platform as a scalable, observable API service.

## What's Included

| Resource | Purpose |
|----------|---------|
| `namespace.yaml` | Isolated `agentops` namespace |
| `serviceaccount.yaml` | Dedicated service account for the API pods |
| `configmap.yaml` | Application configuration (log level, DB path, model, etc.) |
| `secret-template.yaml` | Template for LLM API credentials (fill before applying) |
| `pvc.yaml` | Persistent volume for trace/eval data |
| `deployment.yaml` | 2-replica deployment with resource limits, health probes, rolling updates |
| `service.yaml` | ClusterIP service on port 80 |
| `ingress.yaml` | TLS ingress via cert-manager + nginx |
| `hpa.yaml` | Horizontal pod autoscaling (2–8 replicas on CPU/memory) |
| `pdb.yaml` | Pod disruption budget (min 1 available) |
| `networkpolicy.yaml` | Default-deny with allowlisted ingress/egress |
| `kustomization.yaml` | Kustomize overlay for environment-specific patches |

## Prerequisites

- Kubernetes cluster (≥1.27, any provider: GKE, EKS, AKS, or self-managed)
- `kubectl` configured with cluster context
- `kustomize` (or `kubectl apply -k`, built into kubectl ≥1.21)
- cert-manager installed for TLS (or remove the annotation from Ingress)
- nginx-ingress controller
- API credentials secret applied (see below)

## Quick Start

### 1. Set your API keys

```bash
cp k8s/secret-template.yaml k8s/secret.yaml
# Edit k8s/secret.yaml and set your OPENAI_API_KEY and ANTHROPIC_API_KEY
# OR create the secret directly:
kubectl create secret generic agentops-llm-credentials \
  --namespace agentops \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Set your ingress hostname

Edit `k8s/ingress.yaml` and replace `agentops.example.com` with your domain.

### 3. Apply with Kustomize

```bash
kubectl apply -k k8s/
```

### 4. Verify

```bash
kubectl -n agentops get pods,svc,ingress,hpa
curl -s https://agentops.your-domain.com/health
```

Expected: `{"status": "ok"}`

## Terraform (Alternative)

If you prefer Terraform-managed infrastructure:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your cluster_context and ingress_host

terraform init
terraform plan
terraform apply
```

## Scaling

### Manual

```bash
kubectl -n agentops scale deployment agentops-api --replicas=5
```

### Auto-scaling (HPA, included)

The HPA scales between 2–8 replicas based on CPU (70%) and memory (80%)
utilization. Tune thresholds in `hpa.yaml`.

## Monitoring

```bash
# Pod status
kubectl -n agentops get pods -w

# HPA metrics
kubectl -n agentops describe hpa agentops-api

# Pod logs
kubectl -n agentops logs -l app=agentops-api -f

# Resource usage (requires metrics-server)
kubectl -n agentops top pods
```

## Architecture (on K8s)

```
                   ┌─────────────────┐
                   │   TLS Ingress   │
                   │ (cert-manager)  │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  nginx-ingress  │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │   Service       │
                   │  (ClusterIP:80) │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
     ┌────────▼───┐  ┌──────▼─────┐  ┌───▼──────────┐
     │  Pod A     │  │  Pod B     │  │  Pod C ...    │
     │  (8000)    │  │  (8000)    │  │  (HPA scales) │
     └─────┬──────┘  └──────┬─────┘  └──────┬────────┘
           │                │               │
           └────────────────┼───────────────┘
                            │
                   ┌────────▼────────┐
                   │      PVC        │
                   │  (10Gi RWO)     │
                   └─────────────────┘
```

## Security Notes

- NetworkPolicy restricts ingress to `nginx-ingress` namespace and intra-pod traffic only
- Pods run as non-root (UID 1000)
- API keys stored as Kubernetes Secrets (not in ConfigMaps or env files)
- TLS enforced via cert-manager and ingress annotation
- Pod disruption budget ensures at least 1 replica stays available during voluntary disruptions

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Pods CrashLoopBackOff | `kubectl -n agentops logs -l app=agentops-api` — likely missing API key |
| Ingress 404/503 | Verify service selector: `kubectl -n agentops describe svc agentops-api` |
| HPA not scaling | Verify metrics-server: `kubectl top pods -n agentops` |
| TLS cert not issued | `kubectl describe certificate -n agentops` |
