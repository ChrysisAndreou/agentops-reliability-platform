# ─── AgentOps Reliability Platform ───
# Terraform module for Kubernetes deployment on GKE, EKS, or AKS.
#
# This module provisions the Kubernetes namespace and applies the
# Kustomize-based manifests. It assumes a running cluster and
# kubectl context already configured.
#
# Usage (minimal):
#   module "agentops" {
#     source = "./terraform"
#     cluster_context = "gke_my-project_europe-west4_agentops-cluster"
#   }

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
    kubectl = {
      source  = "alekc/kubectl"
      version = "~> 2.0"
    }
  }
}

variable "cluster_context" {
  description = "Kubernetes context name for kubectl"
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace for agentops resources"
  type        = string
  default     = "agentops"
}

variable "replicas" {
  description = "Number of API replicas"
  type        = number
  default     = 2
}

variable "ingress_host" {
  description = "Hostname for the ingress (e.g., agentops.example.com)"
  type        = string
  default     = "agentops.example.com"
}

variable "tls_cluster_issuer" {
  description = "cert-manager ClusterIssuer name"
  type        = string
  default     = "letsencrypt-prod"
}

variable "openai_api_key" {
  description = "OpenAI API key for LLM calls"
  type        = string
  sensitive   = true
  default     = ""
}

variable "anthropic_api_key" {
  description = "Anthropic API key for Claude models"
  type        = string
  sensitive   = true
  default     = ""
}

variable "storage_class" {
  description = "StorageClass for the data PVC"
  type        = string
  default     = "standard"
}

variable "storage_size" {
  description = "PVC size for trace/eval data"
  type        = string
  default     = "10Gi"
}

variable "cpu_request" {
  description = "CPU request per pod"
  type        = string
  default     = "250m"
}

variable "cpu_limit" {
  description = "CPU limit per pod"
  type        = string
  default     = "1"
}

variable "memory_request" {
  description = "Memory request per pod"
  type        = string
  default     = "512Mi"
}

variable "memory_limit" {
  description = "Memory limit per pod"
  type        = string
  default     = "2Gi"
}

variable "min_replicas" {
  description = "Minimum replicas for HPA"
  type        = number
  default     = 2
}

variable "max_replicas" {
  description = "Maximum replicas for HPA"
  type        = number
  default     = 8
}

# ─── Provider ───────────────────────────────────────────────────────

provider "kubernetes" {
  config_context = var.cluster_context
}

provider "kubectl" {
  config_context = var.cluster_context
}

# ─── Namespace ──────────────────────────────────────────────────────

resource "kubernetes_namespace" "agentops" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/name"    = "agentops-reliability-platform"
      "app.kubernetes.io/part-of" = "agentops"
    }
  }
}

# ─── Service Account ────────────────────────────────────────────────

resource "kubernetes_service_account" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "auth"
    }
  }
}

# ─── ConfigMap ──────────────────────────────────────────────────────

resource "kubernetes_config_map" "agentops_config" {
  metadata {
    name      = "agentops-config"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "config"
    }
  }
  data = {
    AGENTOPS_LOG_LEVEL          = "INFO"
    AGENTOPS_DB_PATH            = "/data/traces.db"
    AGENTOPS_EVAL_OUTPUT_DIR    = "/data/evals"
    AGENTOPS_RETRIEVAL_BACKEND  = "chromadb"
    AGENTOPS_CHROMA_PERSIST_DIR = "/data/chroma"
    AGENTOPS_MAX_RETRIES        = "3"
    AGENTOPS_DEFAULT_MODEL      = "gpt-4o"
  }
}

# ─── Secrets ────────────────────────────────────────────────────────

resource "kubernetes_secret" "llm_credentials" {
  metadata {
    name      = "agentops-llm-credentials"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "secrets"
    }
  }
  data = {
    OPENAI_API_KEY    = base64encode(var.openai_api_key)
    ANTHROPIC_API_KEY = base64encode(var.anthropic_api_key)
  }
  type = "Opaque"
}

# ─── Persistent Volume Claim ────────────────────────────────────────

resource "kubernetes_persistent_volume_claim" "data" {
  metadata {
    name      = "agentops-data"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "storage"
    }
  }
  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = var.storage_class
    resources {
      requests = {
        storage = var.storage_size
      }
    }
  }
}

# ─── Deployment ─────────────────────────────────────────────────────

resource "kubernetes_deployment" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "api"
    }
  }

  spec {
    replicas = var.replicas

    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_surge       = "1"
        max_unavailable = "0"
      }
    }

    selector {
      match_labels = {
        app = "agentops-api"
      }
    }

    template {
      metadata {
        labels = {
          app                          = "agentops-api"
          "app.kubernetes.io/name"    = "agentops-reliability-platform"
        }
      }

      spec {
        service_account_name            = kubernetes_service_account.agentops_api.metadata[0].name
        termination_grace_period_seconds = 30

        security_context {
          run_as_non_root = true
          run_as_user     = 1000
          fs_group        = 1000
        }

        container {
          name  = "api"
          image = "chrysisandreou/agentops-reliability-platform:latest"
          image_pull_policy = "Always"

          port {
            name           = "http"
            container_port = 8000
            protocol       = "TCP"
          }

          env_from {
            config_map_ref {
              name = kubernetes_config_map.agentops_config.metadata[0].name
            }
          }

          dynamic "env" {
            for_each = var.openai_api_key != "" ? [1] : []
            content {
              name = "OPENAI_API_KEY"
              value_from {
                secret_key_ref {
                  name = kubernetes_secret.llm_credentials.metadata[0].name
                  key  = "OPENAI_API_KEY"
                  optional = true
                }
              }
            }
          }

          dynamic "env" {
            for_each = var.anthropic_api_key != "" ? [1] : []
            content {
              name = "ANTHROPIC_API_KEY"
              value_from {
                secret_key_ref {
                  name = kubernetes_secret.llm_credentials.metadata[0].name
                  key  = "ANTHROPIC_API_KEY"
                  optional = true
                }
              }
            }
          }

          resources {
            requests = {
              cpu    = var.cpu_request
              memory = var.memory_request
            }
            limits = {
              cpu    = var.cpu_limit
              memory = var.memory_limit
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 15
            period_seconds        = 10
            timeout_seconds       = 3
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            timeout_seconds       = 2
            failure_threshold     = 3
          }

          volume_mount {
            name       = "data"
            mount_path = "/data"
          }
        }

        volume {
          name = "data"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.data.metadata[0].name
          }
        }
      }
    }
  }
}

# ─── Service ────────────────────────────────────────────────────────

resource "kubernetes_service" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "api"
    }
  }
  spec {
    type = "ClusterIP"
    selector = {
      app = "agentops-api"
    }
    port {
      name        = "http"
      port        = 80
      target_port = "8000"
      protocol    = "TCP"
    }
  }
}

# ─── HPA ────────────────────────────────────────────────────────────

resource "kubernetes_horizontal_pod_autoscaler_v2" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "autoscaling"
    }
  }
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.agentops_api.metadata[0].name
    }
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }

    behavior {
      scale_down {
        stabilization_window_seconds = 300
        policy {
          type           = "Percent"
          value          = 50
          period_seconds = 60
        }
      }
      scale_up {
        stabilization_window_seconds = 60
        policy {
          type           = "Percent"
          value          = 100
          period_seconds = 30
        }
      }
    }
  }
}

# ─── Pod Disruption Budget ──────────────────────────────────────────

resource "kubernetes_pod_disruption_budget_v1" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "reliability"
    }
  }
  spec {
    min_available = 1
    selector {
      match_labels = {
        app = "agentops-api"
      }
    }
  }
}

# ─── Network Policy ─────────────────────────────────────────────────

resource "kubernetes_network_policy_v1" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name" = "agentops-reliability-platform"
    }
  }
  spec {
    pod_selector {
      match_labels = {
        app = "agentops-api"
      }
    }
    policy_types = ["Ingress", "Egress"]

    ingress {
      from {
        namespace_selector {
          match_labels = {
            "kubernetes.io/metadata.name" = "ingress-nginx"
          }
        }
      }
      from {
        pod_selector {
          match_labels = {
            app = "agentops-api"
          }
        }
      }
      ports {
        port     = "8000"
        protocol = "TCP"
      }
    }

    egress {
      to {
        namespace_selector {}
      }
      ports {
        port     = "443"
        protocol = "TCP"
      }
      ports {
        port     = "53"
        protocol = "TCP"
      }
      ports {
        port     = "53"
        protocol = "UDP"
      }
    }
  }
}

# ─── Ingress ────────────────────────────────────────────────────────

resource "kubernetes_ingress_v1" "agentops_api" {
  metadata {
    name      = "agentops-api"
    namespace = kubernetes_namespace.agentops.metadata[0].name
    labels = {
      "app.kubernetes.io/name"      = "agentops-reliability-platform"
      "app.kubernetes.io/component" = "ingress"
    }
    annotations = {
      "cert-manager.io/cluster-issuer"                   = var.tls_cluster_issuer
      "nginx.ingress.kubernetes.io/ssl-redirect"         = "true"
      "nginx.ingress.kubernetes.io/proxy-body-size"      = "10m"
      "nginx.ingress.kubernetes.io/proxy-read-timeout"   = "120"
    }
  }
  spec {
    ingress_class_name = "nginx"
    tls {
      hosts       = [var.ingress_host]
      secret_name = "agentops-tls"
    }
    rule {
      host = var.ingress_host
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service.agentops_api.metadata[0].name
              port {
                name = "http"
              }
            }
          }
        }
      }
    }
  }
}

# ─── Outputs ────────────────────────────────────────────────────────

output "namespace" {
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.agentops.metadata[0].name
}

output "service_name" {
  description = "API service name"
  value       = kubernetes_service.agentops_api.metadata[0].name
}

output "ingress_host" {
  description = "Configured ingress hostname"
  value       = var.ingress_host
}

output "deployment_name" {
  description = "Deployment name"
  value       = kubernetes_deployment.agentops_api.metadata[0].name
}

output "health_endpoint" {
  description = "Health check URL"
  value       = "https://${var.ingress_host}/health"
}

output "api_endpoint" {
  description = "API base URL"
  value       = "https://${var.ingress_host}"
}
