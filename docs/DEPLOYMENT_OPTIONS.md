# Deployment Options (VPS, Managed Cloud, Kubernetes, k3s)

This guide covers all deployment modes for MedVision AI requested for this project:

1. VPS manual (OVH / Hetzner) with uv + systemd
2. VPS docker (OVH / Hetzner) with docker compose
3. AWS managed services (App Runner)
4. Azure managed services (Container Apps)
5. GCP managed services (Cloud Run)
6. Kubernetes (any cluster)
7. k3s (lightweight Kubernetes)

The deployment entrypoint is:

- scripts/deploy/deploy.sh

Environment variable template:

- scripts/deploy/env.example

Kubernetes manifests:

- deploy/k8s/base/

Dedicated target playbook:

- docs/DEPLOYMENT_PLAYBOOK_AWS_AZURE_K3S_OVH.md

Reverse proxy and DNS notes:

- docs/REVERSE_PROXY_DNS.md

## 1. Before you deploy

- Ensure models exist in artifacts/models.
- Ensure reports exist in artifacts/reports.
- Build context includes required runtime files via docker/Dockerfile.
- Optional: include sample data if Streamlit should browse local datasets.

Health endpoint to validate API after deployment:

- GET /health

## 2. Script usage

```bash
chmod +x scripts/deploy/deploy.sh
cp scripts/deploy/env.example .env.deploy
# Edit .env.deploy with your values
set -a
source .env.deploy
set +a

scripts/deploy/deploy.sh --help
```

## 3. VPS deployments (OVH / Hetzner)

### 3.1 Manual mode (uv + systemd)

```bash
scripts/deploy/deploy.sh vps-manual
```

This mode:

- syncs repo on VPS
- installs uv + dependencies
- creates and starts two systemd services:
  - medvision-ai-api
  - medvision-ai-streamlit

Required variables:

- SSH_USER, SSH_HOST, GIT_REPO

### 3.2 Docker mode (docker compose)

```bash
scripts/deploy/deploy.sh vps-docker
```

This mode:

- syncs repo on VPS
- installs Docker if needed
- runs production compose stack with docker-compose.yml + docker-compose.prod.yml

Required variables:

- SSH_USER, SSH_HOST, GIT_REPO

## 4. AWS managed deployment (App Runner)

```bash
scripts/deploy/deploy.sh aws-apprunner
```

This mode:

- builds and pushes two images to ECR
- deploys two App Runner services:
  - medvision-ai-api
  - medvision-ai-streamlit

Required variables:

- AWS_REGION
- AWS_ACCOUNT_ID
- ECR_REPO_API
- ECR_REPO_STREAMLIT

## 5. Azure managed deployment (Container Apps)

```bash
scripts/deploy/deploy.sh azure-containerapps
```

This mode:

- creates resource group + ACR + Container Apps environment
- pushes two images
- deploys two container apps (API and Streamlit)

Required variables:

- AZ_SUBSCRIPTION_ID
- AZ_RESOURCE_GROUP
- AZ_LOCATION
- AZ_ACR_NAME
- AZ_CONTAINERAPPS_ENV

## 6. GCP managed deployment (Cloud Run)

```bash
scripts/deploy/deploy.sh gcp-cloudrun
```

This mode:

- creates Artifact Registry repo if missing
- pushes API and Streamlit images
- deploys two Cloud Run services

Required variables:

- GCP_PROJECT_ID
- GCP_REGION
- GCP_ARTIFACT_REPO

## 7. Kubernetes deployment

```bash
scripts/deploy/deploy.sh k8s
```

Required variables:

- K8S_API_IMAGE
- K8S_STREAMLIT_IMAGE
- Optional: KUBE_CONTEXT

The script applies manifests in deploy/k8s/base and sets deployment images.

To enable ingress:

- install an ingress controller
- uncomment ingress in deploy/k8s/base/kustomization.yaml
- customize hosts in deploy/k8s/base/ingress.yaml

## 8. k3s deployment

```bash
scripts/deploy/deploy.sh k3s
```

Same behavior as k8s mode, using the current kubectl context.

For local verification on k3s node:

```bash
kubectl -n medvision get pods,svc
kubectl -n medvision port-forward svc/medvision-api 8000:8000
kubectl -n medvision port-forward svc/medvision-streamlit 8501:8501
```

## 9. Practical architecture notes

- API and Streamlit are deployed as separate services everywhere.
- Streamlit currently reads local data paths for sample browsing; if those paths are absent, upload-based flows still work.
- Model discovery relies on artifacts/models and artifacts/reports.

## 10. Recommended production hardening

- Put API and Streamlit behind a single domain and TLS reverse proxy.
- Add auth in front of Streamlit if exposed publicly.
- Externalize model and dataset artifacts to object storage.
- Add CI image builds and immutable image tags.
- Add monitoring/alerts (logs, uptime, latency, errors).

## 11. Reverse proxy and DNS

For production hostnames, TLS, and registrar DNS entries:

- docs/REVERSE_PROXY_DNS.md
