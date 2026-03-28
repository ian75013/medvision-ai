#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_NAME="${APP_NAME:-medvision-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

log() {
  echo "[deploy] $*"
}

die() {
  echo "[deploy][error] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

remote_sync_workspace_overlay() {
  local ssh_target="$1"
  local ssh_port="$2"
  local app_dir="$3"

  tar \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='artifacts' \
    --exclude='data' \
    --exclude='mlruns' \
    --exclude='notebooks' \
    --exclude='tex' \
    -czf - \
    -C "$ROOT_DIR" . | ssh -p "$ssh_port" "$ssh_target" "mkdir -p $(printf %q "$app_dir") && tar -xzf - -C $(printf %q "$app_dir")"
}

remote_sudo_mode() {
  if [ "${ASK_SUDO_PASSWORD:-false}" = "true" ]; then
    printf '%s' "prompt"
  elif [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s' "stdin"
  else
    printf '%s' "plain"
  fi
}

write_remote_sudo_helpers() {
  local mode="$1"
  cat <<EOF
run_sudo() {
  if [ "$mode" = "prompt" ]; then
    sudo -v
    sudo "\$@"
  elif [ "$mode" = "stdin" ]; then
    printf '%s\\n' "\${SUDO_PASSWORD}" | sudo -S -p '' "\$@"
  else
    sudo "\$@"
  fi
}
EOF
}

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy/deploy.sh <target>

Targets:
  vps-manual           Deploy on OVH/Hetzner VPS with uv + systemd
  vps-docker           Deploy on OVH/Hetzner VPS with docker compose
  aws-apprunner        Deploy API + Streamlit to AWS App Runner
  azure-containerapps  Deploy API + Streamlit to Azure Container Apps
  gcp-cloudrun         Deploy API + Streamlit to GCP Cloud Run
  k8s                  Deploy on Kubernetes cluster
  k3s                  Deploy on k3s cluster (same manifests)

Important env vars by target:

vps-manual / vps-docker:
  SSH_USER, SSH_HOST, SSH_PORT=22, APP_DIR=/opt/medvision-ai
  GIT_REPO (ex: git@github.com:org/medvision-ai.git), GIT_BRANCH=main

aws-apprunner:
  AWS_REGION, AWS_ACCOUNT_ID
  ECR_REPO_API=medvision-api, ECR_REPO_STREAMLIT=medvision-streamlit

azure-containerapps:
  AZ_SUBSCRIPTION_ID, AZ_RESOURCE_GROUP, AZ_LOCATION
  AZ_ACR_NAME, AZ_CONTAINERAPPS_ENV

gcp-cloudrun:
  GCP_PROJECT_ID, GCP_REGION, GCP_ARTIFACT_REPO=medvision

k8s / k3s:
  KUBE_CONTEXT (optional)
  K8S_API_IMAGE, K8S_STREAMLIT_IMAGE

Common:
  IMAGE_TAG=latest
EOF
}

remote_bootstrap_repo() {
  local ssh_target="$1"
  local ssh_port="$2"
  local app_dir="$3"
  local git_repo="$4"
  local git_branch="$5"
  local sudo_password="${SUDO_PASSWORD:-}"
  local remote_user="${ssh_target%@*}"
  local sudo_mode
  sudo_mode="$(remote_sudo_mode)"
  local ssh_tty_args=()
  local tmp_local_script
  local tmp_remote_script

  if [ "$sudo_mode" = "prompt" ]; then
    ssh_tty_args=(-tt)
  fi

  tmp_local_script="$(mktemp)"
  tmp_remote_script="/tmp/${APP_NAME}-bootstrap-repo.sh"

  cat > "$tmp_local_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

$(write_remote_sudo_helpers "$sudo_mode")

if ! command -v git >/dev/null 2>&1; then
  run_sudo apt-get update
  run_sudo apt-get install -y git
fi
run_sudo mkdir -p "${app_dir}"
run_sudo chown -R "${remote_user}:${remote_user}" "${app_dir}"
if [ ! -d "${app_dir}/.git" ]; then
  git clone "${git_repo}" "${app_dir}"
fi
cd "${app_dir}"
git fetch --all --prune
git checkout "${git_branch}"
git pull --ff-only origin "${git_branch}"
EOF

  chmod +x "$tmp_local_script"
  scp -P "$ssh_port" "$tmp_local_script" "${ssh_target}:${tmp_remote_script}" >/dev/null
  ssh "${ssh_tty_args[@]}" -p "$ssh_port" "$ssh_target" "SUDO_PASSWORD=$(printf %q "$sudo_password") bash ${tmp_remote_script}"
  rm -f "$tmp_local_script"
}

deploy_vps_manual() {
  require_cmd ssh

  local ssh_user="${SSH_USER:-}"
  local ssh_host="${SSH_HOST:-}"
  local ssh_port="${SSH_PORT:-22}"
  local app_dir="${APP_DIR:-/opt/medvision-ai}"
  local git_repo="${GIT_REPO:-}"
  local git_branch="${GIT_BRANCH:-main}"
  local sudo_password="${SUDO_PASSWORD:-}"
  local mlflow_bind_ip="${MLFLOW_BIND_IP:-10.8.0.1}"
  local mlflow_host_port="${MLFLOW_HOST_PORT:-5000}"
  local api_bind_ip="${API_BIND_IP:-127.0.0.1}"
  local api_host_port="${API_HOST_PORT:-18000}"
  local streamlit_bind_ip="${STREAMLIT_BIND_IP:-127.0.0.1}"
  local streamlit_host_port="${STREAMLIT_HOST_PORT:-18501}"
  local sudo_mode
  local ssh_tty_args=()
  local tmp_local_script
  local tmp_remote_script

  [ -n "$ssh_user" ] || die "SSH_USER is required"
  [ -n "$ssh_host" ] || die "SSH_HOST is required"
  [ -n "$git_repo" ] || die "GIT_REPO is required"

  sudo_mode="$(remote_sudo_mode)"
  if [ "$sudo_mode" = "prompt" ]; then
    ssh_tty_args=(-tt)
  fi

  local ssh_target="${ssh_user}@${ssh_host}"
  log "Syncing repository on VPS"
  remote_bootstrap_repo "$ssh_target" "$ssh_port" "$app_dir" "$git_repo" "$git_branch"
  log "Syncing local workspace overlay to VPS"
  remote_sync_workspace_overlay "$ssh_target" "$ssh_port" "$app_dir"

  log "Installing runtime and configuring systemd services"
  tmp_local_script="$(mktemp)"
  tmp_remote_script="/tmp/${APP_NAME}-manual-deploy.sh"
  cat > "$tmp_local_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${app_dir}"

$(write_remote_sudo_helpers "$sudo_mode")

run_sudo apt-get update
run_sudo apt-get install -y curl python3 python3-venv python3-pip

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"

uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

run_sudo tee /etc/systemd/system/${APP_NAME}-api.service >/dev/null <<UNIT
[Unit]
Description=${APP_NAME} FastAPI
After=network.target

[Service]
Type=simple
User=${ssh_user}
WorkingDirectory=${app_dir}
ExecStart=${app_dir}/.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

run_sudo tee /etc/systemd/system/${APP_NAME}-streamlit.service >/dev/null <<UNIT
[Unit]
Description=${APP_NAME} Streamlit
After=network.target

[Service]
Type=simple
User=${ssh_user}
WorkingDirectory=${app_dir}
ExecStart=${app_dir}/.venv/bin/streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

run_sudo systemctl daemon-reload
run_sudo systemctl enable ${APP_NAME}-api ${APP_NAME}-streamlit
run_sudo systemctl restart ${APP_NAME}-api ${APP_NAME}-streamlit
run_sudo systemctl --no-pager status ${APP_NAME}-api ${APP_NAME}-streamlit | cat
EOF

  chmod +x "$tmp_local_script"
  scp -P "$ssh_port" "$tmp_local_script" "${ssh_target}:${tmp_remote_script}" >/dev/null
  ssh "${ssh_tty_args[@]}" -p "$ssh_port" "$ssh_target" "SUDO_PASSWORD=$(printf %q "$sudo_password") bash ${tmp_remote_script}"
  rm -f "$tmp_local_script"

  log "VPS manual deployment completed"
}

deploy_vps_docker() {
  require_cmd ssh

  local ssh_user="${SSH_USER:-}"
  local ssh_host="${SSH_HOST:-}"
  local ssh_port="${SSH_PORT:-22}"
  local app_dir="${APP_DIR:-/opt/medvision-ai}"
  local git_repo="${GIT_REPO:-}"
  local git_branch="${GIT_BRANCH:-main}"
  local sudo_password="${SUDO_PASSWORD:-}"
  local mlflow_bind_ip="${MLFLOW_BIND_IP:-10.8.0.1}"
  local mlflow_host_port="${MLFLOW_HOST_PORT:-5000}"
  local api_bind_ip="${API_BIND_IP:-127.0.0.1}"
  local api_host_port="${API_HOST_PORT:-18000}"
  local streamlit_bind_ip="${STREAMLIT_BIND_IP:-127.0.0.1}"
  local streamlit_host_port="${STREAMLIT_HOST_PORT:-18501}"
  local sudo_mode
  local ssh_tty_args=()
  local tmp_local_script
  local tmp_remote_script

  [ -n "$ssh_user" ] || die "SSH_USER is required"
  [ -n "$ssh_host" ] || die "SSH_HOST is required"
  [ -n "$git_repo" ] || die "GIT_REPO is required"
  [ -n "$mlflow_bind_ip" ] || die "MLFLOW_BIND_IP is required"
  [ -n "$mlflow_host_port" ] || die "MLFLOW_HOST_PORT is required"
  [ -n "$api_bind_ip" ] || die "API_BIND_IP is required"
  [ -n "$api_host_port" ] || die "API_HOST_PORT is required"
  [ -n "$streamlit_bind_ip" ] || die "STREAMLIT_BIND_IP is required"
  [ -n "$streamlit_host_port" ] || die "STREAMLIT_HOST_PORT is required"

  sudo_mode="$(remote_sudo_mode)"
  if [ "$sudo_mode" = "prompt" ]; then
    ssh_tty_args=(-tt)
  fi

  if [ "$mlflow_bind_ip" = "0.0.0.0" ] || [ "$mlflow_bind_ip" = "127.0.0.1" ]; then
    die "MLflow must stay VPN-only on VPS. Set MLFLOW_BIND_IP to your VPN interface IP, e.g. 10.8.0.1"
  fi

  local ssh_target="${ssh_user}@${ssh_host}"
  log "Syncing repository on VPS"
  remote_bootstrap_repo "$ssh_target" "$ssh_port" "$app_dir" "$git_repo" "$git_branch"
  log "Syncing local workspace overlay to VPS"
  remote_sync_workspace_overlay "$ssh_target" "$ssh_port" "$app_dir"

  log "Installing Docker and launching services"
  tmp_local_script="$(mktemp)"
  tmp_remote_script="/tmp/${APP_NAME}-docker-deploy.sh"
  cat > "$tmp_local_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${app_dir}"

$(write_remote_sudo_helpers "$sudo_mode")

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  run_sudo usermod -aG docker ${ssh_user}
fi

if ! run_sudo docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is required" >&2
  exit 1
fi

export MLFLOW_BIND_IP="${mlflow_bind_ip}"
export MLFLOW_HOST_PORT="${mlflow_host_port}"
export API_BIND_IP="${api_bind_ip}"
export API_HOST_PORT="${api_host_port}"
export STREAMLIT_BIND_IP="${streamlit_bind_ip}"
export STREAMLIT_HOST_PORT="${streamlit_host_port}"

run_compose() {
  run_sudo env \
    MLFLOW_BIND_IP="${mlflow_bind_ip}" \
    MLFLOW_HOST_PORT="${mlflow_host_port}" \
    API_BIND_IP="${api_bind_ip}" \
    API_HOST_PORT="${api_host_port}" \
    STREAMLIT_BIND_IP="${streamlit_bind_ip}" \
    STREAMLIT_HOST_PORT="${streamlit_host_port}" \
    docker compose -f docker-compose.yml -f docker-compose.prod.yml "\$@"
}

run_compose down --remove-orphans || true

if command -v ss >/dev/null 2>&1; then
  if ss -ltn "sport = :${mlflow_host_port}" | awk 'NR>1 {print}' | grep -q .; then
    echo "[deploy][error] Host port ${mlflow_host_port} is already in use on VPS." >&2
    echo "[deploy][error] Set MLFLOW_HOST_PORT in your OVH env file, then redeploy." >&2
    ss -ltnp "sport = :${mlflow_host_port}" || true
    exit 1
  fi
  if ss -ltn "sport = :${api_host_port}" | awk 'NR>1 {print}' | grep -q .; then
    echo "[deploy][error] Host port ${api_host_port} is already in use on VPS." >&2
    echo "[deploy][error] Set API_HOST_PORT in your OVH env file, then redeploy." >&2
    ss -ltnp "sport = :${api_host_port}" || true
    exit 1
  fi
  if ss -ltn "sport = :${streamlit_host_port}" | awk 'NR>1 {print}' | grep -q .; then
    echo "[deploy][error] Host port ${streamlit_host_port} is already in use on VPS." >&2
    echo "[deploy][error] Set STREAMLIT_HOST_PORT in your OVH env file, then redeploy." >&2
    ss -ltnp "sport = :${streamlit_host_port}" || true
    exit 1
  fi
fi

run_compose up --build -d --remove-orphans
run_compose ps
EOF

  chmod +x "$tmp_local_script"
  scp -P "$ssh_port" "$tmp_local_script" "${ssh_target}:${tmp_remote_script}" >/dev/null
  ssh "${ssh_tty_args[@]}" -p "$ssh_port" "$ssh_target" "SUDO_PASSWORD=$(printf %q "$sudo_password") bash ${tmp_remote_script}"
  rm -f "$tmp_local_script"

  log "VPS docker deployment completed"
}

aws_build_and_push() {
  local repo_name="$1"
  local image_uri="$2"

  aws ecr describe-repositories --repository-names "$repo_name" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$repo_name" >/dev/null

  docker build -f "$ROOT_DIR/docker/Dockerfile" -t "$image_uri" "$ROOT_DIR"
  docker push "$image_uri"
}

deploy_aws_apprunner() {
  require_cmd aws
  require_cmd docker

  local region="${AWS_REGION:-}"
  local account_id="${AWS_ACCOUNT_ID:-}"
  local repo_api="${ECR_REPO_API:-medvision-api}"
  local repo_streamlit="${ECR_REPO_STREAMLIT:-medvision-streamlit}"

  [ -n "$region" ] || die "AWS_REGION is required"
  [ -n "$account_id" ] || die "AWS_ACCOUNT_ID is required"

  local api_image="${account_id}.dkr.ecr.${region}.amazonaws.com/${repo_api}:${IMAGE_TAG}"
  local streamlit_image="${account_id}.dkr.ecr.${region}.amazonaws.com/${repo_streamlit}:${IMAGE_TAG}"

  aws ecr get-login-password --region "$region" | \
    docker login --username AWS --password-stdin "${account_id}.dkr.ecr.${region}.amazonaws.com"

  log "Building and pushing API image: $api_image"
  aws_build_and_push "$repo_api" "$api_image"

  log "Building and pushing Streamlit image: $streamlit_image"
  aws_build_and_push "$repo_streamlit" "$streamlit_image"

  log "Deploying API service on App Runner"
  aws apprunner create-service \
    --region "$region" \
    --service-name "${APP_NAME}-api" \
    --source-configuration "ImageRepository={ImageIdentifier=${api_image},ImageRepositoryType=ECR,ImageConfiguration={Port=8000,StartCommand='uvicorn src.api.main:app --host 0.0.0.0 --port 8000'}}" \
    --instance-configuration "Cpu=1024,Memory=2048" \
    >/dev/null 2>&1 || \
  aws apprunner update-service \
    --region "$region" \
    --service-arn "$(aws apprunner list-services --region "$region" --query "ServiceSummaryList[?ServiceName=='${APP_NAME}-api'].ServiceArn | [0]" --output text)" \
    --source-configuration "ImageRepository={ImageIdentifier=${api_image},ImageRepositoryType=ECR,ImageConfiguration={Port=8000,StartCommand='uvicorn src.api.main:app --host 0.0.0.0 --port 8000'}}" \
    >/dev/null

  log "Deploying Streamlit service on App Runner"
  aws apprunner create-service \
    --region "$region" \
    --service-name "${APP_NAME}-streamlit" \
    --source-configuration "ImageRepository={ImageIdentifier=${streamlit_image},ImageRepositoryType=ECR,ImageConfiguration={Port=8501,StartCommand='streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501'}}" \
    --instance-configuration "Cpu=1024,Memory=2048" \
    >/dev/null 2>&1 || \
  aws apprunner update-service \
    --region "$region" \
    --service-arn "$(aws apprunner list-services --region "$region" --query "ServiceSummaryList[?ServiceName=='${APP_NAME}-streamlit'].ServiceArn | [0]" --output text)" \
    --source-configuration "ImageRepository={ImageIdentifier=${streamlit_image},ImageRepositoryType=ECR,ImageConfiguration={Port=8501,StartCommand='streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501'}}" \
    >/dev/null

  log "AWS App Runner deployment completed"
}

deploy_azure_containerapps() {
  require_cmd az
  require_cmd docker

  local sub_id="${AZ_SUBSCRIPTION_ID:-}"
  local rg="${AZ_RESOURCE_GROUP:-}"
  local location="${AZ_LOCATION:-westeurope}"
  local acr_name="${AZ_ACR_NAME:-}"
  local env_name="${AZ_CONTAINERAPPS_ENV:-${APP_NAME}-env}"

  [ -n "$sub_id" ] || die "AZ_SUBSCRIPTION_ID is required"
  [ -n "$rg" ] || die "AZ_RESOURCE_GROUP is required"
  [ -n "$acr_name" ] || die "AZ_ACR_NAME is required"

  az account set --subscription "$sub_id"
  az group create -n "$rg" -l "$location" >/dev/null

  az acr show -n "$acr_name" -g "$rg" >/dev/null 2>&1 || \
    az acr create -n "$acr_name" -g "$rg" --sku Basic --admin-enabled true >/dev/null

  local login_server
  login_server="$(az acr show -n "$acr_name" -g "$rg" --query loginServer -o tsv)"

  local api_image="${login_server}/${APP_NAME}-api:${IMAGE_TAG}"
  local streamlit_image="${login_server}/${APP_NAME}-streamlit:${IMAGE_TAG}"

  az acr login -n "$acr_name"
  docker build -f "$ROOT_DIR/docker/Dockerfile" -t "$api_image" "$ROOT_DIR"
  docker push "$api_image"
  docker build -f "$ROOT_DIR/docker/Dockerfile" -t "$streamlit_image" "$ROOT_DIR"
  docker push "$streamlit_image"

  az containerapp env show -n "$env_name" -g "$rg" >/dev/null 2>&1 || \
    az containerapp env create -n "$env_name" -g "$rg" -l "$location" >/dev/null

  local acr_user acr_pass
  acr_user="$(az acr credential show -n "$acr_name" --query username -o tsv)"
  acr_pass="$(az acr credential show -n "$acr_name" --query passwords[0].value -o tsv)"

  az containerapp create \
    -n "${APP_NAME}-api" -g "$rg" --environment "$env_name" \
    --image "$api_image" --target-port 8000 --ingress external \
    --registry-server "$login_server" --registry-username "$acr_user" --registry-password "$acr_pass" \
    --command uvicorn --args "src.api.main:app" "--host" "0.0.0.0" "--port" "8000" \
    --cpu 1.0 --memory 2.0Gi >/dev/null 2>&1 || \
  az containerapp update \
    -n "${APP_NAME}-api" -g "$rg" \
    --image "$api_image" \
    --set-env-vars "PORT=8000" >/dev/null

  az containerapp create \
    -n "${APP_NAME}-streamlit" -g "$rg" --environment "$env_name" \
    --image "$streamlit_image" --target-port 8501 --ingress external \
    --registry-server "$login_server" --registry-username "$acr_user" --registry-password "$acr_pass" \
    --command streamlit --args "run" "streamlit_app.py" "--server.address" "0.0.0.0" "--server.port" "8501" \
    --cpu 1.0 --memory 2.0Gi >/dev/null 2>&1 || \
  az containerapp update \
    -n "${APP_NAME}-streamlit" -g "$rg" \
    --image "$streamlit_image" \
    --set-env-vars "PORT=8501" >/dev/null

  log "Azure Container Apps deployment completed"
}

deploy_gcp_cloudrun() {
  require_cmd gcloud
  require_cmd docker

  local project_id="${GCP_PROJECT_ID:-}"
  local region="${GCP_REGION:-europe-west1}"
  local ar_repo="${GCP_ARTIFACT_REPO:-medvision}"

  [ -n "$project_id" ] || die "GCP_PROJECT_ID is required"

  gcloud config set project "$project_id" >/dev/null
  gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com >/dev/null

  gcloud artifacts repositories describe "$ar_repo" --location "$region" >/dev/null 2>&1 || \
    gcloud artifacts repositories create "$ar_repo" --repository-format docker --location "$region" >/dev/null

  local reg_host="${region}-docker.pkg.dev"
  local api_image="${reg_host}/${project_id}/${ar_repo}/${APP_NAME}-api:${IMAGE_TAG}"
  local streamlit_image="${reg_host}/${project_id}/${ar_repo}/${APP_NAME}-streamlit:${IMAGE_TAG}"

  gcloud auth configure-docker "$reg_host" --quiet

  docker build -f "$ROOT_DIR/docker/Dockerfile" -t "$api_image" "$ROOT_DIR"
  docker push "$api_image"
  docker build -f "$ROOT_DIR/docker/Dockerfile" -t "$streamlit_image" "$ROOT_DIR"
  docker push "$streamlit_image"

  gcloud run deploy "${APP_NAME}-api" \
    --image "$api_image" \
    --region "$region" \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --command uvicorn \
    --args src.api.main:app,--host,0.0.0.0,--port,8000 \
    --cpu 1 --memory 2Gi --min-instances 0 --max-instances 3

  gcloud run deploy "${APP_NAME}-streamlit" \
    --image "$streamlit_image" \
    --region "$region" \
    --platform managed \
    --allow-unauthenticated \
    --port 8501 \
    --command streamlit \
    --args run,streamlit_app.py,--server.address,0.0.0.0,--server.port,8501 \
    --cpu 1 --memory 2Gi --min-instances 0 --max-instances 2

  log "GCP Cloud Run deployment completed"
}

deploy_k8s_core() {
  require_cmd kubectl
  local namespace="medvision"

  local api_image="${K8S_API_IMAGE:-}"
  local streamlit_image="${K8S_STREAMLIT_IMAGE:-}"

  [ -n "$api_image" ] || die "K8S_API_IMAGE is required"
  [ -n "$streamlit_image" ] || die "K8S_STREAMLIT_IMAGE is required"

  if [ -n "${KUBE_CONTEXT:-}" ]; then
    kubectl config use-context "$KUBE_CONTEXT"
  fi

  kubectl apply -f "$ROOT_DIR/deploy/k8s/base/namespace.yaml"
  kubectl apply -f "$ROOT_DIR/deploy/k8s/base/configmap.yaml"
  kubectl apply -f "$ROOT_DIR/deploy/k8s/base/api-deployment.yaml"
  kubectl apply -f "$ROOT_DIR/deploy/k8s/base/api-service.yaml"
  kubectl apply -f "$ROOT_DIR/deploy/k8s/base/streamlit-deployment.yaml"
  kubectl apply -f "$ROOT_DIR/deploy/k8s/base/streamlit-service.yaml"

  kubectl -n "$namespace" set image deployment/medvision-api api="$api_image"
  kubectl -n "$namespace" set image deployment/medvision-streamlit streamlit="$streamlit_image"

  kubectl -n "$namespace" rollout status deployment/medvision-api --timeout=300s
  kubectl -n "$namespace" rollout status deployment/medvision-streamlit --timeout=300s
  kubectl -n "$namespace" get svc,pods

  log "Kubernetes deployment completed"
}

deploy_k8s() {
  deploy_k8s_core
}

deploy_k3s() {
  if ! command -v kubectl >/dev/null 2>&1; then
    die "kubectl is required for k3s deployment"
  fi
  deploy_k8s_core
}

main() {
  local target="${1:-}"
  case "$target" in
    vps-manual)
      deploy_vps_manual
      ;;
    vps-docker)
      deploy_vps_docker
      ;;
    aws-apprunner)
      deploy_aws_apprunner
      ;;
    azure-containerapps)
      deploy_azure_containerapps
      ;;
    gcp-cloudrun)
      deploy_gcp_cloudrun
      ;;
    k8s)
      deploy_k8s
      ;;
    k3s)
      deploy_k3s
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      die "Unknown target: $target"
      ;;
  esac
}

main "$@"
