#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-docker}"
ENV_FILE="${2:-$ROOT_DIR/scripts/deploy/env.ovh.example}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

configure_caddy() {
  local ssh_user="${SSH_USER:-}"
  local ssh_host="${SSH_HOST:-}"
  local ssh_port="${SSH_PORT:-22}"
  local api_domain="${API_DOMAIN:-}"
  local app_domain="${APP_DOMAIN:-}"
  local caddy_email="${CADDY_EMAIL:-}"

  if [ -z "$api_domain" ] || [ -z "$app_domain" ] || [ -z "$caddy_email" ]; then
    echo "Caddy provisioning skipped: set API_DOMAIN, APP_DOMAIN and CADDY_EMAIL in env file." >&2
    return 0
  fi

  if [ -z "$ssh_user" ] || [ -z "$ssh_host" ]; then
    echo "Caddy provisioning skipped: SSH_USER/SSH_HOST are required." >&2
    return 0
  fi

  local ssh_target="${ssh_user}@${ssh_host}"

  ssh -p "$ssh_port" "$ssh_target" bash -s <<EOF
set -euo pipefail

sudo apt-get update
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt-get update
sudo apt-get install -y caddy

sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
{
  email ${caddy_email}
}

${api_domain} {
  encode gzip zstd
  reverse_proxy 127.0.0.1:8000
}

${app_domain} {
  encode gzip zstd
  reverse_proxy 127.0.0.1:8501
}
CADDY

sudo systemctl restart caddy
sudo systemctl enable caddy
sudo systemctl --no-pager status caddy | cat
EOF
}

case "$MODE" in
  docker)
    "$ROOT_DIR/scripts/deploy/deploy.sh" vps-docker
    ;;
  manual)
    "$ROOT_DIR/scripts/deploy/deploy.sh" vps-manual
    ;;
  *)
    echo "Usage: $0 [docker|manual] [env-file]" >&2
    exit 1
    ;;
esac

if [ "${PROVISION_CADDY:-false}" = "true" ]; then
  configure_caddy
fi
