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

PROXY_PROVIDER="${PROXY_PROVIDER:-caddy}"

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  echo "[deploy-ovh] Warning: local sudo is not required. Run this script as your normal user." >&2
fi

if [ -z "${SUDO_PASSWORD:-}" ] && [ "${ASK_SUDO_PASSWORD:-false}" = "true" ]; then
  if [ -t 0 ]; then
    read -r -s -p "Remote sudo password for ${SSH_USER:-user}@${SSH_HOST:-host}: " SUDO_PASSWORD
    echo
    export SUDO_PASSWORD
  else
    echo "ASK_SUDO_PASSWORD=true but no interactive terminal available." >&2
    exit 1
  fi
fi

configure_caddy() {
  local ssh_user="${SSH_USER:-}"
  local ssh_host="${SSH_HOST:-}"
  local ssh_port="${SSH_PORT:-22}"
  local api_domain="${API_DOMAIN:-}"
  local app_domain="${APP_DOMAIN:-}"
  local caddy_email="${CADDY_EMAIL:-}"
  local sudo_password="${SUDO_PASSWORD:-}"
  local api_host_port="${API_HOST_PORT:-18000}"
  local streamlit_host_port="${STREAMLIT_HOST_PORT:-18501}"

  if [ -z "$api_domain" ] || [ -z "$app_domain" ] || [ -z "$caddy_email" ]; then
    echo "Caddy provisioning skipped: set API_DOMAIN, APP_DOMAIN and CADDY_EMAIL in env file." >&2
    return 0
  fi

  if [ -z "$ssh_user" ] || [ -z "$ssh_host" ]; then
    echo "Caddy provisioning skipped: SSH_USER/SSH_HOST are required." >&2
    return 0
  fi

  local ssh_target="${ssh_user}@${ssh_host}"

  ssh -p "$ssh_port" "$ssh_target" "SUDO_PASSWORD=$(printf %q "$sudo_password") bash -s" <<EOF
set -euo pipefail

run_sudo() {
  if [ -n "\${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "\$SUDO_PASSWORD" | sudo -S -p '' "\$@"
  else
    sudo "\$@"
  fi
}

run_sudo apt-get update
run_sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | run_sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | run_sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
run_sudo apt-get update
run_sudo apt-get install -y caddy

run_sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
{
  email ${caddy_email}
}

${api_domain} {
  encode gzip zstd
  reverse_proxy 127.0.0.1:${api_host_port}
}

${app_domain} {
  encode gzip zstd
  reverse_proxy 127.0.0.1:${streamlit_host_port}
}
CADDY

run_sudo caddy validate --config /etc/caddy/Caddyfile
run_sudo systemctl enable caddy
run_sudo systemctl reload caddy || run_sudo systemctl restart caddy
run_sudo systemctl --no-pager status caddy | cat
EOF
}

configure_nginx() {
  local ssh_user="${SSH_USER:-}"
  local ssh_host="${SSH_HOST:-}"
  local ssh_port="${SSH_PORT:-22}"
  local api_domain="${API_DOMAIN:-}"
  local app_domain="${APP_DOMAIN:-}"
  local sudo_password="${SUDO_PASSWORD:-}"
  local api_host_port="${API_HOST_PORT:-18000}"
  local streamlit_host_port="${STREAMLIT_HOST_PORT:-18501}"
  local nginx_site_name="${NGINX_SITE_NAME:-medvision-ai}"

  if [ -z "$api_domain" ] || [ -z "$app_domain" ]; then
    echo "Nginx provisioning skipped: set API_DOMAIN and APP_DOMAIN in env file." >&2
    return 0
  fi

  if [ -z "$ssh_user" ] || [ -z "$ssh_host" ]; then
    echo "Nginx provisioning skipped: SSH_USER/SSH_HOST are required." >&2
    return 0
  fi

  local ssh_target="${ssh_user}@${ssh_host}"

  ssh -p "$ssh_port" "$ssh_target" "SUDO_PASSWORD=$(printf %q "$sudo_password") bash -s" <<EOF
set -euo pipefail

run_sudo() {
  if [ -n "\${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "\$SUDO_PASSWORD" | sudo -S -p '' "\$@"
  else
    sudo "\$@"
  fi
}

run_sudo apt-get update
run_sudo apt-get install -y nginx

run_sudo tee /etc/nginx/sites-available/${nginx_site_name}.conf >/dev/null <<NGINX
server {
    listen 80;
    server_name ${api_domain};

    location / {
        proxy_pass http://127.0.0.1:${api_host_port};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}

server {
    listen 80;
    server_name ${app_domain};

    location / {
        proxy_pass http://127.0.0.1:${streamlit_host_port};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
NGINX

run_sudo ln -sf /etc/nginx/sites-available/${nginx_site_name}.conf /etc/nginx/sites-enabled/${nginx_site_name}.conf
run_sudo nginx -t
run_sudo systemctl enable nginx
run_sudo systemctl reload nginx || run_sudo systemctl restart nginx
run_sudo systemctl --no-pager status nginx | cat
EOF
}

case "$MODE" in
  docker)
    bash "$ROOT_DIR/scripts/deploy/deploy.sh" vps-docker
    ;;
  manual)
    bash "$ROOT_DIR/scripts/deploy/deploy.sh" vps-manual
    ;;
  proxy|caddy)
    case "$PROXY_PROVIDER" in
      caddy)
        configure_caddy
        ;;
      nginx)
        configure_nginx
        ;;
      none|off)
        echo "Proxy provisioning disabled (PROXY_PROVIDER=${PROXY_PROVIDER})." >&2
        ;;
      *)
        echo "Unsupported PROXY_PROVIDER: ${PROXY_PROVIDER}. Use caddy, nginx, none." >&2
        exit 1
        ;;
    esac
    exit 0
    ;;
  *)
    echo "Usage: $0 [docker|manual|proxy|caddy] [env-file]" >&2
    exit 1
    ;;
esac

if [ "${PROVISION_CADDY:-false}" = "true" ]; then
  case "$PROXY_PROVIDER" in
    caddy)
      configure_caddy
      ;;
    nginx)
      configure_nginx
      ;;
    none|off)
      :
      ;;
    *)
      echo "Unsupported PROXY_PROVIDER: ${PROXY_PROVIDER}. Use caddy, nginx, none." >&2
      exit 1
      ;;
  esac
fi
