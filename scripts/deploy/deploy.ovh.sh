#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-docker}"
ENV_FILE="${2:-$ROOT_DIR/scripts/deploy/env.ovh.example}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

XTRACE_WAS_ENABLED=0
if [[ $- == *x* ]]; then
  XTRACE_WAS_ENABLED=1
  set +x
fi

set -a
source "$ENV_FILE"
set +a

PROXY_PROVIDER="${PROXY_PROVIDER:-nginx}"

if [ "$XTRACE_WAS_ENABLED" -eq 1 ] && [ -n "${SUDO_PASSWORD:-}" ]; then
  echo "Refusing to run with bash -x while SUDO_PASSWORD is set. Disable xtrace or unset SUDO_PASSWORD first." >&2
  exit 1
fi

if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
  set -x
fi

if [ "${EUID:-$(id -u)}" -eq 0 ]; then
  echo "[deploy-ovh] Warning: local sudo is not required. Run this script as your normal user." >&2
fi

if [ "${ASK_SUDO_PASSWORD:-false}" = "true" ]; then
  export USE_REMOTE_SUDO_PROMPT=true
  SUDO_PASSWORD=""
else
  export USE_REMOTE_SUDO_PROMPT=false
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
  local ssh_tty_args=()

  if [ -z "$sudo_password" ]; then
    ssh_tty_args=(-tt)
  fi

  if [ -z "$api_domain" ] || [ -z "$app_domain" ] || [ -z "$caddy_email" ]; then
    echo "Caddy provisioning skipped: set API_DOMAIN, APP_DOMAIN and CADDY_EMAIL in env file." >&2
    return 0
  fi

  if [ -z "$ssh_user" ] || [ -z "$ssh_host" ]; then
    echo "Caddy provisioning skipped: SSH_USER/SSH_HOST are required." >&2
    return 0
  fi

  local ssh_target="${ssh_user}@${ssh_host}"

  # Create a temporary installation script
  local tmp_local_script
  tmp_local_script="$(mktemp)"
  local tmp_remote_script="/tmp/medvision-caddy-install.sh"

  cat > "$tmp_local_script" <<'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

API_DOMAIN="${API_DOMAIN}"
APP_DOMAIN="${APP_DOMAIN}"
CADDY_EMAIL="${CADDY_EMAIL}"
API_HOST_PORT="${API_HOST_PORT}"
STREAMLIT_HOST_PORT="${STREAMLIT_HOST_PORT}"

sudo apt-get update
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt-get update
sudo apt-get install -y caddy

sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
{
  email ${CADDY_EMAIL}
}

${API_DOMAIN} {
  encode gzip zstd
  reverse_proxy 127.0.0.1:${API_HOST_PORT}
}

${APP_DOMAIN} {
  encode gzip zstd
  reverse_proxy 127.0.0.1:${STREAMLIT_HOST_PORT}
}
CADDY

sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable caddy
sudo systemctl reload caddy || sudo systemctl restart caddy
sudo systemctl --no-pager status caddy | cat
SCRIPT_EOF

  chmod +x "$tmp_local_script"

  if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
    set +x
  fi

  scp -P "$ssh_port" "$tmp_local_script" "${ssh_target}:${tmp_remote_script}"

  ssh "${ssh_tty_args[@]}" -p "$ssh_port" "$ssh_target" \
    "API_DOMAIN=$(printf %q "$api_domain") APP_DOMAIN=$(printf %q "$app_domain") CADDY_EMAIL=$(printf %q "$caddy_email") API_HOST_PORT=$(printf %q "$api_host_port") STREAMLIT_HOST_PORT=$(printf %q "$streamlit_host_port") bash ${tmp_remote_script}"

  rm -f "$tmp_local_script"
  if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
    set -x
  fi
}

configure_nginx() {
  local ssh_user="${SSH_USER:-}"
  local ssh_host="${SSH_HOST:-}"
  local ssh_port="${SSH_PORT:-22}"
  local api_domain="${API_DOMAIN:-}"
  local app_domain="${APP_DOMAIN:-}"
  local app_dir="${APP_DIR:-/opt/medvision-ai}"
  local api_host_port="${API_HOST_PORT:-18000}"
  local streamlit_host_port="${STREAMLIT_HOST_PORT:-18501}"
  local nginx_ssl_cert="${NGINX_SSL_CERT:-}"
  local nginx_ssl_key="${NGINX_SSL_KEY:-}"
  local api_ssl_cert="${API_SSL_CERT:-${nginx_ssl_cert}}"
  local api_ssl_key="${API_SSL_KEY:-${nginx_ssl_key}}"
  local app_ssl_cert="${APP_SSL_CERT:-${nginx_ssl_cert}}"
  local app_ssl_key="${APP_SSL_KEY:-${nginx_ssl_key}}"

  if [ -z "$api_domain" ] || [ -z "$app_domain" ]; then
    echo "Nginx provisioning skipped: set API_DOMAIN and APP_DOMAIN in env file." >&2
    return 0
  fi

  if [ -z "$ssh_user" ] || [ -z "$ssh_host" ]; then
    echo "Nginx provisioning skipped: SSH_USER/SSH_HOST are required." >&2
    return 0
  fi

  local ssh_target="${ssh_user}@${ssh_host}"
  local tmp_local_conf
  tmp_local_conf="$(mktemp)"
  local nginx_template_http="$ROOT_DIR/deploy/reverse-proxy/nginx/medvision.http.template.conf"
  local nginx_template_https="$ROOT_DIR/deploy/reverse-proxy/nginx/medvision.https.template.conf"
  local nginx_template_selected

  if [ -n "$api_ssl_cert" ] && [ -n "$api_ssl_key" ] && [ -n "$app_ssl_cert" ] && [ -n "$app_ssl_key" ]; then
    nginx_template_selected="$nginx_template_https"
  else
    nginx_template_selected="$nginx_template_http"
    echo "[deploy-ovh] Nginx config generated in HTTP-only mode for the Docker service." >&2
  fi

  [ -f "$nginx_template_selected" ] || {
    rm -f "$tmp_local_conf"
    echo "Missing nginx template: $nginx_template_selected" >&2
    return 1
  }

  cp "$nginx_template_selected" "$tmp_local_conf"
  sed -i \
    -e "s|__API_DOMAIN__|${api_domain}|g" \
    -e "s|__APP_DOMAIN__|${app_domain}|g" \
    -e "s|__API_PORT__|${api_host_port}|g" \
    -e "s|__APP_PORT__|${streamlit_host_port}|g" \
    -e "s|__API_SSL_CERT__|${api_ssl_cert}|g" \
    -e "s|__API_SSL_KEY__|${api_ssl_key}|g" \
    -e "s|__APP_SSL_CERT__|${app_ssl_cert}|g" \
    -e "s|__APP_SSL_KEY__|${app_ssl_key}|g" \
    "$tmp_local_conf"

  if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
    set +x
  fi

  ssh -p "$ssh_port" "$ssh_target" "mkdir -p $(printf %q "$app_dir")/deploy/reverse-proxy/nginx"
  scp -P "$ssh_port" "$tmp_local_conf" "${ssh_target}:$(printf %q "$app_dir")/deploy/reverse-proxy/nginx/medvision.conf"

  echo "[deploy-ovh] Nginx Docker config uploaded to ${app_dir}/deploy/reverse-proxy/nginx/medvision.conf" >&2

  rm -f "$tmp_local_conf"
  if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
    set -x
  fi
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
