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

if [ -n "${SUDO_PASSWORD:-}" ] && [ "${ALLOW_PLAINTEXT_SUDO_PASSWORD:-false}" != "true" ]; then
  echo "[deploy-ovh][error] Refusing plaintext SUDO_PASSWORD from env file." >&2
  echo "[deploy-ovh][error] Use ASK_SUDO_PASSWORD=true (recommended), or set ALLOW_PLAINTEXT_SUDO_PASSWORD=true explicitly." >&2
  exit 1
fi

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
  local sudo_password="${SUDO_PASSWORD:-}"
  local api_host_port="${API_HOST_PORT:-18000}"
  local streamlit_host_port="${STREAMLIT_HOST_PORT:-18501}"
  local letsencrypt_email="${LETSENCRYPT_EMAIL:-${CADDY_EMAIL:-}}"
  local auto_certbot_once="${AUTO_CERTBOT_ONCE:-true}"
  local apache_ssl_fallback_domain="${APACHE_SSL_FALLBACK_DOMAIN:-}"
  local apache_api_ssl_cert="${APACHE_API_SSL_CERT:-/etc/letsencrypt/live/${api_domain}/fullchain.pem}"
  local apache_api_ssl_key="${APACHE_API_SSL_KEY:-/etc/letsencrypt/live/${api_domain}/privkey.pem}"
  local apache_app_ssl_cert="${APACHE_APP_SSL_CERT:-/etc/letsencrypt/live/${app_domain}/fullchain.pem}"
  local apache_app_ssl_key="${APACHE_APP_SSL_KEY:-/etc/letsencrypt/live/${app_domain}/privkey.pem}"
  local ssh_tty_args=()

  if [ -z "$sudo_password" ]; then
    ssh_tty_args=(-tt)
  fi

  if [ -z "$api_domain" ] || [ -z "$app_domain" ]; then
    echo "Apache provisioning skipped: set API_DOMAIN and APP_DOMAIN in env file." >&2
    return 0
  fi

  if [ -z "$ssh_user" ] || [ -z "$ssh_host" ]; then
    echo "Apache provisioning skipped: SSH_USER/SSH_HOST are required." >&2
    return 0
  fi

  local ssh_target="${ssh_user}@${ssh_host}"

  if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
    set +x
  fi

  # Expose public Apache vhosts directly to the Docker API and Streamlit services.
  local tmp_local_apache_conf
  tmp_local_apache_conf="$(mktemp)"
  local tmp_remote_apache_conf="/tmp/medvision-apache-site.conf"
  cat > "$tmp_local_apache_conf" <<EOF
<VirtualHost *:80>
  ServerName ${api_domain}
  ProxyPreserveHost On
  ProxyRequests Off
  ProxyTimeout 3600
  ProxyPass / http://127.0.0.1:${api_host_port}/ retry=0 timeout=3600
  ProxyPassReverse / http://127.0.0.1:${api_host_port}/
  RequestHeader set X-Forwarded-Proto "http"
</VirtualHost>

<VirtualHost *:80>
  ServerName ${app_domain}
  ProxyPreserveHost On
  ProxyRequests Off
  ProxyTimeout 3600
  ProxyPass /_stcore/stream ws://127.0.0.1:${streamlit_host_port}/_stcore/stream retry=0 timeout=3600
  ProxyPassReverse /_stcore/stream ws://127.0.0.1:${streamlit_host_port}/_stcore/stream
  ProxyPass / http://127.0.0.1:${streamlit_host_port}/ retry=0 timeout=3600
  ProxyPassReverse / http://127.0.0.1:${streamlit_host_port}/
  RequestHeader set X-Forwarded-Proto "http"
</VirtualHost>
EOF

  local tmp_local_apache_script
  tmp_local_apache_script="$(mktemp)"
  local tmp_remote_apache_script="/tmp/medvision-apache-proxy-install.sh"
  cat > "$tmp_local_apache_script" <<'SCRIPT_EOF'
#!/usr/bin/env bash
set -euo pipefail

run_sudo() {
  if [ "${USE_REMOTE_SUDO_PROMPT:-false}" = "true" ]; then
    sudo -v
    sudo "$@"
  elif [ -n "${SUDO_PASSWORD:-}" ]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

if ! command -v apache2ctl >/dev/null 2>&1; then
  echo "[deploy-ovh] Apache not detected on VPS; skipping Apache bridge setup." >&2
  exit 0
fi

run_sudo install -m 644 "$TMP_REMOTE_APACHE_CONF" /etc/apache2/sites-available/medvision-ai.conf
run_sudo a2enmod proxy proxy_http proxy_wstunnel headers ssl rewrite >/dev/null
run_sudo a2disconf medvision-ai-proxy >/dev/null 2>&1 || true
run_sudo a2ensite medvision-ai >/dev/null
run_sudo apache2ctl configtest
run_sudo systemctl reload apache2

resolve_cert_pair() {
  local cert_path="$1"
  local key_path="$2"
  local fallback_domain="$3"
  local host_domain="$4"

  resolve_live_dir_pair() {
    local candidate_dir="$1"
    if [ -s "$candidate_dir/fullchain.pem" ] && [ -s "$candidate_dir/privkey.pem" ]; then
      printf '%s|%s' "$candidate_dir/fullchain.pem" "$candidate_dir/privkey.pem"
      return 0
    fi
    return 1
  }

  resolve_domain_glob_pair() {
    local candidate_domain="$1"
    local live_dir
    for live_dir in /etc/letsencrypt/live/"${candidate_domain}"*; do
      [ -d "$live_dir" ] || continue
      if resolve_live_dir_pair "$live_dir"; then
        return 0
      fi
    done
    return 1
  }

  if [ -s "$cert_path" ] && [ -s "$key_path" ]; then
    printf '%s|%s' "$cert_path" "$key_path"
    return 0
  fi

  if resolve_domain_glob_pair "$host_domain"; then
    return 0
  fi

  if [ -n "$fallback_domain" ] && resolve_domain_glob_pair "$fallback_domain"; then
    return 0
  fi

  local parent1="${host_domain#*.}"
  local parent2="${parent1#*.}"
  for candidate in "$parent1" "$parent2"; do
    if [ -n "$candidate" ] && [ "$candidate" != "$host_domain" ] && resolve_domain_glob_pair "$candidate"; then
      return 0
    fi
  done

  return 1
}

api_pair=""
app_pair=""
if api_pair="$(resolve_cert_pair "$APACHE_API_SSL_CERT" "$APACHE_API_SSL_KEY" "${APACHE_SSL_FALLBACK_DOMAIN:-}" "$API_DOMAIN")"; then
  api_cert_ok=true
else
  api_cert_ok=false
fi
if app_pair="$(resolve_cert_pair "$APACHE_APP_SSL_CERT" "$APACHE_APP_SSL_KEY" "${APACHE_SSL_FALLBACK_DOMAIN:-}" "$APP_DOMAIN")"; then
  app_cert_ok=true
else
  app_cert_ok=false
fi

if [ "$api_cert_ok" != "true" ] || [ "$app_cert_ok" != "true" ]; then
  if [ "${AUTO_CERTBOT_ONCE:-true}" = "true" ]; then
    marker_file="/etc/letsencrypt/.medvision-certbot-bootstrap.done"
    if [ ! -f "$marker_file" ]; then
      if [ -n "${LETSENCRYPT_EMAIL:-}" ]; then
        run_sudo apt-get update
        run_sudo apt-get install -y certbot python3-certbot-apache
        run_sudo certbot certonly --apache -d "$API_DOMAIN" -m "$LETSENCRYPT_EMAIL" --agree-tos --no-eff-email -n || true
        run_sudo certbot certonly --apache -d "$APP_DOMAIN" -m "$LETSENCRYPT_EMAIL" --agree-tos --no-eff-email -n || true
      else
        echo "[deploy-ovh] LETSENCRYPT_EMAIL is empty; skipping certbot bootstrap." >&2
      fi
      run_sudo touch "$marker_file"
    fi
  fi
fi

if [ "$api_cert_ok" = "true" ] && [ "$app_cert_ok" = "true" ]; then
  api_cert_resolved="${api_pair%%|*}"
  api_key_resolved="${api_pair##*|}"
  app_cert_resolved="${app_pair%%|*}"
  app_key_resolved="${app_pair##*|}"
  ssl_tmp="$(mktemp)"
  cat > "$ssl_tmp" <<EOF_SSL
<IfModule mod_ssl.c>
<VirtualHost *:443>
  ServerName ${API_DOMAIN}
  SSLEngine on
  SSLCertificateFile ${api_cert_resolved}
  SSLCertificateKeyFile ${api_key_resolved}
  Include /etc/letsencrypt/options-ssl-apache.conf
  ProxyPreserveHost On
  ProxyRequests Off
  SSLProxyEngine On
  ProxyTimeout 3600
  ProxyPass / http://127.0.0.1:${API_HOST_PORT}/ retry=0 timeout=3600
  ProxyPassReverse / http://127.0.0.1:${API_HOST_PORT}/
  RequestHeader set X-Forwarded-Proto "https"
</VirtualHost>

<VirtualHost *:443>
  ServerName ${APP_DOMAIN}
  SSLEngine on
  SSLCertificateFile ${app_cert_resolved}
  SSLCertificateKeyFile ${app_key_resolved}
  Include /etc/letsencrypt/options-ssl-apache.conf
  ProxyPreserveHost On
  ProxyRequests Off
  SSLProxyEngine On
  ProxyTimeout 3600
  ProxyPass /_stcore/stream ws://127.0.0.1:${STREAMLIT_HOST_PORT}/_stcore/stream retry=0 timeout=3600
  ProxyPassReverse /_stcore/stream ws://127.0.0.1:${STREAMLIT_HOST_PORT}/_stcore/stream
  ProxyPass / http://127.0.0.1:${STREAMLIT_HOST_PORT}/ retry=0 timeout=3600
  ProxyPassReverse / http://127.0.0.1:${STREAMLIT_HOST_PORT}/
  RequestHeader set X-Forwarded-Proto "https"
</VirtualHost>
</IfModule>
EOF_SSL
  run_sudo install -m 644 "$ssl_tmp" /etc/apache2/sites-available/medvision-ai-ssl.conf
  rm -f "$ssl_tmp"
  run_sudo a2ensite medvision-ai-ssl >/dev/null
  run_sudo apache2ctl configtest
  run_sudo systemctl reload apache2
  echo "[deploy-ovh] Apache SSL site enabled for MedVision domains." >&2
  echo "[deploy-ovh] SSL certs used: API=${api_cert_resolved} APP=${app_cert_resolved}" >&2
else
  run_sudo a2dissite medvision-ai-ssl >/dev/null 2>&1 || true
  echo "[deploy-ovh] Apache SSL site skipped: certificates are not available yet." >&2
fi
SCRIPT_EOF

  chmod +x "$tmp_local_apache_conf" "$tmp_local_apache_script"
  scp -P "$ssh_port" "$tmp_local_apache_conf" "${ssh_target}:${tmp_remote_apache_conf}"
  scp -P "$ssh_port" "$tmp_local_apache_script" "${ssh_target}:${tmp_remote_apache_script}"
  ssh "${ssh_tty_args[@]}" -p "$ssh_port" "$ssh_target" \
    "SUDO_PASSWORD=$(printf %q "$sudo_password") USE_REMOTE_SUDO_PROMPT=$(printf %q "${USE_REMOTE_SUDO_PROMPT:-false}") TMP_REMOTE_APACHE_CONF=$(printf %q "$tmp_remote_apache_conf") API_DOMAIN=$(printf %q "$api_domain") APP_DOMAIN=$(printf %q "$app_domain") API_HOST_PORT=$(printf %q "$api_host_port") STREAMLIT_HOST_PORT=$(printf %q "$streamlit_host_port") APACHE_API_SSL_CERT=$(printf %q "$apache_api_ssl_cert") APACHE_API_SSL_KEY=$(printf %q "$apache_api_ssl_key") APACHE_APP_SSL_CERT=$(printf %q "$apache_app_ssl_cert") APACHE_APP_SSL_KEY=$(printf %q "$apache_app_ssl_key") APACHE_SSL_FALLBACK_DOMAIN=$(printf %q "$apache_ssl_fallback_domain") LETSENCRYPT_EMAIL=$(printf %q "$letsencrypt_email") AUTO_CERTBOT_ONCE=$(printf %q "$auto_certbot_once") bash ${tmp_remote_apache_script}"
  echo "[deploy-ovh] Apache reverse proxy configured: ${api_domain} -> 127.0.0.1:${api_host_port}, ${app_domain} -> 127.0.0.1:${streamlit_host_port}" >&2

  rm -f "$tmp_local_apache_conf" "$tmp_local_apache_script"
  if [ "$XTRACE_WAS_ENABLED" -eq 1 ]; then
    set -x
  fi
}

provision_proxy_if_enabled() {
  if [ "${PROVISION_CADDY:-false}" != "true" ]; then
    return 0
  fi

  case "$PROXY_PROVIDER" in
    caddy)
      configure_caddy
      ;;
    apache|nginx)
      configure_nginx
      ;;
    none|off)
      echo "Proxy provisioning disabled (PROXY_PROVIDER=${PROXY_PROVIDER})." >&2
      ;;
    *)
      echo "Unsupported PROXY_PROVIDER: ${PROXY_PROVIDER}. Use apache, caddy, nginx, none." >&2
      exit 1
      ;;
  esac
}

case "$MODE" in
  docker)
    provision_proxy_if_enabled
    bash "$ROOT_DIR/scripts/deploy/deploy.sh" vps-docker
    ;;
  manual)
    provision_proxy_if_enabled
    bash "$ROOT_DIR/scripts/deploy/deploy.sh" vps-manual
    ;;
  proxy|caddy)
    provision_proxy_if_enabled
    exit 0
    ;;
  *)
    echo "Usage: $0 [docker|manual|proxy|caddy] [env-file]" >&2
    exit 1
    ;;
esac
