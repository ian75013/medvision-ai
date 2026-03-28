# Reverse proxy + DNS registrar notes

This document gives a practical setup for exposing MedVision services behind a reverse proxy, and what DNS records to configure at your registrar.

## 1. Recommended public hostnames

Use two hostnames:

- api.<your-domain>
- app.<your-domain>

Suggested mapping:

- api.<your-domain> -> FastAPI service
- app.<your-domain> -> Streamlit service

## 2. Reverse proxy options

### 2.1 Caddy (recommended for automatic TLS)

Template file:

- deploy/reverse-proxy/caddy/Caddyfile

Example installation (OVH/Hetzner VPS):

```bash
sudo apt-get update
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy
```

Deploy config:

```bash
sudo cp deploy/reverse-proxy/caddy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl restart caddy
sudo systemctl status caddy --no-pager
```

### 2.2 Nginx (if you already standardize on Nginx)

Template file:

- deploy/reverse-proxy/nginx/medvision.conf

Deploy config:

```bash
sudo cp deploy/reverse-proxy/nginx/medvision.conf /etc/nginx/sites-available/medvision.conf
sudo ln -sf /etc/nginx/sites-available/medvision.conf /etc/nginx/sites-enabled/medvision.conf
sudo nginx -t
sudo systemctl reload nginx
```

For TLS with Nginx, use certbot or your existing ACME flow.

## 3. DNS records to configure at registrar

Set these records in your DNS zone.

### 3.1 OVH / Hetzner VPS (public IP)

- Type: A
  Name: api
  Value: <VPS_IPV4>
  TTL: 300
- Type: A
  Name: app
  Value: <VPS_IPV4>
  TTL: 300
- Optional Type: AAAA
  Name: api
  Value: <VPS_IPV6>
  TTL: 300
- Optional Type: AAAA
  Name: app
  Value: <VPS_IPV6>
  TTL: 300

### 3.2 AWS App Runner

Use CNAME records to App Runner default domains.

- Type: CNAME
  Name: api
  Value: <aws-apprunner-api-default-domain>
  TTL: 300
- Type: CNAME
  Name: app
  Value: <aws-apprunner-streamlit-default-domain>
  TTL: 300

Then map custom domains in App Runner for each service.

### 3.3 Azure Container Apps

Use CNAME records to Container Apps FQDN.

- Type: CNAME
  Name: api
  Value: <azure-api-containerapp-fqdn>
  TTL: 300
- Type: CNAME
  Name: app
  Value: <azure-streamlit-containerapp-fqdn>
  TTL: 300

If apex domain is needed, use ALIAS/ANAME (registrar dependent).

### 3.4 k3s / Kubernetes with ingress

If ingress has a public LoadBalancer IP:

- Type: A
  Name: api
  Value: <INGRESS_PUBLIC_IPV4>
  TTL: 300
- Type: A
  Name: app
  Value: <INGRESS_PUBLIC_IPV4>
  TTL: 300

If ingress provides hostname only:

- Type: CNAME
  Name: api
  Value: <INGRESS_HOSTNAME>
  TTL: 300
- Type: CNAME
  Name: app
  Value: <INGRESS_HOSTNAME>
  TTL: 300

## 4. DNS propagation and validation

Validate name resolution:

```bash
dig +short api.<your-domain>
dig +short app.<your-domain>
```

Validate HTTP endpoint:

```bash
curl -I https://api.<your-domain>/health
curl -I https://app.<your-domain>/
```

## 5. Security notes

- Keep API and Streamlit behind TLS only.
- Add authentication in front of Streamlit for internet exposure.
- Restrict inbound traffic to 80/443 on proxy host; keep 8000/8501 private.

## 6. Beginner procedure (OVH VPS, one command)

Yes, the intended flow is:

1. From your local machine, run one deployment command.
2. The script connects in SSH to the VPS.
3. It clones/updates the GitHub repo on the VPS.
4. It starts API + Streamlit.
5. Optional: it installs Caddy and configures HTTPS automatically.

Minimal steps:

```bash
cp scripts/deploy/env.ovh.example .env.ovh
```

Edit `.env.ovh` with your VPS + domain values, then launch:

```bash
scripts/deploy/deploy.ovh.sh docker .env.ovh
```

To enable automatic Caddy setup, set in `.env.ovh`:

```bash
PROVISION_CADDY=true
API_DOMAIN=api.ton-domaine.tld
APP_DOMAIN=app.ton-domaine.tld
CADDY_EMAIL=ton-email@domaine.tld
```

Before running with Caddy, create DNS entries at your registrar:

- A record: `api` -> `<VPS_IPV4>`
- A record: `app` -> `<VPS_IPV4>`

Then wait for DNS propagation and retry command if needed.
