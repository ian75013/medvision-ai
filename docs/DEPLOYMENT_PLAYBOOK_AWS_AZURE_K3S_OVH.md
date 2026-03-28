# Playbook cible: AWS, Azure, k3s perso, OVH VPS

Ce playbook fournit les commandes concretes et minimales pour deployer MedVision AI sur tes cibles prioritaires.

Scripts utilises:

- scripts/deploy/deploy.aws.sh
- scripts/deploy/deploy.azure.sh
- scripts/deploy/deploy.k3s.sh
- scripts/deploy/deploy.ovh.sh

## 1) AWS (App Runner)

1. Copier le template et renseigner les valeurs:

```bash
cp scripts/deploy/env.aws.example .env.aws
```

2. Deployer:

```bash
scripts/deploy/deploy.aws.sh .env.aws
```

Prerequis locaux:

- aws CLI configure (credentials + region)
- docker

## 2) Azure (Container Apps)

1. Copier le template et renseigner les valeurs:

```bash
cp scripts/deploy/env.azure.example .env.azure
```

2. Deployer:

```bash
scripts/deploy/deploy.azure.sh .env.azure
```

Prerequis locaux:

- az login
- extension containerapp disponible
- docker

## 3) k3s (cluster perso)

1. Copier le template et renseigner les images:

```bash
cp scripts/deploy/env.k3s.example .env.k3s
```

2. Deployer:

```bash
scripts/deploy/deploy.k3s.sh .env.k3s
```

3. Verifier:

```bash
kubectl -n medvision get pods,svc
kubectl -n medvision port-forward svc/medvision-api 8000:8000
kubectl -n medvision port-forward svc/medvision-streamlit 8501:8501
```

Notes k3s:

- les images doivent etre accessibles par les noeuds k3s
- si registre prive: configurer imagePullSecrets dans les manifests
- ingress optionnel dans deploy/k8s/base/ingress.yaml

## 4) OVH VPS

Tu as deux modes:

- docker (recommande, mode prod): scripts/deploy/deploy.ovh.sh docker .env.ovh
- manual (uv + systemd): scripts/deploy/deploy.ovh.sh manual .env.ovh

1. Copier le template et renseigner la connexion:

```bash
cp scripts/deploy/env.ovh.example .env.ovh
```

Un exemple complet (valeurs fictives) est aussi disponible:

- scripts/deploy/env.ovh.demo.example

2. Deployer:

```bash
scripts/deploy/deploy.ovh.sh docker .env.ovh
# ou
scripts/deploy/deploy.ovh.sh manual .env.ovh
```

Note: le mode docker utilise automatiquement l'overlay de production (docker-compose.prod.yml).

### Option "une commande" avec Caddy (TLS auto)

Si tu veux que le reverse proxy TLS soit configure automatiquement:

1. Dans `.env.ovh`, mettre:

```bash
PROVISION_CADDY=true
API_DOMAIN=api.ton-domaine.tld
APP_DOMAIN=app.ton-domaine.tld
CADDY_EMAIL=ton-email@domaine.tld
```

2. Lancer la meme commande:

```bash
scripts/deploy/deploy.ovh.sh docker .env.ovh
```

Important: les enregistrements DNS `api` et `app` doivent deja pointer vers l'IP du VPS avant d'activer Caddy, sinon le certificat TLS ne pourra pas etre emis.

3. Validation:

```bash
curl http://<IP_VPS>:8000/health
```

## 5) Notes production

- API et Streamlit sont deployes en services separes.
- prevoir un reverse proxy TLS (Caddy/Nginx/Traefik) devant les endpoints publics.
- proteger Streamlit par authentification si expose publiquement.
- externaliser artifacts/models et reports vers object storage pour reduire la taille des images.

DNS + reverse proxy details:

- docs/REVERSE_PROXY_DNS.md
