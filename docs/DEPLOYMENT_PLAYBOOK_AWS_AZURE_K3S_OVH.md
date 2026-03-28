# Playbook cible: AWS, Azure, k3s perso, OVH VPS

Ce playbook fournit les commandes concretes et minimales pour deployer MedVision AI sur tes cibles prioritaires.

Scripts utilises:

- scripts/deploy/deploy.aws.sh
- scripts/deploy/deploy.azure.sh
- scripts/deploy/deploy.k3s.sh
- scripts/deploy/deploy.ovh.sh

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
bash scripts/deploy/deploy.ovh.sh docker .env.ovh
# ou
bash scripts/deploy/deploy.ovh.sh manual .env.ovh
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
bash scripts/deploy/deploy.ovh.sh docker .env.ovh
```

Important: les enregistrements DNS `api` et `app` doivent deja pointer vers l'IP du VPS avant d'activer Caddy, sinon le certificat TLS ne pourra pas etre emis.

### Si sudo demande un mot de passe

Tu peux le passer via `.env.ovh`:

```bash
SUDO_PASSWORD=ton_mot_de_passe_sudo
```

Le script utilisera ce mot de passe pour les commandes sudo distantes.

Attention: c'est sensible. Evite de versionner ce fichier.
