# MedVision AI — Index de la documentation

**Projet :** Pipeline de segmentation et classification IRM avec MLOps intégré.  
**Stack :** FastAPI · Streamlit · MLflow · DVC · PyTorch · Docker · k3s/OVH

---

## Racine du projet

| Fichier | Description |
|---------|-------------|
| [../README.md](../README.md) | Vue d'ensemble, quick start |
| [../README_MRI_SPRINTS.md](../README_MRI_SPRINTS.md) | Sprints de développement IRM, historique des itérations |
| [../README_DVC.md](../README_DVC.md) | Guide rapide DVC (versionning données et modèles) |
| [../NOTICE.md](../NOTICE.md) | Mentions légales et licences |

---

## Documentation (`docs/`)

### Architecture & Conception

| Fichier | Description |
|---------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture globale, composants, flux de données |
| [COMPONENTS.md](COMPONENTS.md) | Description détaillée de chaque composant |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | Guide complet du projet, conventions, organisation |

### Données & Datasets

| Fichier | Description |
|---------|-------------|
| [DATASETS.md](DATASETS.md) | Sources de données IRM, formats, téléchargement |
| [DVC_GUIDE.md](DVC_GUIDE.md) | Versionnage des données avec DVC, remotes, pipelines |

### Machine Learning & MLOps

| Fichier | Description |
|---------|-------------|
| [MLOPS_GUIDE.md](MLOPS_GUIDE.md) | Guide MLOps : training, tracking, serving |
| [MLFLOW_EXPERIMENT_PLAN_2026-04-07.md](MLFLOW_EXPERIMENT_PLAN_2026-04-07.md) | Plan d'expériences MLflow du 7 avril 2026 |
| [TRANSFER_LEARNING_GUIDE.md](TRANSFER_LEARNING_GUIDE.md) | Transfer learning : fine-tuning, stratégies, résultats |
| [CI_STRATEGY.md](CI_STRATEGY.md) | Stratégie CI/CD pour les pipelines ML |

### Interfaces utilisateur

| Fichier | Description |
|---------|-------------|
| [SEGMENTATION_UI_GUIDE.md](SEGMENTATION_UI_GUIDE.md) | Guide de l'interface Streamlit de segmentation |
| [FASTAPI_STREAMLIT_ALIGNMENT.md](FASTAPI_STREAMLIT_ALIGNMENT.md) | Alignement API/UI : contrats, modèles Pydantic |

### Déploiement & Infrastructure

| Fichier | Description |
|---------|-------------|
| [DEPLOYMENT_OPTIONS.md](DEPLOYMENT_OPTIONS.md) | Options de déploiement (local, Docker, cloud) |
| [DEPLOYMENT_PLAYBOOK_AWS_AZURE_K3S_OVH.md](DEPLOYMENT_PLAYBOOK_AWS_AZURE_K3S_OVH.md) | Playbook détaillé : AWS, Azure, k3s OVH |
| [DOCKER_WORKFLOW_GUIDE.md](DOCKER_WORKFLOW_GUIDE.md) | Workflow Docker : build, run, compose |
| [REVERSE_PROXY_DNS.md](REVERSE_PROXY_DNS.md) | Configuration reverse proxy et DNS |
| [OPERATIONS.md](OPERATIONS.md) | Opérations courantes, monitoring, maintenance |

### Lacunes connues

| Fichier | Description |
|---------|-------------|
| [KNOWN_GAPS.md](KNOWN_GAPS.md) | Limitations connues, travaux en cours, dette technique |
| [AGENT_DEPLOYMENT_GUARDRAILS.md](AGENT_DEPLOYMENT_GUARDRAILS.md) | Règles de sécurité pour déploiements automatisés par agents |

---

## Terraform (`terraform/`)

| Fichier | Description |
|---------|-------------|
| [../terraform/aws_dvc_remote/README.md](../terraform/aws_dvc_remote/README.md) | Terraform pour remote DVC sur AWS S3 |

---

## Guardrails (`.guardrails/rules/`)

| Fichier | Description |
|---------|-------------|
| [../.guardrails/rules/01-core-principles.md](../.guardrails/rules/01-core-principles.md) | Principes fondamentaux : sécurité, réversibilité, testabilité |
| [../.guardrails/rules/02-engineering-standards.md](../.guardrails/rules/02-engineering-standards.md) | Standards de code et revue |
| [../.guardrails/rules/03-security-privacy.md](../.guardrails/rules/03-security-privacy.md) | Sécurité et confidentialité |
| [../.guardrails/rules/04-testing-quality-gates.md](../.guardrails/rules/04-testing-quality-gates.md) | Couches de tests requises |
| [../.guardrails/rules/05-release-change-management.md](../.guardrails/rules/05-release-change-management.md) | Gestion des releases |
| [../.guardrails/rules/06-observability-operations.md](../.guardrails/rules/06-observability-operations.md) | Observabilité et opérations |
| [../.guardrails/rules/07-documentation-knowledge.md](../.guardrails/rules/07-documentation-knowledge.md) | Exigences de documentation |
