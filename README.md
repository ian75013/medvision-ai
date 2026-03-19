# MedVision AI

MedVision AI est un projet de **computer vision appliqué à l'imagerie médicale** avec une orientation **ML Engineer / MLOps**.

Le dépôt mélange aujourd'hui deux lignes de travail :
- une **base historique** orientée classification 2D de radiographies thoraciques ;
- une **branche MRI** plus récente pour la classification multi-classe de tumeurs cérébrales à partir d'un dataset Kaggle ;
- une couche **MLOps** autour de **MLflow**, **DVC** et d'un remote DVC sur **AWS S3** ou **Google Drive** ;
- un début de packaging produit avec **FastAPI**, **Streamlit**, **Docker Compose** et **Terraform**.

## Ce qu'il faut savoir tout de suite

La partie la plus cohérente et la plus directement exploitable dans ce zip est aujourd'hui la branche **Brain MRI** :
- `src/data/download_brain_mri_dataset.py`
- `src/training/train_brain_mri.py`
- `src/utils/dataset_multiclass.py`
- `src/evaluation/metrics_multiclass.py`
- `configs/brain_tumor_mri.yaml`
- `dvc.yaml`
- `params.yaml`

La partie **historique chest X-ray** est encore documentée et partiellement présente, mais ce zip n'inclut pas les modules `src.models.baseline_model` et `src.models.optimized_model` attendus par certains scripts historiques. Pour cette raison, le **workflow recommandé** dans l'état actuel du dépôt est le workflow **Brain MRI**. Voir `docs/KNOWN_GAPS.md`.

## Sommaire

1. [Vision du projet](#vision-du-projet)
2. [Stack technique](#stack-technique)
3. [Architecture du dépôt](#architecture-du-dépôt)
4. [Installation locale](#installation-locale)
5. [Données et datasets](#données-et-datasets)
6. [Lancer le workflow Brain MRI](#lancer-le-workflow-brain-mri)
7. [MLflow](#mlflow)
8. [DVC](#dvc)
9. [Terraform et remote S3](#terraform-et-remote-s3)
10. [Google Drive comme remote DVC](#google-drive-comme-remote-dvc)
11. [Docker Compose](#docker-compose)
12. [API et UI](#api-et-ui)
13. [Tests](#tests)
14. [Documentation complémentaire](#documentation-complémentaire)
15. [État actuel et limites](#état-actuel-et-limites)

## Vision du projet

L'objectif du dépôt est de montrer une manière de structurer un projet de vision médicale avec :
- ingestion de dataset ;
- entraînement ;
- évaluation ;
- suivi d'expériences ;
- versionnement des données et artefacts ;
- exposition via API ;
- interface légère de démonstration.

## Stack technique

- **Python**
- **TensorFlow / Keras** pour les pipelines actuels du dépôt
- **FastAPI** pour l'inférence HTTP
- **Streamlit** pour une UI simple
- **MLflow** pour le tracking d'expériences et l'UI de runs
- **DVC** pour les pipelines ML, les métriques et les remotes de données
- **Kaggle CLI** pour télécharger les datasets publics
- **Terraform** pour provisionner un bucket S3 dédié au remote DVC
- **Docker Compose** pour lancer rapidement les services principaux

## Architecture du dépôt

```text
medvision-ai-feature-mri-test/
├── .github/workflows/          # CI GitHub Actions
├── configs/                    # Configurations YAML des pipelines
├── data/                       # Données brutes / préparées
├── docker/                     # Dockerfile de base
├── docs/                       # Documentation projet, architecture, MLOps
├── notebooks/                  # Notebooks de démonstration
├── requirements/               # Requirements complémentaires
├── scripts/                    # Scripts shell / utilitaires
├── src/
│   ├── api/                    # FastAPI
│   ├── data/                   # Téléchargement datasets
│   ├── dataio/                 # Lecture NIfTI / I/O médical
│   ├── datasets/               # Datasets MRI de démonstration
│   ├── evaluation/             # Métriques et rapports
│   ├── inference/              # Chargement modèle et prédiction
│   ├── preprocessing/          # Prétraitements images / MRI
│   ├── training/               # Scripts d'entraînement
│   └── utils/                  # Config, paths, logging, datasets helpers
├── terraform/aws_dvc_remote/   # Infra S3 pour DVC
├── tests/                      # Tests unitaires simples
├── dvc.yaml                    # Pipeline DVC
├── params.yaml                 # Paramètres DVC
├── docker-compose.yml          # MLflow + API + Streamlit
├── README_DVC.md               # Guide DVC détaillé
└── README_MRI_SPRINTS.md       # Notes sur la branche MRI
```

Une vue d'architecture plus détaillée est fournie dans `docs/ARCHITECTURE.md`.

## Installation locale

### Prérequis

- Python **3.10 à 3.12** recommandé
- Git
- accès Kaggle si tu veux télécharger les datasets automatiquement
- optionnel : Docker / Docker Compose
- optionnel : Terraform si tu veux créer le bucket S3 DVC

### Linux / macOS

```bash
git clone <ton-repo>
cd medvision-ai-feature-mri-test
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
git clone <ton-repo>
cd medvision-ai-feature-mri-test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dépendances MLOps importantes

Le `requirements.txt` inclut déjà :

```txt
mlflow==2.20.3
dvc[s3,gdrive]==3.59.2
kaggle==1.6.17
```

Cela permet d'utiliser :
- le tracking local MLflow ;
- un remote DVC sur **S3** ;
- un remote DVC sur **Google Drive**.

## Données et datasets

### Dataset Brain MRI recommandé pour ce dépôt

Le workflow MRI s'appuie sur le dataset Kaggle :

```text
masoudnickparvar/brain-tumor-mri-dataset
```

Le téléchargement est géré par :

```text
src/data/download_brain_mri_dataset.py
```

### Authentification Kaggle

Place `kaggle.json` ici :
- Linux/macOS : `~/.kaggle/kaggle.json`
- Windows : `%USERPROFILE%\.kaggle\kaggle.json`

## Lancer le workflow Brain MRI

### Étape 1 — Télécharger le dataset

```bash
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
```

### Étape 2 — Entraîner le modèle

```bash
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized --epochs 10
```

### Artéfacts produits

- modèle Keras dans `artifacts/models/`
- métriques JSON dans `artifacts/reports/`
- classification report texte
- confusion matrix image
- historique d'entraînement JSON
- runs MLflow dans `mlruns/`

## MLflow

Le dépôt utilise **MLflow** dans les scripts d'entraînement pour :
- créer une expérience ;
- loguer les hyperparamètres ;
- loguer les métriques ;
- loguer les artefacts produits par l'entraînement.

Par défaut, le tracking URI est local :

```yaml
mlflow_tracking_uri: file:./mlruns
```

### Lancer l'UI MLflow en local

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Puis ouvrir :

```text
http://localhost:5000
```

### Lancer MLflow avec Docker Compose

```bash
docker compose up mlflow
```

## DVC

DVC est utilisé pour :
- décrire un pipeline ML reproductible ;
- versionner les datasets, métriques et artefacts lourds ;
- brancher un remote sur S3 ou Google Drive.

### Commandes principales

```bash
dvc init
dvc repro
dvc status
dvc exp run
dvc exp show
dvc push
dvc pull
```

Le guide détaillé est dans `README_DVC.md`.

## Terraform et remote S3

Le module Terraform fourni dans `terraform/aws_dvc_remote/` permet de créer un bucket S3 dédié à DVC.

### Utilisation rapide

```bash
cd terraform/aws_dvc_remote
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Ensuite, à la racine du projet :

```bash
dvc remote add -d s3remote s3://<bucket-name>
dvc remote modify s3remote region eu-west-3
dvc push
```

Voir aussi `terraform/aws_dvc_remote/README.md`.

## Google Drive comme remote DVC

Tu peux aussi utiliser un dossier Google Drive comme remote DVC.

### Exemple

```bash
dvc remote add -d gdriveremote gdrive://<folder-id>
dvc push
```

Les détails et précautions sont décrits dans `README_DVC.md`.

## Docker Compose

Le dépôt contient un `docker-compose.yml` avec trois services :
- `mlflow`
- `api`
- `streamlit`

### Lancer l'ensemble

```bash
docker compose up --build
```

### Endpoints attendus

- MLflow : `http://localhost:5000`
- API : `http://localhost:8000`
- Streamlit : `http://localhost:8501`

## API et UI

### API FastAPI

L'API actuelle expose :
- `GET /health`
- `POST /predict`

Le service est défini dans `src/api/main.py`.

### Streamlit

L'interface simple est définie dans `streamlit_app.py`.

> Remarque : l'API et la UI historiques sont encore orientées **classification chest X-ray binaire**. Pour la branche Brain MRI, la partie la plus aboutie aujourd'hui concerne surtout l'entraînement, l'évaluation et la couche MLOps. Cela est documenté dans `docs/KNOWN_GAPS.md`.

## Tests

Lancer :

```bash
pytest -q
```

## Documentation complémentaire

Le dossier `docs/` contient maintenant :
- `ARCHITECTURE.md` : architecture logique et flux de bout en bout ;
- `COMPONENTS.md` : rôle de chaque dossier et fichier principal ;
- `MLOPS_GUIDE.md` : MLflow, DVC, Terraform, remotes ;
- `DATASETS.md` : datasets, conventions et structure des données ;
- `OPERATIONS.md` : commandes utiles et runbook ;
- `KNOWN_GAPS.md` : incohérences et points à corriger dans le dépôt.

## État actuel et limites

Le dépôt est une bonne base de travail **portfolio / démonstration / refactor ML Engineer**, mais il y a encore des écarts à combler pour en faire un projet totalement homogène :
- coexistence d'un flux historique chest X-ray et d'un flux MRI ;
- API/UI encore centrées sur le flux historique ;
- références à des modules de modèles non présents dans ce zip pour une partie des anciens scripts ;
- branche MRI encore en 2D Kaggle, pas encore sur un pipeline IRM volumique clinique type BraTS + MONAI.

Ces points sont explicités dans `docs/KNOWN_GAPS.md`.
