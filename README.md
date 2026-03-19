# MedVision AI

MedVision AI est un projet de **computer vision appliqué à l'imagerie médicale** conçu comme base de travail **Machine Learning Engineer / Computer Vision Engineer**.

Le dépôt contient désormais deux axes :
- une base historique orientée classification 2D,
- une branche de travail pour la **classification de tumeurs cérébrales sur MRI**,
- une couche **MLOps légère** avec **DVC** et un remote possible sur **AWS S3** ou **Google Drive**.

---

## Sommaire

1. [Objectif](#objectif)
2. [Stack](#stack)
3. [Structure](#structure)
4. [Installation rapide](#installation-rapide)
5. [Dataset Kaggle Brain Tumor MRI](#dataset-kaggle-brain-tumor-mri)
6. [Lancer l'entraînement](#lancer-lentraînement)
7. [Utiliser DVC](#utiliser-dvc)
8. [Remote DVC sur S3 avec Terraform](#remote-dvc-sur-s3-avec-terraform)
9. [Remote DVC sur Google Drive](#remote-dvc-sur-google-drive)
10. [API FastAPI](#api-fastapi)
11. [Streamlit](#streamlit)
12. [Docker](#docker)
13. [Tests](#tests)

---

## Objectif

Ce projet montre une manière de structurer un projet ML appliqué :
- entraînement,
- évaluation,
- versionnement des données et artefacts,
- API,
- UI légère,
- documentation.

---

## Stack

- Python
- TensorFlow / Keras
- FastAPI
- Streamlit
- MLflow
- DVC
- Kaggle CLI
- Terraform pour le bucket S3 DVC

---

## Structure

```text
medvision-ai-feature-mri-test/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── notebooks/
├── requirements/
├── scripts/
├── src/
├── terraform/
│   └── aws_dvc_remote/
├── tests/
├── dvc.yaml
├── params.yaml
├── README.md
├── README_DVC.md
└── requirements.txt
```

---

## Installation rapide

### Linux / macOS

```bash
git clone <ton-repo>
cd medvision-ai-feature-mri-test
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
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

Le `requirements.txt` installe maintenant aussi :

```txt
dvc[s3,gdrive]==3.59.2
```

Cela active directement les remotes **S3** et **Google Drive**.

---

## Dataset Kaggle Brain Tumor MRI

Le projet utilise pour la branche MRI le dataset Kaggle :

```text
masoudnickparvar/brain-tumor-mri-dataset
```

Le téléchargement est effectué par le script :

```text
src/data/download_brain_mri_dataset.py
```

### Configuration Kaggle

Place ton fichier `kaggle.json` ici :

- Linux/macOS : `~/.kaggle/kaggle.json`
- Windows : `%USERPROFILE%\.kaggle\kaggle.json`

---

## Lancer l'entraînement

### Sans DVC

```bash
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized --epochs 10
```

---

## Utiliser DVC

### Initialisation

```bash
dvc init
```

### Pipeline complet

```bash
dvc repro
```

### Vérifier les changements

```bash
dvc status
dvc params diff
```

### Expériences

```bash
dvc exp run
dvc exp show
```

Le détail est expliqué dans [README_DVC.md](README_DVC.md).

---

## Remote DVC sur S3 avec Terraform

Le dépôt contient un module Terraform dans :

```text
terraform/aws_dvc_remote/
```

### Étapes

```bash
cd terraform/aws_dvc_remote
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Terraform crée un bucket S3 privé avec :
- blocage de l'accès public,
- chiffrement serveur,
- versioning optionnel.

Ensuite, depuis la racine du projet :

```bash
dvc remote add -d s3remote s3://<nom-du-bucket>
dvc remote modify s3remote region eu-west-3
dvc push
```

---

## Remote DVC sur Google Drive

Tu peux aussi stocker les artefacts DVC dans un dossier Google Drive.

### Ajouter le remote

```bash
dvc remote add -d gdriveremote gdrive://<folder-id>
```

Puis :

```bash
dvc push
dvc pull
```

C'est pratique pour un POC personnel. Pour un projet plus solide ou partagé, S3 reste le meilleur choix.

---

## API FastAPI

```bash
uvicorn src.api.main:app --reload
```

---

## Streamlit

```bash
streamlit run streamlit_app.py
```

---

## Docker

Le projet contient un environnement Docker pour exécuter l'application, mais **Docker et DVC ont des rôles différents** :
- Docker exécute le projet dans un environnement figé,
- DVC suit les données, les modèles et le pipeline.

---

## Tests

```bash
pytest -q
```

---

## Notes utiles

- `dvc.yaml` décrit le pipeline ML.
- `params.yaml` contient les hyperparamètres suivis par DVC.
- `README_DVC.md` explique le fonctionnement détaillé de DVC dans le projet.
- `terraform/aws_dvc_remote/` permet de provisionner un bucket S3 pour le remote DVC.
