# README_DVC — Utiliser DVC dans MedVision

## 1. Ce que fait DVC

DVC (**Data Version Control**) ne remplace ni Git ni Docker.

- **Git** versionne le code.
- **DVC** versionne les gros artefacts ML : datasets, modèles, métriques, pipelines.
- **Docker** fige l'environnement d'exécution.

Dans MedVision, DVC sert surtout à :
- télécharger ou préparer le dataset,
- rejouer l'entraînement,
- stocker les artefacts hors de Git,
- brancher un remote local, S3 ou Google Drive.

## 2. Dépendances Python

Le projet installe maintenant DVC directement via `requirements.txt` :

```bash
pip install -r requirements.txt
```

La ligne importante est :

```txt
dvc[s3,gdrive]==3.59.2
```

Elle active :
- le support **AWS S3** pour un bucket DVC,
- le support **Google Drive** pour un remote DVC sur Drive.

## 3. Fichiers DVC du projet

- `dvc.yaml` : pipeline DVC
- `params.yaml` : hyperparamètres suivis par DVC
- `.dvc/` : configuration locale DVC
- `.dvc/cache/` : cache local

## 4. Pipeline actuel

Le pipeline DVC de MedVision contient quatre étapes (option B, multi-problèmes) :

### `download_chest_xray`
Télécharge le dataset Kaggle Chest X-ray dans :

```text
data/raw/chest_xray/
```

### `train_chest_xray`
Entraîne le modèle Chest X-ray (optimized) et produit :

```text
artifacts/models/optimized_model.keras
artifacts/reports/optimized_metrics.json
artifacts/reports/optimized_history.json
artifacts/reports/optimized_classification_report.txt
artifacts/reports/optimized_confusion_matrix.png
```

### `download_brain_mri`
Télécharge le dataset Kaggle Brain Tumor MRI dans :

```text
data/raw/brain_tumor_mri/
```

### `train_brain_mri`
Entraîne le modèle à partir du dataset téléchargé et produit :

```text
artifacts/models/brain_mri_optimized.keras
artifacts/reports/brain_mri_metrics.json
artifacts/reports/brain_mri_optimized_history.json
artifacts/reports/brain_mri_optimized_classification_report.txt
artifacts/reports/brain_mri_optimized_confusion_matrix.png
```

## 5. Commandes de base

### Initialiser DVC

```bash
dvc init
```

### Exécuter tout le pipeline

```bash
dvc repro
```

### Voir ce qui a changé

```bash
dvc status
dvc params diff
```

### Expériences DVC

```bash
dvc exp run
dvc exp show
```

## 6. Remote local simple

Exemple avec un dossier local :

```bash
dvc remote add -d localremote /chemin/vers/medvision-dvc-storage
```

Puis :

```bash
dvc push
dvc pull
```

## 7. Remote AWS S3

Le projet contient maintenant Terraform dans :

```text
terraform/aws_dvc_remote/
```

### 7.1 Créer le bucket avec Terraform

```bash
cd terraform/aws_dvc_remote
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

### 7.2 Configurer DVC avec le bucket

Depuis la racine du projet :

```bash
dvc remote add -d s3remote s3://<nom-du-bucket>
dvc remote modify s3remote region eu-west-3
```

Si tu utilises des variables d'environnement AWS :

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-west-3
```

Ensuite :

```bash
dvc push
dvc pull
```

## 8. Remote Google Drive

Google Drive est utile pour un usage personnel ou un petit POC.

### Ajouter le remote

```bash
dvc remote add -d gdriveremote gdrive://<folder-id>
```

Le `<folder-id>` correspond à l'identifiant du dossier Google Drive cible.

### Authentification

Au premier accès, DVC peut demander une authentification Google.

Ensuite :

```bash
dvc push
dvc pull
```

## 9. Exemple de workflow recommandé

### Une première fois

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
dvc init
```

### Télécharger et entraîner

```bash
dvc repro
```

### Exécuter seulement une composante (si besoin)

```bash
dvc repro train_chest_xray
dvc repro train_brain_mri
```

### Sauvegarder les artefacts sur le remote

```bash
dvc push
```

## 10. Ce qu'il faut retenir

La bonne image mentale est :

- **Git** = code
- **DVC** = données + modèles + pipeline ML
- **Docker** = environnement

Donc : **DVC n'est pas un conteneur**. C'est l'outil qui structure et versionne ton pipeline ML.
