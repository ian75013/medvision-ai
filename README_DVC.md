# README_DVC — DVC dans MedVision

## DVC, ce que c'est ici

Dans ce projet, **DVC** sert à :
- décrire un pipeline reproductible ;
- suivre les données et artefacts trop lourds pour Git ;
- brancher un remote S3 ou Google Drive ;
- comparer des expériences via `dvc exp`.

DVC **n'est pas** un conteneur Docker. Docker isole l'environnement ; DVC orchestre le pipeline de données et de modèles.

## Articulation avec MLflow

Dans MedVision :
- **DVC** suit le pipeline, les paramètres, les métriques et les sorties ;
- **MLflow** suit les runs d'entraînement, les métriques et les artefacts ;
- **Terraform** peut créer l'infrastructure S3 du remote DVC ;
- **Docker Compose** facilite le lancement des services.

## Pipeline DVC du dépôt

Le pipeline est défini dans `dvc.yaml`.

Dans ce zip, le pipeline canonique est orienté **Brain MRI Kaggle** :
- `download_data`
- `train_brain_mri`

Il peut être enrichi plus tard avec des étapes `prepare`, `evaluate` et `package`.

## Fichiers à connaître

- `dvc.yaml` : définition des stages
- `params.yaml` : paramètres pilotés par DVC
- `.dvc/config` : remotes DVC
- `artifacts/` : sorties du pipeline
- `mlruns/` : suivi MLflow

## Démarrage rapide

### 1. Initialiser DVC

```bash
dvc init
```

### 2. Rejouer le pipeline

```bash
dvc repro
```

### 3. Vérifier ce qui a changé

```bash
dvc status
dvc params diff
```

### 4. Lancer une expérience DVC

```bash
dvc exp run
```

### 5. Voir les expériences

```bash
dvc exp show
```

## Remote S3

Après provisionnement du bucket Terraform :

```bash
dvc remote add -d s3remote s3://<bucket-name>
dvc remote modify s3remote region eu-west-3
dvc push
```

## Remote Google Drive

```bash
dvc remote add -d gdriveremote gdrive://<folder-id>
dvc push
```

## Bon usage recommandé dans ce dépôt

### Pour l'apprentissage / local
- Git pour le code
- DVC pour les données et sorties
- MLflow pour les runs
- stockage local des données et de `mlruns`

### Pour une version plus MLOps
- remote DVC sur S3 ou Google Drive
- MLflow persistant
- exécution via Docker Compose
- CI/CD ensuite

## Commandes utiles

```bash
dvc repro
dvc status
dvc exp run
dvc exp show
dvc push
dvc pull
```

## Limite importante

DVC décrit ici surtout le **workflow Brain MRI**. La partie historique chest X-ray existe encore dans le repo, mais ce n'est pas le chemin le plus cohérent à utiliser dans l'état actuel du zip.
