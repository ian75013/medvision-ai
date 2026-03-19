# README_DVC — Mon guide DVC pour MedVision

## 1. DVC, ce n’est pas Docker

Note shell:

* **PowerShell** = Windows
* **Ubuntu/bash** = Linux (et WSL)

J’ai mis les deux formats de commandes dans ce document.

DVC signifie **Data Version Control**.

Son rôle est de :

* versionner les **datasets**, **modèles** et **artefacts ML** trop volumineux pour Git ;
* décrire un **pipeline reproductible** de machine learning ;
* relancer uniquement les étapes nécessaires quand un fichier, un paramètre ou du code change.

**DVC n’est pas** :

* un conteneur,
* un orchestrateur de conteneurs,
* un remplacement de Docker.

### Différence simple

* **Docker** = isole l’environnement d’exécution.
* **DVC** = versionne les données/modèles et pilote le pipeline ML.
* **Git** = versionne le code et les petits fichiers texte.

En pratique, je peux utiliser les trois ensemble :

* Git pour le code,
* DVC pour les données et les pipelines,
* Docker pour exécuter le projet dans un environnement stable.

---

## 2. À quoi me sert DVC dans MedVision

Dans MedVision, DVC me sert à rendre mon workflow ML **reproductible**.

Au lieu de lancer manuellement plusieurs scripts dans le désordre, je décris un pipeline comme :

1. télécharger le dataset,
2. préparer les données,
3. entraîner le modèle,
4. évaluer le modèle,
5. produire les métriques et artefacts.

DVC mémorise :

* quelles commandes ont été exécutées,
* quels fichiers entrent dans chaque étape,
* quels fichiers sont produits,
* quels paramètres influencent le résultat.

Ainsi, si je modifie seulement un hyperparamètre ou un script d’entraînement, DVC peut relancer seulement les étapes impactées.

---

## 3. Comment penser DVC correctement

Il faut voir DVC comme un **chef d’orchestre du pipeline ML**, pas comme un runtime.

### Ce que DVC suit

DVC sait suivre :

* des fichiers de données,
* des dossiers de datasets,
* des modèles entraînés,
* des métriques JSON,
* des graphiques,
* des paramètres déclarés dans `params.yaml`,
* des étapes dans `dvc.yaml`.

### Ce que DVC ne fait pas tout seul

DVC ne :

* télécharge pas magiquement Kaggle sans script,
* n’entraîne pas de modèle sans commande définie,
* ne remplace pas FastAPI, MLflow ou Docker,
* ne stocke pas les données dans Git.

---

## 4. Les fichiers DVC importants

### `dvc.yaml`

C’est le fichier qui décrit le pipeline.

Exemple conceptuel :

```yaml
stages:
  download:
    cmd: python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
    deps:
      - src/data/download_brain_mri_dataset.py
      - configs/brain_tumor_mri.yaml
    outs:
      - data/raw/brain_tumor_mri

  train:
    cmd: python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml
    deps:
      - src/training/train_brain_mri.py
      - src/utils/dataset_multiclass.py
      - data/raw/brain_tumor_mri
      - configs/brain_tumor_mri.yaml
    params:
      - train
      - dataset
      - model
    outs:
      - artifacts/models/brain_tumor_classifier.keras
    metrics:
      - artifacts/reports/metrics.json
```

### `params.yaml`

C’est le fichier de paramètres :

```yaml
dataset:
  name: brain_tumor_mri
  image_size: 224
  batch_size: 16

model:
  name: optimized
  num_classes: 4

train:
  epochs: 10
  learning_rate: 0.0001
```

DVC peut détecter qu’un paramètre a changé et relancer la bonne étape.

### `.dvc/config`

Contient la configuration DVC locale du projet, notamment le remote éventuel.

### `.dvc/cache`

Cache local DVC. Il ne faut généralement pas le manipuler à la main.

---

## 5. Mon flux de travail DVC pour MedVision

### Mode A — Local simple, sans remote cloud

C’est le meilleur point de départ.

J’utilise DVC uniquement pour :

* pipeline,
* cache local,
* reproductibilité,
* versionnement logique des artefacts.

Commandes typiques :

**PowerShell**

```powershell
git init
dvc init
```

**Ubuntu/bash**

```bash
git init
dvc init
```

Puis :

**PowerShell**

```powershell
dvc repro
```

**Ubuntu/bash**

```bash
dvc repro
```

Cela exécute les stages de `dvc.yaml` dans le bon ordre. DVC relance uniquement ce qui est nécessaire. Le comportement général de `dvc repro` et des pipelines reproductibles est au cœur du fonctionnement officiel de DVC. ([dvc.org](https://dvc.org/))

### Mode B — Avec remote partagé

Quand je veux sauvegarder le dataset, les modèles et artefacts ailleurs que sur mon disque local, j’ajoute un remote DVC.

Exemple avec un dossier local partagé :

**PowerShell**

```powershell
dvc remote add -d localremote C:\chemin\vers\stockage-dvc
```

**Ubuntu/bash**

```bash
dvc remote add -d localremote /chemin/vers/stockage-dvc
```

Exemple avec S3 :

**PowerShell**

```powershell
dvc remote add -d s3remote s3://mon-bucket/medvision-dvc
```

**Ubuntu/bash**

```bash
dvc remote add -d s3remote s3://mon-bucket/medvision-dvc
```

Ensuite :

**PowerShell**

```powershell
dvc push
dvc pull
```

**Ubuntu/bash**

```bash
dvc push
dvc pull
```

Le principe du remote DVC est de laisser Git stocker seulement les métadonnées, tandis que les gros artefacts sont stockés à part. C’est un usage central de DVC. ([dvc.org](https://dvc.org))

---

## 6. Workflow concret pour mon projet MedVision

### Étape 1 — Installer les outils

**PowerShell**

```powershell
pip install dvc
pip install kaggle
```

**Ubuntu/bash**

```bash
pip install dvc
pip install kaggle
```

Selon le remote choisi, je peux avoir besoin d’un extra, par exemple :

**PowerShell**

```powershell
pip install "dvc[s3]"
```

**Ubuntu/bash**

```bash
pip install "dvc[s3]"
```

### Étape 2 — Initialiser DVC dans le repo

**PowerShell**

```powershell
dvc init
```

**Ubuntu/bash**

```bash
dvc init
```

### Étape 3 — Préparer l’auth Kaggle

Créer :

* sous Linux/macOS : `~/.kaggle/kaggle.json`
* sous Windows : `%USERPROFILE%\.kaggle\kaggle.json`

Puis sécuriser les droits si nécessaire.

### Étape 4 — Lancer le pipeline

**PowerShell**

```powershell
dvc repro
```

**Ubuntu/bash**

```bash
dvc repro
```

### Étape 5 — Visualiser les changements

**PowerShell**

```powershell
dvc status
dvc params diff
git status
```

**Ubuntu/bash**

```bash
dvc status
dvc params diff
git status
```

### Étape 6 — Sauvegarder les artefacts DVC

**PowerShell**

```powershell
git add dvc.yaml params.yaml .gitignore .dvcignore
git commit -m "Add DVC pipeline for brain MRI"
```

**Ubuntu/bash**

```bash
git add dvc.yaml params.yaml .gitignore .dvcignore
git commit -m "Add DVC pipeline for brain MRI"
```

Si un remote est configuré :

**PowerShell**

```powershell
dvc push
```

**Ubuntu/bash**

```bash
dvc push
```

---

## 7. Ce que je recommande de changer dans mon usage de DVC

Je recommande de l’utiliser de cette manière, plus propre :

### Stage 1 — `download`

Télécharger le dataset Kaggle brut.

Sortie typique :

* `data/raw/brain_tumor_mri/`

### Stage 2 — `prepare`

Vérifier la structure, éventuellement nettoyer, générer un résumé du dataset.

Sortie typique :

* `data/interim/brain_tumor_mri/`
* `artifacts/reports/dataset_summary.json`

### Stage 3 — `train`

Entraîner le modèle.

Sortie typique :

* `artifacts/models/brain_tumor_classifier.keras`
* `artifacts/reports/history.json`

### Stage 4 — `evaluate`

Calculer les métriques détaillées.

Sortie typique :

* `artifacts/reports/metrics.json`
* `artifacts/reports/classification_report.txt`
* `artifacts/plots/confusion_matrix.png`

### Stage 5 — `package` ou `export`

Préparer le modèle pour FastAPI ou Streamlit.

Sortie typique :

* `deploy/model/`
* `artifacts/reports/model_card.md`

Cette séparation est plus lisible qu’un gros stage unique et correspond mieux à l’esprit des pipelines DVC. ([dvc.org](https://dvc.org/blog/jupyter-notebook-dvc-pipeline/))

---

## 8. Exemple de `dvc.yaml` recommandé

```yaml
stages:
  download:
    cmd: python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
    deps:
      - src/data/download_brain_mri_dataset.py
      - configs/brain_tumor_mri.yaml
    outs:
      - data/raw/brain_tumor_mri

  prepare:
    cmd: python -m src.data.prepare_brain_mri_dataset --config configs/brain_tumor_mri.yaml
    deps:
      - src/data/prepare_brain_mri_dataset.py
      - data/raw/brain_tumor_mri
      - configs/brain_tumor_mri.yaml
    outs:
      - data/interim/brain_tumor_mri
    metrics:
      - artifacts/reports/dataset_summary.json

  train:
    cmd: python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized
    deps:
      - src/training/train_brain_mri.py
      - src/utils/dataset_multiclass.py
      - data/interim/brain_tumor_mri
      - configs/brain_tumor_mri.yaml
    params:
      - dataset
      - model
      - train
    outs:
      - artifacts/models/brain_tumor_classifier.keras
      - artifacts/reports/history.json

  evaluate:
    cmd: python -m src.evaluation.metrics_multiclass --config configs/brain_tumor_mri.yaml
    deps:
      - src/evaluation/metrics_multiclass.py
      - artifacts/models/brain_tumor_classifier.keras
      - data/interim/brain_tumor_mri
      - configs/brain_tumor_mri.yaml
    metrics:
      - artifacts/reports/metrics.json
    plots:
      - artifacts/plots/confusion_matrix.png
```

---

## 9. Où intervient Docker alors ?

Docker intervient si je veux figer l’environnement.

Par exemple :

* un conteneur `training` avec Python, TensorFlow, DVC, Kaggle ;
* un conteneur `api` pour FastAPI ;
* un conteneur `ui` pour Streamlit ;
* un conteneur `mlflow` pour le tracking.

Mais **DVC reste au niveau du pipeline ML**, même si j’exécute les commandes DVC depuis un conteneur.

Exemple :

**PowerShell**

```powershell
docker compose run --rm training dvc repro
```

**Ubuntu/bash**

```bash
docker compose run --rm training dvc repro
```

Ici :

* Docker fournit l’environnement,
* DVC exécute le pipeline.

---

## 10. Les commandes DVC les plus utiles

### Exécuter le pipeline

```bash
dvc repro
```

### Voir ce qui a changé

```bash
dvc status
```

### Tester un changement de paramètres comme expérience

```bash
dvc exp run
```

La fonctionnalité `dvc exp run` est au cœur des expériences DVC modernes. ([dvc.org](https://dvc.org/blog/experiment-refs/))

### Lister les expériences

```bash
dvc exp show
```

### Envoyer les données et modèles sur le remote

```bash
dvc push
```

### Récupérer les données et modèles depuis le remote

```bash
dvc pull
```

### Voir les différences de paramètres

```bash
dvc params diff
```

---

## 11. Ce que je versionne avec Git et ce que je laisse à DVC

### Dans Git

Je commit :

* code source,
* `dvc.yaml`,
* `params.yaml`,
* configs YAML,
* petits rapports texte,
* README.

### Dans DVC

Je laisse DVC gérer :

* datasets bruts,
* datasets préparés,
* checkpoints lourds,
* modèles finaux,
* gros artefacts.

C’est exactement l’idée centrale de DVC : Git garde les références, DVC garde la gestion des fichiers lourds. ([dvc.org](https://dvc.org/))

---

## 12. Bon usage concret dans mon cas

Pour MedVision, je me fixe cette progression :

### Au début

* **pas de remote cloud obligatoire**,
* DVC en local pour apprendre,
* pipeline simple `download -> prepare -> train -> evaluate`.

### Ensuite

* j’ajoute un remote local ou S3/MinIO,
* je branche MLflow,
* j’ajoute `dvc exp run` pour comparer les hyperparamètres,
* éventuellement, j’exécute le tout depuis Docker.

### Plus tard

* j’intègre BraTS,
* j’ajoute la segmentation,
* je versionne des modèles plus lourds,
* je trace les données préparées et les checkpoints.

---

## 13. Résumé ultra simple

Si je devais retenir une seule image mentale :

* **Git** suit le code.
* **DVC** suit les données, les modèles et le pipeline.
* **Docker** fige l’environnement d’exécution.

Donc non, **DVC n’est pas un conteneur Docker**.
C’est plutôt un **gestionnaire de pipeline et de versionnement pour les projets ML**. ([dvc.org](https://dvc.org/))

---

## 14. Commandes minimales à retenir

**PowerShell**

```powershell
dvc init
dvc repro
dvc status
dvc exp run
dvc exp show
dvc push
dvc pull
```

**Ubuntu/bash**

```bash
dvc init
dvc repro
dvc status
dvc exp run
dvc exp show
dvc push
dvc pull
```

---

## 15. Mon cap final pour MedVision

Je considère DVC comme :

* la colonne vertébrale du pipeline d’entraînement,
* pas comme un substitut à Docker,
* et pas comme un outil d’inférence.

Le plus propre pour mon projet est :

* **DVC** pour la préparation/entraînement/évaluation,
* **MLflow** pour le suivi d’expériences,
* **FastAPI** pour l’inférence,
* **Docker** pour l’environnement et le déploiement.
