# MedVision AI

MedVision AI est un projet de **computer vision appliqué à l'imagerie médicale** conçu comme un **template réaliste de projet Machine Learning Engineer / Computer Vision Engineer**.

Il montre un pipeline de bout en bout autour d'un cas d'usage simple mais crédible :

- **classification binaire de radiographies thoraciques**
- **entraînement d'un modèle baseline**
- **optimisation par transfer learning**
- **suivi des expériences avec MLflow**
- **serving via FastAPI**
- **démo via Streamlit**
- **explicabilité locale via Grad-CAM**
- **tests et structure de projet propre**

L'objectif n'est pas seulement d'obtenir une prédiction, mais de montrer une façon de travailler proche d'un projet **MedTech / AI product / applied ML**.

---

# Table des matières

- [1. Pourquoi ce projet ?](#1-pourquoi-ce-projet-)
- [2. Ce que démontre techniquement le projet](#2-ce-que-démontre-techniquement-le-projet)
- [3. Architecture du repository](#3-architecture-du-repository)
- [4. Dataset utilisé](#4-dataset-utilisé)
- [5. Téléchargement automatique du dataset](#5-téléchargement-automatique-du-dataset)
- [6. Configuration Kaggle](#6-configuration-kaggle)
- [7. Installation du projet](#7-installation-du-projet)
- [8. Entraînement des modèles](#8-entraînement-des-modèles)
- [9. Suivi des expériences avec MLflow](#9-suivi-des-expériences-avec-mlflow)
- [10. API FastAPI](#10-api-fastapi)
- [11. Interface Streamlit](#11-interface-streamlit)
- [12. Notebook Grad-CAM](#12-notebook-grad-cam)
- [13. Tests](#13-tests)
- [14. Docker](#14-docker)
- [15. Structure de la configuration](#15-structure-de-la-configuration)
- [16. Explication du pipeline de training](#16-explication-du-pipeline-de-training)
- [17. Ce qu'on optimise réellement dans le projet](#17-ce-quon-optimise-réellement-dans-le-projet)
- [18. Résultats attendus et livrables](#18-résultats-attendus-et-livrables)
- [19. Dépannage](#19-dépannage)
- [20. Pistes d'amélioration](#20-pistes-damélioration)
- [21. Pitch entretien](#21-pitch-entretien)

---

# 1. Pourquoi ce projet ?

Beaucoup de projets IA sont limités à un notebook unique.  
Ce repository vise à aller plus loin en montrant :

1. une **structure de code propre**
2. une **séparation claire des responsabilités**
3. une **reproductibilité minimale**
4. une **capacité à améliorer un modèle existant**
5. une **mise à disposition du modèle via API**
6. une **visualisation simple pour démonstration**
7. une **base pour expliquer le modèle**

Autrement dit, il montre non seulement que tu sais entraîner un modèle, mais aussi que tu sais :

- organiser un projet ML,
- manipuler des images médicales,
- exposer une solution utilisable,
- et documenter ton travail.

---

# 2. Ce que démontre techniquement le projet

Le projet met en avant les compétences suivantes :

- **Python**
- **TensorFlow / Keras**
- **prétraitement d'images**
- **classification binaire**
- **transfer learning**
- **class weights**
- **early stopping / reduce LR**
- **MLflow**
- **FastAPI**
- **Streamlit**
- **Docker**
- **pytest**
- **Explainable AI avec Grad-CAM**

Il est particulièrement pertinent pour des postes du type :

- Machine Learning Engineer
- Computer Vision Engineer
- Applied AI Engineer
- AI Engineer en MedTech
- ML Engineer orienté produit

---

# 3. Architecture du repository

Voici la structure actuelle du projet :

```text
medvision-ai-main/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── docker/
│   └── Dockerfile
├── notebooks/
│   ├── data/
│   │   └── gradcam_demo_image.png
│   └── gradcam_medvision_complete.ipynb
├── scripts/
│   ├── run_api.sh
│   ├── run_streamlit.sh
│   └── run_training.sh
├── src/
│   ├── api/
│   │   └── main.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   └── predict.py
│   ├── preprocessing/
│   │   ├── augmentation.py
│   │   └── image_loader.py
│   ├── training/
│   │   └── train.py
│   └── utils/
│       ├── config.py
│       ├── dataset.py
│       └── paths.py
├── tests/
├── tex/
├── docker-compose.yml
├── requirements.txt
├── README.md
└── streamlit_app.py
```

## Rôle des dossiers

### `configs/`
Contient la configuration principale du projet :

- taille d'image
- batch size
- nombre d'epochs
- learning rate
- chemin du dataset
- répertoires de sortie
- classes

### `data/raw/`
Contient les données brutes.  
Le dataset Kaggle doit y être décompressé.

### `data/processed/`
Réservé aux données prétraitées si tu veux étendre le projet plus tard.

### `src/training/`
Contient le pipeline d'entraînement.

### `src/inference/`
Contient la logique de prédiction à partir d'une image.

### `src/api/`
Expose le modèle sous forme de service FastAPI.

### `src/evaluation/`
Contient les métriques et fonctions d'évaluation.

### `src/preprocessing/`
Contient le chargement et les augmentations d'images.

### `notebooks/`
Contient les notebooks de démonstration, notamment le notebook Grad-CAM.

### `tex/`
Contient les documents mathématiques et pédagogiques liés au projet.

---

# 4. Dataset utilisé

Le projet est configuré pour utiliser le dataset public Kaggle :

**Chest X-Ray Pneumonia**

Page Kaggle :
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Ce dataset contient des radiographies thoraciques organisées en deux classes :

- `NORMAL`
- `PNEUMONIA`

Le projet attend par défaut cette structure :

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Le chemin est défini dans :

```yaml
dataset_dir: data/raw/chest_xray
```

dans `configs/config.yaml`.

> Important : le dataset n'est **pas inclus** dans ce repository.

---

# 5. Téléchargement automatique du dataset

Le moyen le plus propre d'automatiser le téléchargement consiste à utiliser la **Kaggle CLI**.

## Installation de la CLI

```bash
pip install kaggle
```

Tu peux ensuite télécharger automatiquement le dataset.

---

## Linux / macOS

Depuis la racine du projet :

```bash
mkdir -p data/raw
cd data/raw

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -o chest-xray-pneumonia.zip
rm -f chest-xray-pneumonia.zip

cd ../..
```

À la fin, tu dois avoir :

```text
data/raw/chest_xray/
```

---

## Windows PowerShell

Depuis la racine du projet :

```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
Set-Location data\raw

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
Expand-Archive -Path chest-xray-pneumonia.zip -Force
Remove-Item chest-xray-pneumonia.zip -Force

Set-Location ..\..
```

---

## Windows CMD

Depuis la racine du projet :

```bat
mkdir data\raw
cd data\raw

kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
tar -xf chest-xray-pneumonia.zip
del chest-xray-pneumonia.zip

cd ..\..
```

> Selon ta version de Windows, `tar -xf` peut fonctionner directement.  
> Sinon, utilise PowerShell avec `Expand-Archive`.

---

# 6. Configuration Kaggle

Pour que la CLI Kaggle fonctionne, il faut configurer les credentials.

## Étape 1 — créer une clé API Kaggle

- connecte-toi à Kaggle
- va dans ton profil
- ouvre **Account**
- dans la section **API**, clique sur **Create New Token**

Tu récupères un fichier `kaggle.json`.

## Étape 2 — placer le fichier au bon endroit

### Linux / macOS

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Windows

Place le fichier dans :

```text
C:\Users\<TON_USER>\.kaggle\kaggle.json
```

## Alternative avec variables d'environnement

Tu peux aussi définir :

### Linux / macOS

```bash
export KAGGLE_USERNAME=ton_username
export KAGGLE_KEY=ta_cle
```

### Windows PowerShell

```powershell
$env:KAGGLE_USERNAME="ton_username"
$env:KAGGLE_KEY="ta_cle"
```

---

# 7. Installation du projet

## Version de Python recommandée

Le projet utilise TensorFlow `2.16.1`, ce qui implique :

- **Python 3.10 à 3.12 recommandé**
- éviter Python 3.13 pour l'instant

## Créer un environnement virtuel

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate.bat
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient notamment :

- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Pillow
- OpenCV headless
- MLflow
- FastAPI
- Uvicorn
- Streamlit
- PyYAML
- Matplotlib
- pytest
- httpx

---

# 8. Entraînement des modèles

Le pipeline de training est dans :

```text
src/training/train.py
```

Le script prend au minimum :

- un fichier de config
- un type de modèle

## Entraîner le modèle baseline

```bash
python -m src.training.train --config configs/config.yaml --model baseline
```

## Entraîner le modèle optimisé

```bash
python -m src.training.train --config configs/config.yaml --model optimized
```

## Lancer avec le script shell existant

```bash
bash scripts/run_training.sh
```

Ce script lance actuellement :

```bash
python -m src.training.train --config configs/config.yaml --model optimized
```

## Surcharge du nombre d'epochs

```bash
python -m src.training.train --config configs/config.yaml --model optimized --epochs 10
```

---

# 9. Suivi des expériences avec MLflow

Le projet loggue les expériences dans MLflow.

Le tracking URI configuré par défaut est :

```yaml
mlflow_tracking_uri: file:./mlruns
```

## Lancer l'interface MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Ensuite ouvrir :

```text
http://localhost:5000
```

## Ce qui est loggué

Le training enregistre notamment :

- le type de modèle
- la taille des images
- le batch size
- le nombre d'epochs
- le learning rate
- le seed
- les métriques de validation et de test

---

# 10. API FastAPI

L'API est définie dans :

```text
src/api/main.py
```

Le modèle attendu par défaut est :

```text
artifacts/models/optimized_model.keras
```

## Lancer l'API

### Directement

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Avec le script existant

```bash
bash scripts/run_api.sh
```

## Endpoints disponibles

### Health check

```http
GET /health
```

Réponse :

```json
{"status": "ok"}
```

### Prédiction

```http
POST /predict
```

avec un fichier image en upload multipart.

## Documentation Swagger

Une fois l'API lancée :

```text
http://localhost:8000/docs
```

Tu peux y tester l'upload directement.

---

# 11. Interface Streamlit

Le projet contient une interface de démonstration dans :

```text
streamlit_app.py
```

Elle :
- charge le modèle optimisé,
- permet d'uploader une image,
- affiche la prédiction,
- affiche la probabilité de pneumonie.

## Lancer Streamlit

### Directement

```bash
streamlit run streamlit_app.py
```

### Avec le script existant

```bash
bash scripts/run_streamlit.sh
```

## Cas d'usage

C'est très pratique pour :
- montrer le projet en entretien,
- faire une démonstration rapide,
- tester visuellement le modèle sans passer par Swagger.

---

# 12. Notebook Grad-CAM

Le projet contient le notebook :

```text
notebooks/gradcam_medvision_complete.ipynb
```

Ce notebook montre un pipeline complet :

1. chargement d'un modèle Keras
2. préparation d'une image
3. prédiction
4. extraction de la dernière couche convolutionnelle
5. calcul des gradients
6. calcul de la heatmap Grad-CAM
7. superposition à l'image

## Pourquoi c'est important

Ce notebook est particulièrement utile pour :

- l'explicabilité locale,
- la démonstration technique,
- la discussion en entretien sur la confiance dans les modèles médicaux.

---

# 13. Tests

Le projet contient des tests dans :

```text
tests/
```

## Lancer tous les tests

```bash
pytest -q
```

Les tests couvrent notamment :
- l'API
- les métriques
- certaines briques de preprocessing

---

# 14. Docker

Le projet contient :

- `docker/Dockerfile`
- `docker-compose.yml`

Cela permet de poser une base de reproductibilité locale.

## Build manuel

```bash
docker build -f docker/Dockerfile -t medvision-ai .
```

## Via Docker Compose

```bash
docker compose up --build
```

> Selon ton usage, il faudra peut-être ajuster les volumes et le montage du dataset local.

---

# 15. Structure de la configuration

Le fichier `configs/config.yaml` contient actuellement :

```yaml
project_name: medvision-ai
seed: 42
image_size: 224
batch_size: 16
epochs: 3
learning_rate: 0.0005
validation_split: 0.2
dataset_dir: data/raw/chest_xray
model_dir: artifacts/models
reports_dir: artifacts/reports
mlflow_tracking_uri: file:./mlruns
class_names:
  - NORMAL
  - PNEUMONIA
```

## Explication des paramètres

- `project_name` : nom du projet pour MLflow
- `seed` : reproductibilité
- `image_size` : taille des images en entrée
- `batch_size` : taille des mini-lots
- `epochs` : nombre d'époques par défaut
- `learning_rate` : taux d'apprentissage
- `validation_split` : part de validation
- `dataset_dir` : chemin du dataset
- `model_dir` : dossier de sauvegarde des modèles
- `reports_dir` : rapports et métriques
- `class_names` : noms des classes

---

# 16. Explication du pipeline de training

Le script `src/training/train.py` fait essentiellement les choses suivantes :

1. charge la configuration YAML
2. fixe le seed aléatoire
3. construit les datasets `train / val / test`
4. construit un modèle `baseline` ou `optimized`
5. calcule les class weights
6. entraîne le modèle avec callbacks
7. évalue le modèle
8. enregistre métriques, rapports et modèle

## Callbacks utilisés

Le code utilise :

- `EarlyStopping`
- `ReduceLROnPlateau`

Cela améliore la stabilité de l'entraînement et évite certains sur-apprentissages simples.

---

# 17. Ce qu'on optimise réellement dans le projet

Le projet est intéressant parce qu'il raconte une histoire crédible :

## Modèle baseline
Le baseline représente un modèle existant, simple, rapide, mais limité.

## Modèle optimisé
Le modèle optimisé ajoute une logique plus sérieuse :

- transfer learning
- meilleure capacité de généralisation
- meilleur usage du dataset
- meilleure stabilité à l'entraînement

## Message entretien possible

Tu peux expliquer le projet comme ceci :

> I started from a simple baseline CNN to simulate an existing legacy model, then I improved performance and robustness using transfer learning, class weighting, better callbacks and experiment tracking with MLflow. I also exposed the model through FastAPI and built a simple demo UI with Streamlit.

---

# 18. Résultats attendus et livrables

Après un entraînement complet, tu devrais retrouver :

## Modèle sauvegardé

```text
artifacts/models/
```

## Rapports

```text
artifacts/reports/
```

## Runs MLflow

```text
mlruns/
```

## Démo locale

- API disponible sur `localhost:8000`
- Swagger sur `localhost:8000/docs`
- Streamlit en local

---

# 19. Dépannage

## 1. TensorFlow ne s'installe pas
Vérifie ta version de Python.

Utilise de préférence :
- Python 3.10
- Python 3.11
- Python 3.12

## 2. Kaggle CLI ne fonctionne pas
Vérifie :
- la présence de `kaggle.json`
- les permissions
- ou les variables `KAGGLE_USERNAME` / `KAGGLE_KEY`

## 3. L'API dit que le modèle est introuvable
Il faut d'abord entraîner le modèle optimisé pour générer :

```text
artifacts/models/optimized_model.keras
```

## 4. Streamlit s'arrête immédiatement
Même cause : pas de modèle entraîné.

## 5. Le dataset n'est pas trouvé
Vérifie que le dossier final est bien :

```text
data/raw/chest_xray/
```

et non un niveau de dossier supplémentaire du type :

```text
data/raw/chest-xray-pneumonia/chest_xray/
```

---

# 20. Pistes d'amélioration

Voici plusieurs prolongements possibles si tu veux professionnaliser encore le projet :

## ML / Vision
- multiclass classification
- segmentation
- attention maps avancées
- calibration des probabilités
- comparaison ResNet / EfficientNet / DenseNet

## Explainable AI
- intégration automatique de Grad-CAM dans l'API
- sauvegarde systématique des overlays
- comparaison Grad-CAM / Grad-CAM++

## Produit
- authentification simple
- historique des inférences
- stockage des runs et métadonnées
- upload multiple d'images

## MLOps
- versionnage des modèles
- DVC
- CI plus avancée
- conteneurisation complète API + UI + MLflow

---

# 21. Pitch entretien

Tu peux présenter le projet en entretien de cette façon :

> MedVision AI is a production-minded medical imaging project built around chest X-ray classification.  
> I implemented a baseline model and an optimized model to simulate the improvement of an existing pipeline.  
> The project includes image preprocessing, TensorFlow training, class balancing, experiment tracking with MLflow, model serving through FastAPI, a Streamlit demo, and explainability with Grad-CAM.  
> The goal was not just to train a model, but to demonstrate an end-to-end workflow similar to what a small MedTech or applied AI team would build.

---

# Commandes utiles récapitulatives

## Installer

```bash
pip install -r requirements.txt
```

## Télécharger le dataset

```bash
mkdir -p data/raw
cd data/raw
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -o chest-xray-pneumonia.zip
rm -f chest-xray-pneumonia.zip
cd ../..
```

## Entraîner le baseline

```bash
python -m src.training.train --config configs/config.yaml --model baseline
```

## Entraîner l'optimisé

```bash
python -m src.training.train --config configs/config.yaml --model optimized
```

## Lancer MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

## Lancer l'API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Lancer Streamlit

```bash
streamlit run streamlit_app.py
```
