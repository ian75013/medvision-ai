# Guide MLOps

## Positionnement des briques

### Git
Suit le code, les fichiers texte et la documentation.

### DVC
Suit le pipeline ML, les métriques, les datasets lourds et les remotes.

### MLflow
Suit les runs d'entraînement, les paramètres et les artefacts expérimentaux.

### Terraform
Crée l'infrastructure de stockage distante utilisée par DVC.

### Docker Compose
Fournit un environnement de lancement rapide des services de démonstration.

## Workflow recommandé

### 1. Télécharger les données
```bash
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
```

### 2. Entraîner et logger vers MLflow
```bash
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized --epochs 10
```

### 3. Rejouer via DVC
```bash
dvc repro
```

### 4. Visualiser les runs MLflow
```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

## Ce que stocke chaque brique

### Git
- code source
- README
- configs
- fichiers Terraform
- `dvc.yaml`
- `params.yaml`

### DVC
- datasets bruts
- artefacts de modèle lourds
- métriques/plots produits par pipeline
- remote S3 ou Google Drive

### MLflow
- runs
- paramètres
- métriques
- artefacts liés à un run

## Remote DVC S3

1. provisionner le bucket via Terraform ;
2. déclarer le remote DVC ;
3. pousser les artefacts.

## Remote DVC Google Drive

Adapté pour des tests personnels ou un partage léger, moins adapté qu'un backend objet pour un vrai usage d'équipe.

## Évolution MLOps conseillée

- ajouter une étape `prepare` dans DVC ;
- ajouter une étape `evaluate` séparée ;
- persister MLflow sur un backend plus robuste ;
- séparer explicitement les pipelines `chest_xray` et `brain_mri` ;
- ajouter CI pour lint/tests/smoke training.
