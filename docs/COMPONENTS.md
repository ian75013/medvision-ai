# Components Map

## Racine du projet

- `README.md` : vue d'ensemble du dépôt
- `README_DVC.md` : guide DVC
- `README_MRI_SPRINTS.md` : notes de transition MRI
- `dvc.yaml` : pipeline DVC
- `params.yaml` : paramètres du pipeline
- `docker-compose.yml` : lancement rapide des services
- `requirements.txt` : dépendances Python principales

## `configs/`

- `config.yaml` : configuration historique chest X-ray
- `brain_tumor_mri.yaml` : configuration du workflow Brain MRI
- `brain_mri_2d_demo.yaml` : configuration de démonstration MRI synthétique

## `src/data/`

- `download_dataset.py` : téléchargement historique du dataset chest X-ray
- `download_brain_mri_dataset.py` : téléchargement du dataset Kaggle Brain MRI

## `src/utils/`

- `config.py` : lecture YAML
- `dataset.py` : helpers dataset historique
- `dataset_multiclass.py` : chargeur dataset MRI multi-classe
- `paths.py` : création de répertoires
- `logging.py` : utilitaires logging
- `seed.py` : seed globale

## `src/training/`

- `train.py` : entraînement historique binaire chest X-ray
- `train_brain_mri.py` : entraînement multi-classe Brain MRI
- `train_classifier.py` : démo PyTorch MRI 2D
- `trainer.py` : boucle d'entraînement démonstrative

## `src/evaluation/`

- `metrics.py` : métriques binaires
- `metrics_multiclass.py` : métriques multi-classes MRI

## `src/inference/`

- `predict.py` : inférence historique
- `predict_brain_mri.py` : inférence MRI
- `predict_classifier.py` : démo MRI classifier

## `src/api/`

- `main.py` : API FastAPI minimale

## `terraform/aws_dvc_remote/`

Contient le module Terraform pour créer un bucket S3 privé utilisable comme remote DVC.

## `docs/`

Documentation fonctionnelle, architecturelle et opérationnelle ajoutée pour rendre le dépôt plus compréhensible.
