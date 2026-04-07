# MLflow experiment plan (2026-04-07)

Ce document resume les dernieres modifications utiles pour lancer de nouvelles experiences MLflow et augmenter l'accuracy.

## 1) Modifications recentes a prendre en compte

### Commit `f155010` (07/04/2026)

Objectif: renforcer le transfer learning pour la classification (chest X-ray + brain MRI).

Principaux changements:

- `src/training/train.py`:
  - passage a un schema warmup + fine-tuning progressif pour TensorFlow,
  - support de plusieurs backbones (`densenet121`, `efficientnetv2b0`, `convnexttiny`, `resnet50v2`, etc.),
  - logging MLflow enrichi (`epochs_total`, `warmup_epochs`, `finetune_epochs`, `warmup_lr`, `finetune_lr`, `final_*`, `best_*`).
- `src/training/train_brain_mri.py`:
  - meme logique warmup + fine-tuning progressif,
  - class weights pour le multiclasse,
  - logging MLflow des params de schedule et des metriques finales.
- `src/training/transfer_utils.py`:
  - centralisation de la configuration de fine-tuning (`FineTuneConfig`),
  - callbacks par defaut (`EarlyStopping`, `ReduceLROnPlateau`),
  - fonction d'inference automatique de la profondeur de de-gel (`infer_unfreeze_layers`).
- `src/training/train_brain_mri_torch.py`:
  - nouveau pipeline PyTorch pour benchmark croise framework,
  - warmup puis de-gel partiel des derniers blocs,
  - tracking MLflow (params + metriques + artefacts).
- `configs/brain_tumor_mri.yaml`:
  - nouveaux parametres de schedule (`warmup_epochs`, `warmup_learning_rate`, `finetune_learning_rate`, `unfreeze_layers`, `dropout`, `torch_unfreeze_blocks`, `label_smoothing`).

### Commits segmentation (29/03/2026)

Objectif: fiabiliser les runs MLflow pour eviter les executions incompletes.

- `src/segmentation/train_segmentation.py`:
  - calcul dynamique des `steps_per_epoch` / `validation_steps`,
  - gestion d'erreurs plus robuste autour de l'entrainement,
  - logging d'artefacts en `finally` (meme en cas d'echec),
  - run MLflow nested pour pousser metriques + artefacts en fin de pipeline.

Impact direct: moins de runs "perdus" et meilleure comparabilite entre experiences.

## 2) Commandes de base pour experimenter

Lancer l'UI MLflow:

```bash
mlflow ui --backend-store-uri ./mlruns
```

### Chest X-ray (TensorFlow)

```bash
python -m src.training.train --config configs/config.yaml --model densenet121
python -m src.training.train --config configs/config.yaml --model efficientnetv2b0
python -m src.training.train --config configs/config.yaml --model convnexttiny
```

### Brain MRI (TensorFlow)

```bash
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model densenet121
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model efficientnetv2b0
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model convnexttiny
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model resnet50v2
```

### Brain MRI (PyTorch)

```bash
python -m src.training.train_brain_mri_torch --config configs/brain_tumor_mri.yaml --model densenet121_torch
python -m src.training.train_brain_mri_torch --config configs/brain_tumor_mri.yaml --model resnet50_torch
python -m src.training.train_brain_mri_torch --config configs/brain_tumor_mri.yaml --model swin_v2_s_torch
```

## 3) Plan d'experiences MLflow (objectif: monter l'accuracy)

Faire les campagnes par blocs pour garder des comparaisons propres.

### Bloc A - Selection du backbone

Fixer les hyperparametres de schedule et comparer uniquement les backbones.

- TensorFlow: `densenet121`, `efficientnetv2b0`, `convnexttiny`, `resnet50v2`
- PyTorch (MRI): `densenet121_torch`, `resnet50_torch`, `swin_v2_s_torch`

Critere de sortie:

- garder les 2 meilleurs backbones selon `accuracy` + `f1_macro` (MRI) ou `f1` (chest).

### Bloc B - Schedule warmup / finetune

Sur les 2 meilleurs backbones, tester:

- `warmup_epochs`: 4, 6, 8
- `finetune_learning_rate`: 1e-5, 3e-5, 1e-4
- `unfreeze_layers` (TensorFlow): 30, 50, 80
- `torch_unfreeze_blocks` (PyTorch): 1, 2, 3

Critere de sortie:

- meilleur compromis `accuracy` + stabilite (`best_val_loss`, ecart train/val).

### Bloc C - Regularisation

Tester ensuite:

- `dropout`: 0.2, 0.25, 0.35
- `label_smoothing`: 0.0, 0.05, 0.1

Critere de sortie:

- gain de generalisation sur test sans baisse forte du rappel macro.

## 4) Metriques MLflow a prioriser

### Brain MRI

- `accuracy`
- `f1_macro`
- `recall_macro`
- `top2_accuracy` (TensorFlow)
- `best_val_loss` / `final_val_loss`

### Chest X-ray

- `accuracy`
- `f1`
- `recall`
- `roc_auc`
- `best_val_loss`

### Segmentation

- `dice`, `iou`, `pixel_accuracy`
- `classification_accuracy` / `classification_f1` (multitask)

## 5) Regles de decision rapides

1. Ne pas promouvoir un run sur `accuracy` seule.
2. En cas de scores proches, choisir le run avec meilleure `f1_macro` (MRI) ou meilleur `recall` (chest).
3. Ecarter les runs avec forte divergence train/val meme si l'accuracy brute est haute.
4. Conserver les artefacts (model + metrics + history + confusion/overlays) pour chaque candidat final.

## 6) Checklist avant comparaison finale

1. Meme split de donnees et meme seed pour les runs compares.
2. Verification de la presence des params MLflow de schedule.
3. Verification de la presence des artefacts attendus.
4. Classement final par metrique primaire + metrique secondaire.
