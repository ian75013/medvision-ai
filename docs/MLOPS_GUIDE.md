# MLOps Guide

This guide explains how MedVision AI manages reproducibility, experiment tracking, and artifact lifecycle.

## 1. Responsibilities by tool

### 1.1 Git

Git tracks:

- source code
- configs
- docs
- notebooks
- DVC metadata files

### 1.2 DVC

DVC tracks:

- pipeline graph and stage dependencies
- raw and processed data outputs
- trained model artifacts
- metrics files and intermediate outputs

### 1.3 MLflow

MLflow tracks:

- run parameters
- scalar metrics
- run artifacts and visual outputs

Practical interpretation:

- DVC answers lineage and reproducibility.
- MLflow answers performance and run diagnostics.

## 2. Pipeline lifecycle model

Typical lifecycle:

1. Acquire or refresh dataset inputs.
2. Prepare deterministic training-ready data (manifests for segmentation).
3. Train models and generate artifacts.
4. Inspect run quality in MLflow.
5. Compare model behavior via Streamlit and API.
6. Persist large artifacts to remote storage when required.

## 3. DVC usage patterns

Core commands:

```bash
dvc status
dvc repro
dvc exp run
dvc exp show
dvc push
dvc pull
```

Recommended usage:

- Use stage-specific repro during iterative debugging.
- Use full repro for integration checks.
- Use exp run and exp show for controlled hyperparameter iteration.

## 4. MLflow usage patterns

Start local UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

What to inspect on each run:

1. Parameter set used by training.
2. Final and intermediate metric behavior.
3. Presence of expected model/report artifacts.
4. Segmentation overlays for qualitative sanity.

## 5. Reproducibility checklist

Before claiming a run is reproducible:

1. Confirm code revision and config files are committed.
2. Confirm DVC stages are up to date.
3. Confirm artifact paths match expected contracts.
4. Confirm MLflow run has parameters and metrics logged.
5. Confirm rerun with same inputs produces expected variance envelope.

## 6. Remote storage strategy

When local artifacts become large:

- Configure DVC remote (S3 or GDrive, depending on environment).
- Push artifacts with dvc push.
- Pull artifacts with dvc pull on another machine.

Terraform infrastructure under terraform/aws_dvc_remote can be used for S3-backed remote provisioning.

## 7. Anti-patterns to avoid

1. Training directly with ad-hoc local data paths outside DVC contracts.
2. Renaming artifact outputs without registry updates.
3. Comparing runs without checking parameter parity.
4. Interpreting segmentation metrics without visual overlay checks.

## 8. Team operating recommendations

1. Keep dvc.yaml and params.yaml changes explicit in PR descriptions.
2. Attach MLflow run IDs when discussing model quality changes.
3. Run at least one short smoke training before long experiments.
4. Prefer deterministic, documented experiment names and output paths.

## 9. MLflow metrics by track

This section defines the expected metrics logged to MLflow for each training track.

### 9.1 Chest X-ray binary classification

Core evaluation metrics:

- accuracy
- balanced_accuracy
- precision
- recall
- f1
- specificity
- roc_auc
- pr_auc

Confusion-derived counters:

- tp
- tn
- fp
- fn

Per-class metrics:

- normal_precision
- normal_recall
- normal_f1_score
- pneumonia_precision
- pneumonia_recall
- pneumonia_f1_score
- macro_f1

Optimization/context metrics:

- class_weight_0
- class_weight_1
- final_* and best_* history metrics (for example final_val_loss, best_val_auc)

### 9.2 Brain MRI multiclass classification

Core evaluation metrics:

- accuracy
- precision_macro
- recall_macro
- f1_macro
- precision_weighted
- recall_weighted
- f1_weighted
- top2_accuracy
- log_loss
- num_classes

Per-class metrics:

- <class_name>_f1 for each class in class_names

Optimization/context metrics:

- class_weight_<index> for each class index
- final_* and best_* history metrics (for example final_val_accuracy, best_val_loss)

### 9.3 Segmentation tracks (brain and chest)

Mask quality metrics:

- dice
- iou
- pixel_accuracy
- mask_precision
- mask_recall
- mask_f1

Classification branch metrics (multitask runs):

- classification_accuracy
- classification_precision
- classification_recall
- classification_f1

Optimization/context metrics:

- final_* and best_* history metrics from model.fit history

## 10. Minimum acceptance checks for a run

Before considering a run valid in MLflow:

1. Required metrics for the selected track are present.
2. Model artifact is logged.
3. Metrics JSON and history JSON are logged.
4. For segmentation, at least one overlay artifact is logged.

If one of these is missing, treat the run as incomplete and rerun after fixing logging or artifact contract issues.
