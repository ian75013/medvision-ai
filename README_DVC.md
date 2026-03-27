# README_DVC — MedVision

## DVC is not Docker

- **Git** versions the code.
- **DVC** versions datasets, model artifacts, and ML pipelines.
- **Docker** isolates the execution environment.
- **MLflow** tracks runs, metrics, and artifacts.

In MedVision, DVC orchestrates a reproducible graph of stages for both classification and segmentation.

## Main stages in this repository

### Classification
- `download_chest_xray`
- `train_chest_xray`
- `download_brain_mri`
- `train_brain_mri`

### Segmentation
- `download_brain_tumor_segmentation`
- `prepare_brain_tumor_segmentation`
- `train_brain_tumor_segmentation`
- `download_chest_xray_segmentation`
- `prepare_chest_xray_segmentation`
- `train_chest_xray_segmentation`

## Key commands

```bash
dvc repro
dvc status
dvc exp run
dvc exp show
dvc push
dvc pull
```

## Why the prepare stage matters

Segmentation datasets are often messy:
- images and masks live in different folders,
- naming conventions are inconsistent,
- labels can be stored in folder names.

The `prepare_segmentation_dataset` stage creates a manifest CSV so the training code receives one clean table:
- `image_path`
- `mask_path`
- `label`
- `split`

This is a classic MLOps move: convert messy raw data into a deterministic intermediate representation.

## DVC + MLflow together

Use DVC to guarantee the right files and code versions are connected to a run.
Use MLflow to inspect the results of that run.

A good mental model is:
- **DVC answers:** “What exact pipeline produced this artifact?”
- **MLflow answers:** “How well did that run perform?”
