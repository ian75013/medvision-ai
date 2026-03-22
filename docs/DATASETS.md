# Datasets

This document describes dataset expectations, preparation flow, and validation checks.

## 1. Dataset strategy

The repository handles two families of tasks:

1. Classification datasets with class-folder conventions.
2. Segmentation datasets with variable directory conventions.

Classification data is usually ready to train once downloaded.
Segmentation data requires a manifest normalization step.

## 2. Classification datasets

### 2.1 Chest X-ray pneumonia

Expected layout under data/raw/chest_xray:

- train/NORMAL
- train/PNEUMONIA
- val/NORMAL and val/PNEUMONIA
- test/NORMAL and test/PNEUMONIA

Download path:

- python -m src.data.download_dataset

### 2.2 Brain MRI tumor classification

Expected layout under data/raw/brain_tumor_mri:

- Training/glioma
- Training/meningioma
- Training/notumor
- Training/pituitary
- Testing/* with equivalent class folders

Download path:

- python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml

## 3. Segmentation datasets

Segmentation sources are heterogeneous in naming and structure.
The repository intentionally uses a two-step ingestion model:

1. Download raw dataset files.
2. Build a unified manifest used by training code.

Download path:

- python -m src.data.download_segmentation_dataset --problem brain_tumor_seg
- python -m src.data.download_segmentation_dataset --problem chest_xray_seg

Prepare path:

- python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml
- python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml

## 4. Manifest contract

Generated manifest columns:

- image_path
- mask_path
- label
- split

The manifest is the primary contract between data preparation and segmentation training.
Training code does not depend on original raw directory structure once this CSV exists.

## 5. Validation checklist before long runs

1. Open the manifest and verify path integrity for random rows.
2. Confirm image and mask files exist for sampled pairs.
3. Visualize several image/mask examples.
4. Check split distribution to avoid severe imbalance.
5. Run a short training cycle before full-epoch training.

## 6. Common failure patterns

### 6.1 Missing or mismatched masks

Symptoms:

- poor qualitative overlays
- unstable segmentation metrics

Action:

- inspect matching heuristics and sampled manifest rows

### 6.2 Incorrect class semantics

Symptoms:

- unexpectedly low classification confidence
- inconsistent class-level metrics

Action:

- verify source labels and folder mapping in downloaded dataset

## 7. Storage conventions

- Raw data lives in data/raw
- Derived manifests live in data/processed
- Trained outputs live in artifacts

Keep this separation strict to simplify reproducibility and DVC stage behavior.
