# Components

This document maps the main code components and their responsibilities.

## 1. Data acquisition and preparation

### src/data/download_dataset.py

- Downloads the chest X-ray classification dataset.
- Writes raw assets to data/raw/chest_xray.

### src/data/download_brain_mri_dataset.py

- Downloads the brain MRI classification dataset.
- Uses config-driven behavior through configs/brain_tumor_mri.yaml.

### src/data/download_segmentation_dataset.py

- Downloads segmentation datasets using problem-specific selectors.
- Supports current segmentation tracks for brain tumor and chest X-ray.

### src/data/prepare_segmentation_dataset.py

- Converts heterogeneous raw segmentation data into a consistent manifest CSV.
- Produces stable columns required by training: image_path, mask_path, label, split.

### src/segmentation/datasets/manifest.py

- Contains manifest-building logic and dataset matching heuristics.
- Central place for image/mask pairing behavior.

## 2. Training components

### src/training/train.py

- Main training entry point for chest X-ray classification.
- Produces model and report artifacts consumed by registry and serving.

### src/training/train_brain_mri.py

- Brain MRI classification training entry point.
- Handles multiclass training and associated metrics artifacts.

### src/models/

- Contains classification model builders.
- Includes baseline and optimized variants used by training scripts.

### src/segmentation/train_segmentation.py

- Main training entry point for multitask segmentation tracks.
- Produces model, metrics, and visual overlays.

### src/segmentation/models/unet.py

- U-Net style architecture for segmentation with classification head.
- Core multitask model definition.

### src/segmentation/data.py

- Builds TensorFlow datasets from segmentation manifests.
- Encodes preprocessing and batching contract for segmentation runs.

### src/segmentation/metrics.py and src/segmentation/overlays.py

- Computes segmentation and classification metrics.
- Produces qualitative overlays for sanity checks and reporting.

## 3. Serving components

### src/registry/model_registry.py

- Shared registry used by both API and Streamlit.
- Discovers model availability from artifacts and merges metadata.

### src/api/main.py

- FastAPI app exposing health, registry, model listing, comparison, and prediction endpoints.
- Uses registry + model loader for runtime inference behavior.

### streamlit_app.py

- Interactive UI for problem selection, model comparison, and prediction previews.
- Reads from the same registry contract as FastAPI.

## 4. Pipeline and operations components

### dvc.yaml

- Declares stage graph, dependencies, outputs, and metrics files.
- Defines reproducible lifecycle for classification and segmentation tracks.

### params.yaml and configs/

- Hold hyperparameters and task-specific runtime settings.
- Should be treated as first-class inputs to any reproducibility claim.

### docker-compose.yml and docker/Dockerfile

- Provide local composed runtime for MLflow, API, and Streamlit.
- Useful for demo environments and quick service bring-up.
