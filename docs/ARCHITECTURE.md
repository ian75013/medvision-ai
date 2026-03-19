# Architecture

## Overview

MedVision now exposes a single comparison surface for two problems:

- `chest_xray`: binary pneumonia classification
- `brain_mri`: 4-class brain tumor MRI classification

The project is organized around four layers:

1. **Training**
   - `src/training/train.py` for chest X-ray
   - `src/training/train_brain_mri.py` for brain MRI
2. **Registry**
   - `src/registry/model_registry.py` discovers trained models and metrics in `artifacts/`
3. **Serving**
   - `src/api/main.py` exposes `/models`, `/compare`, and `/predict`
4. **UI**
   - `streamlit_app.py` lets you compare metrics and run side-by-side predictions across models

## Artifact convention

The comparison layer relies on predictable artifact names:

### Chest X-ray
- `artifacts/models/baseline_model.keras`
- `artifacts/models/optimized_model.keras`
- `artifacts/reports/baseline_metrics.json` (optional)
- `artifacts/reports/optimized_metrics.json` (optional)

### Brain MRI
- `artifacts/models/brain_mri_baseline.keras`
- `artifacts/models/brain_mri_optimized.keras`
- `artifacts/reports/brain_mri_baseline_metrics.json` (optional)
- `artifacts/reports/brain_mri_metrics.json`

## Why a registry?

FastAPI and Streamlit should not hardcode a single model file anymore.
The registry scans the artifact directories and produces one normalized view for both problems so the API and UI can stay aligned.
