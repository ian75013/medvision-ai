# MedVision AI Project Guide

This guide is the entry point for engineers and contributors who need a practical understanding of the codebase.

## 1. How to use this guide

Read in this order:

1. Main README for quick setup and first run.
2. This guide for project-level operating context.
3. Specialized docs in docs/ for architecture, MLOps, datasets, and operations.

## 2. What this project provides

The repository supports four runnable tracks:

1. Chest X-ray pneumonia classification
2. Brain MRI tumor classification
3. Brain tumor segmentation + classification (multitask U-Net)
4. Chest X-ray segmentation + classification (multitask U-Net)

The focus is not only model metrics, but reproducible engineering workflows.

## 3. First-day onboarding checklist

1. Create a Python 3.12 virtual environment.
2. Install requirements.txt (prefer uv, legacy venv/pip also supported).
3. Configure Kaggle API token.
4. Run a short training smoke test.
5. Start API and confirm /health.
6. Start Streamlit and verify model visibility.
7. Open MLflow and inspect run details.

Quick command sequence:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.data.download_dataset
python -m src.training.train --config configs/config.yaml --model optimized --epochs 1
uvicorn src.api.main:app --reload
streamlit run streamlit_app.py
mlflow ui --backend-store-uri ./mlruns
```

Legacy setup alternative:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Project operating model

Data path:

- Download raw datasets
- Normalize segmentation inputs with manifest generation

Training path:

- Run task-specific trainers
- Emit model, metrics, report, and overlay artifacts

Serving path:

- FastAPI and Streamlit consume the same registry contract

Reproducibility path:

- DVC captures stage dependencies and outputs
- MLflow captures run metadata and outcomes

## 5. Contributor rules of thumb

1. Treat config files as first-class inputs.
2. Keep artifact naming aligned with registry expectations.
3. Prefer stage-by-stage DVC iteration during debugging.
4. Validate segmentation inputs visually before long runs.
5. Update docs when behavior, contracts, or outputs change.

## 6. Document map

- docs/ARCHITECTURE.md: platform architecture, contracts, and extension strategy.
- docs/COMPONENTS.md: module responsibilities and component boundaries.
- docs/DATASETS.md: dataset contracts, manifest schema, validation checks.
- docs/FASTAPI_STREAMLIT_ALIGNMENT.md: serving contract across API and UI.
- docs/SEGMENTATION_UI_GUIDE.md: how to use segmentation in Streamlit, threshold tuning, and diagnostics.
- docs/DVC_GUIDE.md: day-to-day DVC runbook, experiments, remotes, and troubleshooting.
- docs/MLOPS_GUIDE.md: reproducibility and experiment lifecycle.
- docs/OPERATIONS.md: runbook for setup, execution, and troubleshooting.
- docs/KNOWN_GAPS.md: current limitations, impact, and mitigation roadmap.

## 7. Maintenance policy for documentation

Documentation is part of the deliverable, not an afterthought.

When a change affects runtime behavior, outputs, or operating assumptions:

1. Update the relevant docs in the same change set.
2. Keep examples executable with current repository commands.
3. Call out any temporary workaround or known limitation explicitly.
