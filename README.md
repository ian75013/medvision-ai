> Legal notice  
> This repository is proprietary software owned by Doctum Consilium.  
> No use, reproduction, modification, or redistribution is permitted without prior written authorization.

# MedVision AI

This repository is our working base for medical imaging experiments.
It is structured so a new team member can get from setup to a first successful run quickly, then understand how data, training, evaluation, and serving fit together.

We currently maintain four runnable tracks:

1. Chest X-ray pneumonia classification
2. Brain MRI tumor classification
3. Brain tumor segmentation + classification (multitask U-Net)
4. Chest X-ray lung segmentation + pneumonia classification (multitask U-Net)

Main stack:

- TensorFlow/Keras for training and inference
- MLflow for experiment tracking
- DVC for reproducible data/model pipelines
- FastAPI for inference endpoints
- Streamlit for model comparison
- Terraform for S3-backed DVC remote infrastructure

## 1. Repository layout

```text
configs/                  YAML configs per task
data/                     raw and processed datasets
artifacts/                trained models, reports, overlays
src/api/                  FastAPI application
src/data/                 dataset download and preparation
src/training/             classification training entry points
src/segmentation/         multitask segmentation pipeline
src/registry/             shared model registry (API + Streamlit)
docs/                     architecture, operations, MLOps notes
notebooks/                experimentation notebooks
scripts/                  helper scripts (shell, ps1, bat)
docker/                   Dockerfile
terraform/aws_dvc_remote/ Terraform for DVC remote on AWS
tests/                    unit and API tests
```

## 2. Architecture (what matters)

### Layered view

Data layer:

- Kaggle download scripts for classification and segmentation datasets
- Segmentation manifest builder turning raw image/mask trees into deterministic CSV manifests
- DVC stages that capture each data flow for reproducibility

Training layer:

- Classification trainers for chest X-ray and brain MRI
- Multitask U-Net trainer for segmentation + image-level classification
- MLflow logging in training entry points (metrics, params, artifacts)

Serving layer:

- FastAPI for programmatic inference
- Streamlit for side-by-side model comparison and overlays

MLOps layer:

- DVC for pipeline orchestration and remotes
- MLflow for run tracking
- Terraform for S3 remote provisioning

### Data flows

Classification flow:

raw dataset -> TensorFlow dataset builder -> classifier -> metrics/report/model -> API/UI

Segmentation flow:

raw segmentation dataset -> manifest builder -> multitask U-Net -> mask metrics + class metrics + overlays -> API/UI

### Why multitask segmentation

The segmentation branch solves two objectives in one model:

- localize pathology with a dense mask
- predict an image-level class

This makes evaluation and demos more practical than pure mask prediction alone and allows direct comparison against classification-first tracks.

### API/UI alignment by design

FastAPI and Streamlit both consume the same registry from src/registry/model_registry.py.
As long as artifact naming stays consistent, both serving surfaces remain aligned without duplicating task-specific logic.

## 3. Prerequisites

- Python 3.10 to 3.12 recommended
- Git
- Kaggle API token for dataset download (kaggle.json)
- Optional: Docker Desktop
- Optional: CUDA GPU for faster training

Important notes:

- TensorFlow 2.16.1 is not published for Python 3.13+ in this setup.
- requirements.txt includes DVC with S3 and Google Drive remotes.

If you are new to the project, use Python 3.12 to avoid most compatibility issues.

## 4. Installation (quick path)

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or WSL

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Kaggle setup

1. Create an API token from your Kaggle account.
2. Place kaggle.json in:

- Windows: %USERPROFILE%/.kaggle/kaggle.json
- Linux/WSL: ~/.kaggle/kaggle.json

3. On Linux/WSL, set file permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

Without this file, dataset download commands will fail.

## 6. First successful run (10 minutes)

Goal: validate local setup end-to-end (download + short train + API).

1. Download the chest X-ray dataset

```bash
python -m src.data.download_dataset
```

2. Run a short training

```bash
python -m src.training.train --config configs/config.yaml --model optimized --epochs 1
```

3. Start the API

```bash
uvicorn src.api.main:app --reload
```

4. Check the health endpoint

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

If you get this response, your local environment is ready for full workflows.

## 7. Training workflows by track

### 7.1 Chest X-ray classification

```bash
python -m src.data.download_dataset
python -m src.training.train --config configs/config.yaml --model optimized
```

Main outputs:

- artifacts/models/optimized_model.keras
- artifacts/reports/optimized_metrics.json
- artifacts/reports/optimized_classification_report.txt

This is the fastest track to onboard with.

### 7.2 Brain MRI classification

```bash
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized
```

Main outputs:

- artifacts/models/brain_mri_optimized.keras
- artifacts/reports/brain_mri_metrics.json

This is a multi-class track (glioma, meningioma, notumor, pituitary), so metrics interpretation differs from binary classification.

### 7.3 Brain tumor segmentation (multitask)

```bash
python -m src.data.download_segmentation_dataset --problem brain_tumor_seg
python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml
python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml
```

Main outputs:

- data/processed/brain_tumor_segmentation/manifest.csv
- artifacts/models/brain_tumor_segmentation_unet.keras
- artifacts/reports/brain_tumor_segmentation_unet_metrics.json
- artifacts/overlays/brain_tumor_segmentation_unet_sample_overlay.png

The generated manifest in data/processed is critical because it defines image/mask pairs used by training.

### 7.4 Chest X-ray segmentation (multitask)

```bash
python -m src.data.download_segmentation_dataset --problem chest_xray_seg
python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml
python -m src.segmentation.train_segmentation --config configs/chest_xray_segmentation.yaml
```

Main outputs:

- data/processed/chest_xray_segmentation/manifest.csv
- artifacts/models/chest_xray_segmentation_unet.keras
- artifacts/reports/chest_xray_segmentation_unet_metrics.json
- artifacts/overlays/chest_xray_segmentation_unet_sample_overlay.png

Same principle as brain segmentation: if the manifest is wrong, training quality is compromised.

## 8. FastAPI

Run:

```bash
uvicorn src.api.main:app --reload
```

Useful endpoints:

- GET /health
- GET /registry
- GET /models
- GET /models?problem=brain_mri
- GET /compare?problem=chest_xray
- POST /predict?problem=brain_mri&model_name=optimized
- POST /predict?problem=brain_tumor_segmentation&model_name=unet_multitask

The registry is built from available artifacts. If a model does not appear, first check that a .keras file exists in artifacts/models.

Prediction example (PowerShell):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?problem=brain_mri&model_name=optimized" -F "file=@C:/path/to/image.png"
```

## 9. Streamlit

Run:

```bash
streamlit run streamlit_app.py
```

Features:

- Compare available models per problem
- Upload an image and run multi-model predictions
- View class probability tables
- Visualize segmentation overlays
- Inspect a live registry snapshot

In day-to-day usage, Streamlit is the fastest way to compare models before digging into MLflow run details.

## 10. MLflow

Run local UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Default URL: http://127.0.0.1:5000

Each run typically logs:

- training parameters
- scalar metrics
- model artifacts
- reports and training histories
- segmentation overlays

Tip: keep MLflow open during training to detect metric drift early.

## 11. DVC

The pipeline is already defined in dvc.yaml.

When you change data or training logic, think in DVC stages first. It keeps reproducibility clean.

Common commands:

```bash
dvc repro
dvc status
dvc exp run
dvc exp show
dvc push
dvc pull
```

Main stages:

- download_chest_xray
- train_chest_xray
- download_brain_mri
- train_brain_mri
- download_brain_tumor_segmentation
- prepare_brain_tumor_segmentation
- train_brain_tumor_segmentation
- download_chest_xray_segmentation
- prepare_chest_xray_segmentation
- train_chest_xray_segmentation

Run one stage:

```bash
dvc repro train_brain_tumor_segmentation
```

For onboarding, reproducing one stage is usually more useful than running the full graph immediately.

## 12. Docker Compose

docker-compose.yml starts three services:

- mlflow (port 5000)
- api (port 8000)
- streamlit (port 8501)

Commands:

```bash
docker compose up --build
docker compose down
```

Good for a full local demo. For debugging, running services one by one is often easier.

## 13. Helper scripts

Available scripts:

- scripts/run_api.sh
- scripts/run_streamlit.sh
- scripts/run_training.sh
- scripts/download_dataset.ps1
- scripts/download_dataset.bat

They call the same Python modules documented above.

Use them as shortcuts, but it is worth knowing the direct Python commands.

## 14. Tests

Run the suite:

```bash
pytest -q
```

Example existing test:

- test API health endpoint

Before opening a PR, run at least this test suite to catch obvious regressions.

## 15. Configuration and tuning

- Global hyperparameters in params.yaml
- Chest X-ray config in configs/config.yaml
- Brain MRI config in configs/brain_tumor_mri.yaml
- Brain segmentation config in configs/brain_tumor_segmentation.yaml
- Chest segmentation config in configs/chest_xray_segmentation.yaml

To reduce runtime during local checks, lower epochs in params.yaml or pass --epochs where supported.

Do not commit very low epoch values to main configs without calling it out in your PR.

## 16. Troubleshooting

### Kaggle API error (401/403)

- Verify kaggle.json exists
- Verify file permissions on Linux/WSL
- Test with kaggle datasets list

In most cases, the issue is token location or file permissions.

### TensorFlow error on Python 3.13+

- Use Python 3.12 for this environment
- Recreate a clean venv and reinstall requirements.txt

If needed, start from a clean environment instead of stacking dependency workarounds.

### No model available in Streamlit or API

- Check that .keras files exist in artifacts/models
- Check that metrics json files exist in artifacts/reports

Serving only exposes what actually exists in artifacts.

### DVC cannot find outputs

- Run dvc status
- Re-run the relevant stage with dvc repro
- Check data/raw and data/processed paths

After config changes, dvc status is usually the fastest way to see what is out of sync.

## 17. Recommended onboarding order

1. Set up environment and Kaggle token
2. Run the short chest X-ray training
3. Open MLflow and inspect runs
4. Open Streamlit and compare models
5. Execute one full segmentation pipeline
6. Use dvc exp run for controlled iteration
7. Read architecture and operations docs

## 18. Additional documentation

- docs/ARCHITECTURE.md
- docs/PROJECT_GUIDE.md
- docs/OPERATIONS.md
- docs/MLOPS_GUIDE.md
- docs/COMPONENTS.md
- docs/KNOWN_GAPS.md
- README_DVC.md
- README_MRI_SPRINTS.md

If anything is unclear during onboarding, update this README directly so the next person benefits from it.
