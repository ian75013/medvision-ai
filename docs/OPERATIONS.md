# Operations

This document is the practical runbook for local operation, verification, and troubleshooting.

## 1. Environment setup

### 1.1 Create and activate virtual environment (recommended: uv)

Windows PowerShell:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Linux/WSL:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 1.2 Legacy alternative (venv + pip)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 Configure Kaggle token

- Windows: %USERPROFILE%/.kaggle/kaggle.json
- Linux/WSL: ~/.kaggle/kaggle.json

Linux/WSL permission:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 2. Core services

### 2.1 Start MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Default URL: http://127.0.0.1:5000

### 2.2 Start FastAPI

```bash
uvicorn src.api.main:app --reload
```

Default URL: http://127.0.0.1:8000

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 2.3 Start Streamlit

```bash
streamlit run streamlit_app.py
```

Default URL: http://127.0.0.1:8501

## 3. Pipeline runbooks

### 3.0 DVC initialization check (one-time)

At repository root, verify whether .dvc exists.

- If .dvc exists: skip initialization.
- If .dvc is missing: run dvc init once, then continue.

Commands:

```bash
dvc init
dvc status
```

### 3.1 Quick smoke run (recommended after setup)

```bash
python -m src.data.download_dataset
python -m src.training.train --config configs/config.yaml --model optimized --epochs 1
```

### 3.2 Classification full runs

```bash
python -m src.data.download_dataset
python -m src.training.train --config configs/config.yaml --model optimized

python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized
```

### 3.3 Segmentation full runs

```bash
python -m src.data.download_segmentation_dataset --problem brain_tumor_seg
python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml
python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml

python -m src.data.download_segmentation_dataset --problem chest_xray_seg
python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml
python -m src.segmentation.train_segmentation --config configs/chest_xray_segmentation.yaml
```

### 3.4 DVC stage execution

```bash
dvc status
dvc repro train_brain_tumor_segmentation
dvc repro train_chest_xray_segmentation
```

## 4. Validation runbook

After each training cycle, verify:

1. Expected model files exist under artifacts/models.
2. Expected metrics files exist under artifacts/reports.
3. Segmentation overlays exist under artifacts/overlays.
4. Models are visible in GET /registry and Streamlit model list.

## 5. Docker-based operation

Bring up all services:

```bash
docker compose up --build
```

Shutdown:

```bash
docker compose down
```

Use compose for integrated demos; use individual service commands for debugging.

## 6. Incident and troubleshooting checklist

### 6.1 Kaggle download failures

1. Validate kaggle.json path.
2. Validate permissions on Linux/WSL.
3. Test kaggle CLI independently.

### 6.2 FastAPI model missing in /models or /registry

1. Confirm .keras artifact exists in artifacts/models.
2. Confirm metrics/report files exist when expected.
3. Confirm registry candidate names in src/registry/model_registry.py.

### 6.3 Segmentation quality unexpectedly poor

1. Inspect generated manifest rows.
2. Visualize random image/mask pairs.
3. Re-run short training to isolate data issues.

### 6.4 DVC output mismatch

1. Run dvc status.
2. Re-run specific stage with dvc repro.
3. Check whether config or params changed since last run.

## 7. Minimal pre-PR operational checks

1. Run pytest -q.
2. Run one short training command relevant to your change.
3. Verify API health endpoint.
4. Verify no accidental artifact naming drift.
