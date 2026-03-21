> **Legal notice**  
> This repository is proprietary software owned by Doctum Consilium.  
> No use, reproduction, modification, or redistribution is permitted without prior written authorization.

# MedVision AI

MedVision AI is a medical-imaging experimentation platform covering **four runnable tracks**:

1. **Chest X-ray pneumonia classification**
2. **Brain MRI tumor classification**
3. **Brain tumor segmentation + classification (multitask U-Net)**
4. **Chest X-ray lung segmentation + pneumonia classification (multitask U-Net)**

The repository combines:
- **TensorFlow/Keras** training for classification and segmentation
- **MLflow** for experiment tracking and artifact logging
- **DVC** for dataset / model pipeline reproducibility
- **FastAPI** for inference APIs
- **Streamlit** for side-by-side model comparison
- **Terraform** for provisioning an S3 bucket usable as a DVC remote
- **Notebooks** for experimentation, explainability, and model comparison

## 1. Project layout

```text
configs/                  YAML configs for each task
src/api/                  FastAPI app
src/data/                 dataset download / preparation scripts
src/models/               classification models
src/segmentation/         segmentation datasets, models, metrics, training, overlays
src/training/             classification training entry points
src/evaluation/           classification metrics and reports
src/registry/             model registry used by API and Streamlit
scripts/                  helper shell scripts
docs/                     architecture and MLOps documentation
notebooks/                experimental notebooks
terraform/                S3 bucket infra for DVC remote
```

## 2. Installation

### Create a virtual environment

```bash
python -m venv .venv-wsl
source .venv-wsl/bin/activate        # Linux / macOS
# .venv-wsl\Scriptsctivate        # Windows PowerShell
pip install -r requirements.txt
```

### Kaggle authentication

Create a Kaggle API token and place it here:
- Linux/macOS/WSL: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`

Then restrict permissions on Linux/WSL:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 3. Classification datasets

### Chest X-ray pneumonia

```bash
python -m src.data.download_dataset
python -m src.training.train --config configs/config.yaml --model optimized
```

### Brain MRI tumor classification

```bash
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized
```

## 4. Segmentation datasets

### Brain tumor segmentation + tumor-type recognition

```bash
python -m src.data.download_segmentation_dataset --problem brain_tumor_seg
python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml
python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml
```

### Chest X-ray segmentation + pneumonia recognition

```bash
python -m src.data.download_segmentation_dataset --problem chest_xray_seg
python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml
python -m src.segmentation.train_segmentation --config configs/chest_xray_segmentation.yaml
```

The segmentation training uses a **multitask U-Net**:
- one head predicts a segmentation mask,
- one head predicts the image-level class.

This is useful because, mathematically, the model learns both:
- a **dense pixel map** `g(x)` for segmentation,
- and a **global label** `f(x)` for classification.

## 5. FastAPI and Streamlit

### FastAPI

```bash
uvicorn src.api.main:app --reload
```

Useful endpoints:
- `GET /health`
- `GET /registry`
- `GET /models`
- `GET /compare?problem=brain_mri`
- `POST /predict?problem=brain_tumor_segmentation&model_name=unet_multitask`

### Streamlit

```bash
streamlit run streamlit_app.py
```

The UI now supports:
- classification model comparison,
- segmentation model comparison,
- overlay visualization for segmentation models,
- metrics comparison across tasks.

## 6. DVC and MLflow

### DVC

```bash
dvc init
dvc repro
dvc exp run
dvc exp show
```

DVC stages now exist for:
- classification downloads and training,
- segmentation downloads,
- segmentation manifest preparation,
- segmentation training.

### MLflow

By default tracking is local:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Each training script logs:
- parameters,
- metrics,
- model artifacts,
- reports,
- sample overlays for segmentation.

## 7. Notebooks

The `notebooks/` folder now contains notebooks for:
- classification baselines,
- Grad-CAM and weak localization,
- U-Net segmentation experiments,
- ConvNeXt / ViT style innovation tracks.

## 8. Terraform and DVC remote

Terraform under `terraform/aws_dvc_remote/` provisions an S3 bucket suitable for a DVC remote.
You can also configure a Google Drive remote with DVC thanks to `dvc[gdrive]` in `requirements.txt`.

## 9. Important note on datasets

Your current classification datasets are image-level labeled datasets.
The added segmentation pipeline expects datasets that contain **image / mask pairs**.
The `prepare_segmentation_dataset` step builds a manifest automatically by matching image files to mask files using filename heuristics.

That means the pipeline is now concrete and runnable, but you should still inspect each downloaded dataset structure to validate the image-mask matching quality before long training runs.

## 10. Recommended learning order

1. Run chest X-ray classification.
2. Run brain MRI classification.
3. Inspect the segmentation manifests.
4. Train the multitask U-Net models.
5. Compare everything in Streamlit and MLflow.
6. Iterate with DVC experiments and notebooks.
