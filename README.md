# MedVision AI

A production-minded **medical imaging ML project template** built to showcase the exact skills expected for a **Machine Learning Engineer – Computer Vision** role:

- Python
- TensorFlow
- Pandas
- MLflow
- Docker
- medical image preprocessing and evaluation
- optimization of an existing model
- reproducible experimentation

This repository demonstrates a realistic workflow for **binary classification on chest X-ray images** (for example pneumonia vs normal), with:

- a **baseline CNN** to simulate an existing legacy model
- an **optimized transfer learning model** to demonstrate measurable improvement
- **MLflow tracking** for experiments
- **FastAPI** for inference serving
- **Streamlit** for demo visualization
- **Docker Compose** for local reproducibility
- **pytest** tests
- **GitHub Actions CI**

## Why this project ?

It is not just a notebook.
It shows my ability to:

1. understand and improve an existing image model
2. structure a clean ML project
3. handle image preprocessing and metrics correctly
4. track experiments with MLflow
5. package and serve the model with Docker and an API
6. work in a way that resembles a small MedTech or AI product team

## Project structure

```text
medvision-ai/
├── .github/workflows/ci.yml
├── configs/config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docker/
│   └── Dockerfile
├── notebooks/
├── scripts/
│   ├── run_api.sh
│   ├── run_streamlit.sh
│   └── run_training.sh
├── src/
│   ├── api/main.py
│   ├── evaluation/metrics.py
│   ├── inference/predict.py
│   ├── models/baseline_model.py
│   ├── models/optimized_model.py
│   ├── preprocessing/augmentation.py
│   ├── preprocessing/image_loader.py
│   ├── training/train.py
│   └── utils/
│       ├── config.py
│       ├── dataset.py
│       └── paths.py
├── tests/
├── docker-compose.yml
├── requirements.txt
└── streamlit_app.py
```

## Suggested dataset

Use a public chest X-ray dataset such as the Kaggle Chest X-Ray Pneumonia dataset.

Expected folder layout:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

> The dataset is **not included** in this repository.

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Put the dataset in `data/raw/chest_xray`

Then update `configs/config.yaml` if needed.

### 4. Train the baseline model

```bash
python -m src.training.train --config configs/config.yaml --model baseline
```

### 5. Train the optimized model

```bash
python -m src.training.train --config configs/config.yaml --model optimized
```

### 6. Launch MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

### 7. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 8. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Example training commands

```bash
python -m src.training.train --config configs/config.yaml --model baseline
python -m src.training.train --config configs/config.yaml --model optimized --epochs 5
```

## What is actually optimized here?

The project simulates a real business need:

- **Baseline model**: small CNN, simple and fast, but limited performance
- **Optimized model**: transfer learning using MobileNetV2, class weighting, augmentation, better regularization, better monitoring

This gives you a realistic story in an interview:

> “I started from a simple existing model, analyzed its limitations, then improved generalization and training stability through transfer learning, augmentation, and better evaluation tracking.”

## Metrics

The evaluation module computes:

- accuracy
- precision
- recall / sensitivity
- F1-score
- ROC AUC
- confusion matrix

In medical imaging, discussing **recall/sensitivity** and **false negatives** is especially important.

## Docker

Build and run locally:

```bash
docker compose up --build
```

This starts:

- MLflow server on port `5000`
- FastAPI on port `8000`
- Streamlit on port `8501`

## Interview pitch

You can present the project like this:

> I built a medical imaging ML pipeline in Python and TensorFlow, starting from a baseline CNN and improving it with transfer learning, augmentation, and better experiment tracking using MLflow. I packaged the solution with Docker and exposed inference through FastAPI and a small Streamlit demo.

## Nice extensions you can add later

- Grad-CAM explainability
- segmentation with U-Net
- DICOM support with pydicom
- model registry with MLflow
- CI/CD deployment to a VPS or Kubernetes
- data validation and drift monitoring

## License

MIT
