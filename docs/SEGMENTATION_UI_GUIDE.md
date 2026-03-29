# Segmentation UI Guide

This guide explains how to use the segmentation UI and related scripts in Medvision-AI. For a complete Docker workflow, see `DOCKER_WORKFLOW_GUIDE.md`.

---

## 1. Prerequisites

- Docker and Docker Compose installed
- Dataset downloaded (see `scripts/download_chest_xray_seg.sh`)
- All scripts are executable (`find scripts/ -type f -name "*.sh" -exec chmod +x {} \;`)

---

## 2. Data Preparation & Manifest Generation

Run the following command to generate the manifest for chest X-ray segmentation:

```bash
cd /opt/medvision-ai
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com \
docker compose run --rm --no-deps api \
bash -lc 'python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml'
```

The manifest will be generated in `data/processed/chest_xray_segmentation/manifest.csv`.

---

## 3. Model Training

Train the segmentation model with:

```bash
cd /opt/medvision-ai
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com \
docker compose run --rm --no-deps api \
bash -lc 'python -m src.training.train_segmentation --config configs/chest_xray_segmentation.yaml'
```

Models are saved in `artifacts/models/chest_xray_segmentation/`.

---

## 4. Launching the UI (Streamlit) and API

To launch the full stack (API, Streamlit, etc.):

```bash
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com docker compose up
```

To launch only the API:

```bash
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com docker compose up api
```

To launch only Streamlit:

```bash
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com docker compose up streamlit
```

---

## 5. Quick Manifest Validation (inside Docker)

```bash
cd /opt/medvision-ai
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com \
docker compose run --rm --no-deps api \
bash -lc 'python - <<"PY"
import pandas as pd
df = pd.read_csv("data/processed/chest_xray_segmentation/manifest.csv")
print("rows", len(df))
print("labels", df["label"].value_counts(dropna=False).to_dict())
print("splits", df["split"].value_counts(dropna=False).to_dict())
print(pd.crosstab(df["split"], df["label"]))
PY'
```

---

## 6. Troubleshooting

- If a script fails due to execution rights, make it executable with `chmod +x ...`
- If the dataset is incomplete, rerun the download script or check the `data/raw/chest_xray_segmentation/` folder
- For any error, check Docker logs (`docker compose logs ...`)

---

For a full Docker workflow and advanced usage, see `docs/DOCKER_WORKFLOW_GUIDE.md`.
