# Medvision-AI: Complete Docker Workflow Guide

This guide explains how to use Docker for all stages of the Medvision-AI project: data preparation, manifest generation, training, inference, API, UI, and troubleshooting.

---

## 1. Prerequisites

- Docker and Docker Compose installed
- Access to the repository and scripts
- Datasets downloaded (see `scripts/download_*.sh`)

---

## 2. Build the Docker image

```bash
# From the root of the repository
DOCKER_BUILDKIT=1 docker compose build
```

---

## 3. Data preparation & manifest generation

### Example: Chest X-ray segmentation

```bash
cd /opt/medvision-ai
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com \
docker compose run --rm --no-deps api \
bash -lc 'python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml'
```
- The manifest will be generated in `data/processed/chest_xray_segmentation/manifest.csv`

### Other tracks
- Adapt the config file path (`--config ...`).

---

## 4. Model training

### Example: Chest X-ray segmentation

```bash
cd /opt/medvision-ai
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com \
docker compose run --rm --no-deps api \
bash -lc 'python -m src.training.train_segmentation --config configs/chest_xray_segmentation.yaml'
```
- Models are saved in `artifacts/models/chest_xray_segmentation/`

### Other tracks
- Adapt the config file path (`--config ...`).

---

## 5. Launching the API and/or UI (Streamlit)

```bash
# Launch the full stack (API, Streamlit, etc.)
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com docker compose up

# Or just the API
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com docker compose up api

# Or just Streamlit
API_DOMAIN=api.example.com APP_DOMAIN=app.example.com docker compose up streamlit
```

---

## 6. Quick manifest validation (inside Docker)

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

## 7. Tips & best practices

- Always use the `API_DOMAIN` and `APP_DOMAIN` environment variables to avoid configuration issues.
- To make all scripts executable after a git pull:
  ```bash
  find scripts/ -type f -name "*.sh" -exec chmod +x {} \;
  ```
- For brain segmentation, simply adapt the config and folder names.
- For other tasks, adapt the Python script and config file paths.

---

## 8. Troubleshooting

- If a script fails due to execution rights, make it executable with `chmod +x ...`
- If the dataset is incomplete, rerun the download script or check the `data/raw/...` folder
- For any error, check Docker logs (`docker compose logs ...`)
- To force rebuild an image:
  ```bash
  docker compose build --no-cache
  ```
- To clean up unused containers/images:
  ```bash
  docker system prune -af
  ```

---

This guide covers all steps for a robust and reproducible Medvision-AI workflow with Docker.
