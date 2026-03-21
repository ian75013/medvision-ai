# Operations

## Launch local MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Launch FastAPI

```bash
uvicorn src.api.main:app --reload
```

## Launch Streamlit

```bash
streamlit run streamlit_app.py
```

## Run a full segmentation pipeline with DVC

```bash
dvc repro train_brain_tumor_segmentation
dvc repro train_chest_xray_segmentation
```
