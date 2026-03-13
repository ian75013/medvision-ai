from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predict import load_model, predict_from_path

app = FastAPI(title="MedVision AI API", version="1.0.0")

MODEL_PATH = Path("artifacts/models/optimized_model.keras")
_model = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    global _model

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail="Model file not found. Train the model first.")
        _model = load_model(MODEL_PATH)

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = predict_from_path(_model, tmp_path)
        return result
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
