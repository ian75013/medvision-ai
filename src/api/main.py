from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.preprocessing.image_loader import load_and_preprocess_image
from src.registry.model_registry import compare_models, get_model_entry, load_registry, load_tf_model

app = FastAPI(title="MedVision AI API", version="2.0.0")


def _predict_with_entry(model_entry: Dict[str, Any], image_path: Path, image_size: int = 224) -> Dict[str, Any]:
    model_path = Path(model_entry["model_path"])
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    model = load_tf_model(str(model_path.resolve()))
    image = load_and_preprocess_image(image_path, image_size=image_size)
    batch = np.expand_dims(image, axis=0)
    raw = model.predict(batch, verbose=0)[0]

    class_names = model_entry["class_names"]
    task_type = model_entry["task_type"]
    if task_type == "binary":
        probability = float(raw[0]) if np.ndim(raw) > 0 else float(raw)
        predicted_class = class_names[1] if probability >= 0.5 else class_names[0]
        probabilities = {
            class_names[0]: float(1.0 - probability),
            class_names[1]: float(probability),
        }
        confidence = max(probabilities.values())
    else:
        probs = np.asarray(raw, dtype=float)
        pred_idx = int(np.argmax(probs))
        predicted_class = class_names[pred_idx]
        probabilities = {name: float(probs[i]) for i, name in enumerate(class_names)}
        confidence = float(probs[pred_idx])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_metadata": {
            "metrics": model_entry.get("metrics", {}),
            "model_path": model_entry["model_path"],
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/registry")
def registry() -> dict[str, Any]:
    return load_registry()


@app.get("/models")
def list_models(problem: str | None = Query(default=None)) -> dict[str, Any]:
    registry = load_registry()
    if problem:
        if problem not in registry["problems"]:
            raise HTTPException(status_code=404, detail=f"Unknown problem: {problem}")
        return registry["problems"][problem]
    return registry


@app.get("/compare")
def compare(problem: str = Query(..., description="Problem id: chest_xray or brain_mri")) -> dict[str, Any]:
    try:
        return {"problem": problem, "rows": compare_models(problem)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    problem: str = Query(..., description="Problem id: chest_xray or brain_mri"),
    model_name: str = Query(..., description="Model id: baseline or optimized"),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        model_entry = get_model_entry(problem, model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = _predict_with_entry(model_entry, tmp_path)
        return {
            "problem": problem,
            "model_name": model_name,
            **result,
        }
    finally:
        tmp_path.unlink(missing_ok=True)
