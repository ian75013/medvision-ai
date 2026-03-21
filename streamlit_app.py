from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing.image_loader import load_and_preprocess_image
from src.registry.model_registry import compare_models, get_model_entry, load_registry, load_tf_model


def _predict(problem: str, model_name: str, image_path: Path) -> dict:
    model_entry = get_model_entry(problem, model_name)
    model = load_tf_model(str(Path(model_entry["model_path"]).resolve()))
    image_size = 256 if "segmentation" in problem else 224
    image = load_and_preprocess_image(image_path, image_size=image_size)
    batch = np.expand_dims(image, axis=0)
    raw = model.predict(batch, verbose=0)
    class_names = model_entry["class_names"]
    if model_entry["task_type"] == "segmentation_multitask":
        seg = raw["segmentation_output"][0, ..., 0]
        cls = raw["classification_output"][0]
        if len(class_names) == 2:
            probs = {class_names[0]: float(1 - cls[0]), class_names[1]: float(cls[0])}
        else:
            probs = {name: float(cls[i]) for i, name in enumerate(class_names)}
        pred = max(probs, key=probs.get)
        return {
            "predicted_class": pred,
            "probabilities": probs,
            "metrics": model_entry.get("metrics", {}),
            "mask": (seg >= 0.5).astype(np.float32),
            "image": image,
        }
    if model_entry["task_type"] == "binary":
        p1 = float(raw[0][0]) if np.ndim(raw) > 1 else float(raw[0])
        probs = {class_names[0]: float(1 - p1), class_names[1]: p1}
    else:
        raw = raw[0]
        probs = {name: float(raw[i]) for i, name in enumerate(class_names)}
    pred = max(probs, key=probs.get)
    return {"predicted_class": pred, "probabilities": probs, "metrics": model_entry.get("metrics", {})}


st.set_page_config(page_title="MedVision AI", layout="wide")
st.title("MedVision AI — Multi-model comparison")
st.caption("Classification and segmentation experiments tracked with MLflow and DVC.")

registry = load_registry()
problems = registry["problems"]
problem = st.selectbox("Choose a problem", options=list(problems.keys()), format_func=lambda key: problems[key]["label"])

left, right = st.columns([1, 1])
with left:
    st.subheader("Available models")
    rows = compare_models(problem)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
with right:
    st.subheader("Prediction lab")
    available_models = [name for name, meta in problems[problem]["models"].items() if meta["available"]]
    if not available_models:
        st.warning("No trained models found for this problem in artifacts/models.")
    else:
        selected_models = st.multiselect("Models to compare", options=available_models, default=available_models)
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)
            try:
                prediction_rows = []
                prob_rows = []
                masks = []
                for model_name in selected_models:
                    result = _predict(problem, model_name, tmp_path)
                    prediction_rows.append({
                        "model": model_name,
                        "predicted_class": result["predicted_class"],
                        **{f"metric_{k}": v for k, v in result["metrics"].items() if isinstance(v, (int, float))},
                    })
                    prob_row = {"model": model_name}
                    prob_row.update(result["probabilities"])
                    prob_rows.append(prob_row)
                    if "mask" in result:
                        masks.append((model_name, result["image"], result["mask"]))
                if prediction_rows:
                    st.markdown("#### Predictions")
                    st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True)
                if prob_rows:
                    st.markdown("#### Class probabilities")
                    st.dataframe(pd.DataFrame(prob_rows), use_container_width=True)
                if masks:
                    st.markdown("#### Segmentation overlays")
                    cols = st.columns(len(masks))
                    for col, (model_name, image, mask) in zip(cols, masks):
                        overlay = image.copy()
                        overlay[..., 0] = np.maximum(overlay[..., 0], mask)
                        with col:
                            st.caption(model_name)
                            st.image(overlay, use_container_width=True)
            finally:
                tmp_path.unlink(missing_ok=True)

st.markdown("---")
st.subheader("Registry snapshot")
st.json(registry)
