from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from src.inference.predict import load_model, predict_from_path

st.set_page_config(page_title="MedVision AI", layout="centered")
st.title("MedVision AI - Medical Image Demo")
st.write("Upload a chest X-ray image to test the trained classification model.")

model_path = Path("artifacts/models/optimized_model.keras")

if not model_path.exists():
    st.warning("No trained model found yet. Train the optimized model first.")
    st.stop()

model = load_model(model_path)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    result = predict_from_path(model, tmp_path)
    tmp_path.unlink(missing_ok=True)

    st.subheader("Prediction")
    st.write(f"**Predicted class:** {result['predicted_class']}")
    st.write(f"**Pneumonia probability:** {result['probability_pneumonia']:.4f}")
