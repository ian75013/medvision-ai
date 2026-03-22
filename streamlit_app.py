from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing.image_loader import load_and_preprocess_image
from src.registry.model_registry import compare_models, get_model_entry, load_registry, load_tf_model


def _predict(problem: str, model_name: str, image_path: Path, mask_threshold: float = 0.5) -> dict:
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
        pred = max(probs.items(), key=lambda item: item[1])[0]
        confidence = float(max(probs.values()))
        mask = (seg >= mask_threshold).astype(np.float32)
        return {
            "predicted_class": pred,
            "confidence": confidence,
            "probabilities": probs,
            "metrics": model_entry.get("metrics", {}),
            "mask": mask,
            "mask_foreground_ratio": float(mask.mean()),
            "image": image,
        }
    if model_entry["task_type"] == "binary":
        p1 = float(raw[0][0]) if np.ndim(raw) > 1 else float(raw[0])
        probs = {class_names[0]: float(1 - p1), class_names[1]: p1}
    else:
        raw = raw[0]
        probs = {name: float(raw[i]) for i, name in enumerate(class_names)}
    pred = max(probs.items(), key=lambda item: item[1])[0]
    confidence = float(max(probs.values()))
    return {"predicted_class": pred, "confidence": confidence, "probabilities": probs, "metrics": model_entry.get("metrics", {})}


def _inject_styles() -> None:
    st.markdown(
        """
<style>
:root {
  --bg-soft: #f4f6f2;
  --ink-main: #1e2a24;
  --ink-muted: #55645a;
  --card: #ffffff;
  --line: #d8ded7;
  --accent: #0f7a66;
  --accent-2: #c65d2f;
}

.stApp {
  background: radial-gradient(circle at 8% 8%, #eef6f1 0%, #f7f5ee 45%, #fbfaf7 100%);
}

.hero {
  background: linear-gradient(120deg, #133f35 0%, #0f7a66 55%, #2a8d70 100%);
  color: #f4fff9;
  border-radius: 18px;
  padding: 1.3rem 1.4rem;
  margin-bottom: 1rem;
  border: 1px solid rgba(255,255,255,0.2);
  box-shadow: 0 8px 24px rgba(16, 45, 38, 0.18);
}

.hero h2 {
  margin: 0;
  font-size: 1.65rem;
  line-height: 1.15;
}

.hero p {
  margin: 0.45rem 0 0;
  color: #d7f2e8;
  font-size: 0.96rem;
}

.kpi-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.75rem 0.9rem;
}

.kpi-title {
  color: var(--ink-muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.kpi-value {
  color: var(--ink-main);
  font-size: 1.35rem;
  font-weight: 700;
  margin-top: 0.2rem;
}

.pred-card {
  background: #ffffff;
  border: 1px solid #d8ded7;
  border-left: 5px solid var(--accent);
  border-radius: 12px;
  padding: 0.8rem;
  margin-bottom: 0.7rem;
}

.pred-card h4 {
  margin: 0;
  color: #1e2a24;
  font-size: 1rem;
}

.pred-meta {
  margin-top: 0.35rem;
  color: #4a5a51;
  font-size: 0.92rem;
}

[data-testid="stTabs"] button {
  font-weight: 600;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _blend_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = np.clip(image.copy(), 0.0, 1.0)
    alpha = np.clip(mask, 0.0, 1.0) * 0.7
    overlay[..., 1] = np.maximum(overlay[..., 1], alpha)
    overlay[..., 0] = overlay[..., 0] * (1.0 - 0.35 * alpha)
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.15 * alpha)
    return overlay


def _render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-title">{label}</div>
  <div class="kpi-value">{value}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="MedVision AI", layout="wide")
_inject_styles()

st.markdown(
    """
<section class="hero">
  <h2>MedVision AI: Model Comparison Studio</h2>
  <p>Evaluate classification and segmentation models side-by-side with shared registry metadata, prediction confidence, and visual overlays.</p>
</section>
    """,
    unsafe_allow_html=True,
)

registry = load_registry()
problems = registry["problems"]

with st.sidebar:
    st.header("Control Panel")
    problem = st.selectbox("Problem", options=list(problems.keys()), format_func=lambda key: problems[key]["label"])
    show_registry = st.toggle("Show full registry JSON", value=False)

problem_meta = problems[problem]
is_segmentation_problem = problem_meta["task_type"] == "segmentation_multitask"

with st.sidebar:
    if is_segmentation_problem:
        st.markdown("---")
        st.subheader("Segmentation")
        mask_threshold = st.slider("Mask threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    else:
        mask_threshold = 0.5

all_models = problem_meta["models"]
available_models = [name for name, meta in all_models.items() if meta["available"]]

k1, k2, k3, k4 = st.columns(4)
with k1:
    _render_kpi("Problem", problem_meta["label"])
with k2:
    _render_kpi("Task Type", problem_meta["task_type"])
with k3:
    _render_kpi("Models Available", str(len(available_models)))
with k4:
    _render_kpi("Classes", str(len(problem_meta.get("class_names", []))))

tab_compare, tab_predict, tab_registry = st.tabs(["Model Compare", "Prediction Studio", "Registry"])

with tab_compare:
    st.subheader("Model Benchmarks")
    rows = compare_models(problem)
    benchmark_df = pd.DataFrame(rows)
    if benchmark_df.empty:
        st.info("No benchmark rows are currently available.")
    else:
        st.dataframe(benchmark_df, use_container_width=True)
        numeric_cols = [col for col in benchmark_df.columns if col not in {"model_name", "available"}]
        numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(benchmark_df[c])]
        if numeric_cols:
            chart_metric = st.selectbox("Benchmark metric to visualize", options=numeric_cols)
            chart_df = benchmark_df[["model_name", chart_metric]].set_index("model_name")
            st.bar_chart(chart_df, use_container_width=True)

with tab_predict:
    st.subheader("Prediction Studio")
    if not available_models:
        st.warning("No trained models found for this problem in artifacts/models.")
    else:
        selected_models = st.multiselect("Select models", options=available_models, default=available_models)
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            left_col, right_col = st.columns([1.1, 1.4])
            with left_col:
                st.image(uploaded, caption="Input image", use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            try:
                prediction_rows = []
                prob_rows = []
                overlays = []

                with st.spinner("Running predictions..."):
                    for model_name in selected_models:
                        result = _predict(problem, model_name, tmp_path, mask_threshold=mask_threshold)
                        prediction_rows.append(
                            {
                                "model": model_name,
                                "predicted_class": result["predicted_class"],
                                "confidence": float(result.get("confidence", 0.0)),
                                "mask_foreground_ratio": float(result.get("mask_foreground_ratio", 0.0)),
                                **{f"metric_{k}": v for k, v in result["metrics"].items() if isinstance(v, (int, float))},
                            }
                        )
                        prob_row = {"model": model_name}
                        prob_row.update(result["probabilities"])
                        prob_rows.append(prob_row)
                        if "mask" in result:
                            overlays.append((model_name, result["image"], result["mask"]))

                with right_col:
                    st.markdown("#### Prediction Cards")
                    for row in prediction_rows:
                        st.markdown(
                            f"""
<div class="pred-card">
  <h4>{row['model']}</h4>
  <div class="pred-meta">Prediction: <strong>{row['predicted_class']}</strong></div>
  <div class="pred-meta">Confidence: <strong>{row['confidence']:.3f}</strong></div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("#### Prediction Table")
                st.dataframe(pd.DataFrame(prediction_rows), use_container_width=True)

                if prob_rows:
                    st.markdown("#### Class Probabilities")
                    prob_df = pd.DataFrame(prob_rows)
                    st.dataframe(prob_df, use_container_width=True)
                    prob_chart = prob_df.set_index("model")
                    st.bar_chart(prob_chart, use_container_width=True)

                if overlays:
                    st.markdown("#### Segmentation View")
                    seg_rows = []
                    for model_name, image, mask in overlays:
                        seg_rows.append({
                            "model": model_name,
                            "mask_foreground_ratio": float(mask.mean()),
                            "threshold": float(mask_threshold),
                        })
                    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)

                    for model_name, image, mask in overlays:
                        st.markdown(f"##### {model_name}")
                        c1, c2, c3 = st.columns([1, 1, 1.2])
                        with c1:
                            st.caption("Preprocessed image")
                            st.image(image, use_container_width=True)
                        with c2:
                            st.caption("Binary mask")
                            st.image(mask, clamp=True, use_container_width=True)
                        with c3:
                            st.caption("Overlay")
                            st.image(_blend_overlay(image, mask), use_container_width=True)
            finally:
                tmp_path.unlink(missing_ok=True)

with tab_registry:
    st.subheader("Registry Snapshot")
    if show_registry:
        st.json(registry)
    else:
        st.info("Enable 'Show full registry JSON' in the left panel to display raw registry data.")
