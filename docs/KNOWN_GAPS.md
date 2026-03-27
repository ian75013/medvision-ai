# Known Gaps

This document tracks current limitations and practical mitigation guidance.

## 1. Data and labeling limitations

### 1.1 Heuristic segmentation matching

Gap:

- Segmentation manifest generation relies on filename and path heuristics.

Impact:

- Incorrect image/mask pairing can degrade training quality and qualitative overlays.

Current mitigation:

- Visual validation of random manifest samples before long runs.

### 1.2 Annotation quality dependency

Gap:

- Class semantics and mask quality depend on source dataset annotations.

Impact:

- Reported model metrics may reflect dataset noise more than model limitations.

Current mitigation:

- Keep dataset provenance explicit and compare multiple runs before conclusions.

## 2. Modeling and scope limitations

### 2.1 2D segmentation only

Gap:

- Current segmentation implementation is 2D TensorFlow/Keras.

Impact:

- Limited representation for full volumetric MRI segmentation use cases.

Current mitigation:

- Position current results as 2D baselines and avoid over-claiming volumetric capability.

### 2.2 Limited post-training lifecycle features

Gap:

- No active learning loop and no continuous production monitoring pipeline.

Impact:

- Reduced feedback loops after deployment-style serving.

Current mitigation:

- Manual review in MLflow and Streamlit after each experiment cycle.

## 3. Operational limitations

### 3.1 Artifact naming sensitivity

Gap:

- Registry discovery depends on expected artifact naming conventions.

Impact:

- Valid models may not appear in API/UI if names drift from registry candidates.

Current mitigation:

- Keep artifact names stable and update registry definitions when introducing new models.

## 4. Recommended roadmap

Short-term improvements:

1. Add stricter manifest validation reports.
2. Add lightweight contract checks between registry and artifact outputs.
3. Add smoke tests for API/UI parity on at least one model per task.

Mid-term improvements:

1. Introduce optional curated mapping files for segmentation pairing.
2. Add richer error reporting for missing artifact contracts.
3. Add basic monitoring hooks for inference requests and outputs.

Long-term improvements:

1. Evaluate volumetric 3D segmentation path for MRI tracks.
2. Introduce active learning and drift-aware retraining loops.
