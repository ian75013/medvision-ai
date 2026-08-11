# FastAPI and Streamlit Alignment

This document defines how API and UI stay behaviorally consistent across all supported tracks.

## 1. Shared source of truth

Both FastAPI and Streamlit consume metadata from src/registry/model_registry.py.

The registry provides:

- problem identifiers and labels
- task type
- class names
- model availability
- metrics metadata
- artifact paths

As long as artifact naming and registry mappings are consistent, both serving surfaces remain aligned.

## 2. Supported problem families

Current registry coverage includes:

- chest_xray
- brain_mri
- brain_tumor_segmentation
- chest_xray_segmentation

Each problem declares task_type, which controls inference payload semantics.

## 3. Prediction contract by task type

### 3.1 Classification tasks

Expected outputs:

- predicted_class
- confidence
- probabilities by class

### 3.2 Multitask segmentation tasks

Expected outputs:

- image-level predicted_class
- confidence and class probabilities
- segmentation-derived metadata (for example mask foreground ratio)
- optional overlay rendering in Streamlit

## 4. Why this alignment matters

Without a shared registry contract, API and UI can silently diverge when:

- model files are renamed
- class order changes
- new tasks are introduced

The registry-centered approach minimizes drift and keeps behavior inspectable.

## 5. Change protocol for contributors

When adding or changing a model track:

1. Update model_registry.py entries.
2. Ensure artifact names match declared candidates.
3. Validate API endpoints:
	GET /registry, GET /models, POST /predict.
4. Validate Streamlit comparison and prediction outputs.
5. Update docs if task semantics changed.

## 6. Validation checklist

After training a new model:

1. Confirm model appears in GET /registry.
2. Confirm model appears in Streamlit model selection.
3. Run POST /predict on a sample image.
4. Compare API output with Streamlit output for same model/image.

If any mismatch appears, treat it as a contract issue between artifacts, registry metadata, and serving code.
