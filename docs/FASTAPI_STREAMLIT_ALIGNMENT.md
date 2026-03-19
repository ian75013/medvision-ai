# FastAPI and Streamlit alignment

## What changed

Before this update:
- FastAPI only served one chest X-ray model
- Streamlit only loaded one chest X-ray model

After this update:
- both layers use the same registry
- both support `baseline` and `optimized` models for both problems
- both can expose stored metrics for performance comparison

## FastAPI endpoints

### `GET /models`
Returns the full registry or one problem registry.

### `GET /compare?problem=chest_xray`
Returns comparison rows built from saved metrics.

### `POST /predict?problem=brain_mri&model_name=optimized`
Runs inference with the selected trained model.

## Streamlit behavior

The Streamlit app now:
- lets you choose a problem
- shows a comparison table of available models and metrics
- lets you upload one image and compare predictions from several trained models

## Current limitation

The UI and API rely on artifact naming conventions. If you train a new model with a custom filename, either rename it to match the convention or extend `src/registry/model_registry.py`.
