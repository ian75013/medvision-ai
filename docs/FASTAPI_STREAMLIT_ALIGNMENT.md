# FastAPI / Streamlit alignment

FastAPI and Streamlit both consume the same registry from `src/registry/model_registry.py`.

That registry now covers:
- chest X-ray classification
- brain MRI classification
- brain tumor segmentation + classification
- chest X-ray segmentation + classification

This means the UI and API stay aligned as long as artifact names remain consistent with the registry.

## Classification problems
The UI/API return predicted class probabilities.

## Segmentation problems
The UI/API return:
- the image-level predicted class,
- segmentation-derived metadata such as mask foreground ratio,
- overlays in Streamlit.

This common registry pattern is important architecturally because it avoids hard-coding separate serving logic for each experiment.
