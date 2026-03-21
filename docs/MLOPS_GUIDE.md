# MLOps Guide

## What is tracked where?

### Git
- code
- configs
- docs
- notebooks
- DVC metadata

### DVC
- raw datasets
- processed manifests
- trained models
- overlays
- metric JSON files as pipeline outputs

### MLflow
- run parameters
- scalar metrics
- artifacts per run

## Typical workflow

1. Download or refresh datasets with DVC.
2. Prepare segmentation manifests.
3. Train models.
4. Compare runs in MLflow.
5. Compare saved artifacts in Streamlit.
6. Push large artifacts to a DVC remote.

## Why this matters pedagogically

A machine-learning project becomes an engineering project when you can answer these questions reliably:
- Which data produced this model?
- Which parameters were used?
- Which code version trained it?
- Where are the artifacts?
- How do I reproduce the run?

This repository is now structured to answer those questions for both classification and segmentation.
