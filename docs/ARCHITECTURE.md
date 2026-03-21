# Architecture

## High-level components

### Data layer
- Kaggle download scripts for classification and segmentation datasets
- segmentation manifest builder converting raw image-mask trees into CSV manifests
- DVC stages to make each dataset flow reproducible

### Training layer
- classification trainers for chest X-ray and brain MRI
- multitask segmentation trainer using U-Net
- MLflow logging in every training entry point

### Serving layer
- FastAPI registry-driven inference API
- Streamlit comparison application

### MLOps layer
- DVC for pipelines and remotes
- MLflow for experiment tracking
- Terraform for S3-backed DVC remote infrastructure

## Data flows

### Classification flow
raw dataset -> TensorFlow dataset builder -> classifier -> metrics/report/model -> API/UI

### Segmentation flow
raw segmentation dataset -> manifest builder -> multitask U-Net -> mask metrics + class metrics + overlay -> API/UI

## Why multitask segmentation

The segmentation branch solves two objectives simultaneously:
- localize the pathology with a dense mask,
- recognize the image-level class.

This gives a better software demonstration than pure mask prediction alone and helps compare segmentation vs classification performance inside the same product.
