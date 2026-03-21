# Datasets

## Classification datasets

### Chest X-ray pneumonia
Expected layout:
- `data/raw/chest_xray/train/NORMAL`
- `data/raw/chest_xray/train/PNEUMONIA`
- `data/raw/chest_xray/val/...`
- `data/raw/chest_xray/test/...`

### Brain MRI tumor classification
Expected layout:
- `data/raw/brain_tumor_mri/Training/glioma`
- `data/raw/brain_tumor_mri/Training/meningioma`
- `data/raw/brain_tumor_mri/Training/notumor`
- `data/raw/brain_tumor_mri/Training/pituitary`
- and the same under `Testing/`

## Segmentation datasets

Segmentation datasets are less standardized. The repository therefore uses a two-step strategy:

1. **Download raw files from Kaggle**
2. **Build a manifest automatically** by matching images and masks through filename heuristics

## Manifest schema

The generated CSV contains:
- `image_path`
- `mask_path`
- `label`
- `split`

This decouples training code from the original dataset folder structure.

## Important caveat

Heuristic image-mask matching is practical but not infallible. Always inspect the first rows of the generated manifest and visualize a few image/mask pairs before starting long experiments.
