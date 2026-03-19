# MedVision MRI - Sprints 1 & 2

This extension turns the original MedVision repo into a more engineering-friendly foundation for brain MRI classification.

## What is included

### Sprint 1 - repo refactor foundation
- `src/dataio/` for medical volume loading
- `src/preprocessing/brain_mri_2d.py` for MRI-specific preprocessing
- `src/datasets/` for patient-level split creation and PyTorch datasets
- `src/models/classification_2d/simple_cnn.py` as a clear baseline
- `src/training/train_classifier.py` and `src/training/trainer.py`
- `src/inference/predict_classifier.py`
- config-driven training with `configs/brain_mri_2d_demo.yaml`

### Sprint 2 - demonstrative brain MRI 2D baseline
- synthetic MRI dataset generator: `scripts/generate_demo_brain_mri_dataset.py`
- patient-level train/val/test splitting
- 3D volume -> central K slices conversion
- a minimal PyTorch 2D classifier to validate the workflow end-to-end

## Demo commands

From the repository root:

```bash
python -m scripts.generate_demo_brain_mri_dataset --output-dir data/raw/brain_mri_demo --processed-dir data/processed/brain_mri_demo
python -m src.training.train_classifier --config configs/brain_mri_2d_demo.yaml
```

## Real project next step
Replace the synthetic `.npy` files with real BraTS `.nii.gz` files and install `nibabel`.
Then create a metadata CSV with:
- `patient_id`
- `path`
- `label`

The same training entry point can then be reused.
