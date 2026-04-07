# Transfer learning & fine-tuning guide

## Why the previous scores could stay near 30%

Two issues were especially likely:

1. The training schedule was too short for the MRI task.
2. The Keras application backbones were plugged into datasets normalized to `[0, 1]`, while several pretrained backbones expect either their own preprocessing or values in `[0, 255]`.

The repository now exposes a safer setup:

- warmup with frozen backbone,
- progressive fine-tuning,
- backbone-specific preprocessing inside the model,
- richer set of candidate backbones,
- MLflow logging of the full schedule.

## TensorFlow backbones now supported

- `densenet121`
- `efficientnetv2b0`
- `convnexttiny`
- `resnet50v2`
- `baseline`

## PyTorch backbones for Brain MRI

- `densenet121_torch`
- `resnet50_torch`
- `swin_v2_s_torch`

## Recommended commands

### Chest X-ray

```bash
python -m src.training.train --config configs/config.yaml --model densenet121
python -m src.training.train --config configs/config.yaml --model efficientnetv2b0
```

### Brain MRI (TensorFlow)

```bash
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model densenet121
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model efficientnetv2b0
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model convnexttiny
```

### Brain MRI (PyTorch)

```bash
python -m src.training.train_brain_mri_torch --config configs/brain_tumor_mri.yaml --model densenet121_torch
python -m src.training.train_brain_mri_torch --config configs/brain_tumor_mri.yaml --model swin_v2_s_torch
```

## Practical order of experiments

1. `densenet121` on both classification tasks.
2. `efficientnetv2b0` on both tasks.
3. `convnexttiny` on Brain MRI if GPU memory allows it.
4. `densenet121_torch` on Brain MRI as a cross-framework check.
5. `swin_v2_s_torch` only after the TensorFlow baselines are healthy.

## What to compare in MLflow

Do not look only at raw accuracy.

### Chest X-ray

- accuracy
- balanced accuracy
- recall
- AUC
- F1

### Brain MRI

- accuracy
- macro precision
- macro recall
- macro F1
- top-2 accuracy for TensorFlow runs

## Interpretation

If a pretrained model still stays near chance level after this refactor, investigate:

- corrupted labels or wrong directory structure,
- data leakage or accidental train/test mismatch,
- extreme class imbalance,
- image quality issues,
- overly aggressive resizing.
