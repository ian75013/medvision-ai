#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/4] Chest X-ray classification"
python -m src.training.train --config configs/config.yaml --model optimized

echo "==> [2/4] Brain MRI classification"
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized

echo "==> [3/4] Brain tumor segmentation"
python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml

echo "==> [4/4] Chest X-ray segmentation"
python -m src.segmentation.train_segmentation --config configs/chest_xray_segmentation.yaml

echo ""
echo "All training jobs completed."
