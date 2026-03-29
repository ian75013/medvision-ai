#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/4] Chest X-ray classification"
python -m src.training.train --config configs/config.yaml --model optimized

echo "==> [2/4] Brain MRI classification"
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized

echo "==> [3/6] Prepare brain tumor segmentation manifest"
python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml

echo "==> [4/6] Brain tumor segmentation"
python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml

echo "==> [5/6] Prepare chest X-ray segmentation manifest"
if [[ -d "data/raw/chest_xray_segmentation" ]]; then
	python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml

	echo "==> [6/6] Chest X-ray segmentation"
	python -m src.segmentation.train_segmentation --config configs/chest_xray_segmentation.yaml
else
	echo "Chest X-ray segmentation dataset missing in data/raw/chest_xray_segmentation: skipping step [5/6] and [6/6]."
fi

echo ""
echo "All training jobs completed."
