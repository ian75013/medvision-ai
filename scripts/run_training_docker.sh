#!/usr/bin/env bash
set -euo pipefail

run_training_step() {
	local label="$1"
	local cmd="$2"
	echo "==> ${label}"
	API_DOMAIN="${API_DOMAIN:-api.example.com}" APP_DOMAIN="${APP_DOMAIN:-app.example.com}" \
		docker compose run --rm --no-deps api bash -lc "${cmd}"
}

run_training_step "[1/4] Chest X-ray classification" \
	"python -m src.training.train --config configs/config.yaml --model optimized"

run_training_step "[2/4] Brain MRI classification" \
	"python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized"

run_training_step "[3/6] Prepare brain tumor segmentation manifest" \
	"python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml"

run_training_step "[4/6] Brain tumor segmentation" \
	"python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml"

run_training_step "[5/6] Prepare chest X-ray segmentation manifest" \
	"python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml"

run_training_step "[6/6] Chest X-ray segmentation" \
	"python -m src.segmentation.train_segmentation --config configs/chest_xray_segmentation.yaml"

echo ""
echo "All Docker training jobs completed."
