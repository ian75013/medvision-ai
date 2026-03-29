#!/usr/bin/env bash
# Download all datasets (chest X-ray, brain MRI, brain segmentation, chest segmentation).
# Run from the project root with an activated virtual environment.
# Usage: bash scripts/download_dataset.sh [--force]
set -euo pipefail

FORCE=""
for arg in "$@"; do
    [[ "$arg" == "--force" ]] && FORCE="--force"
done

echo "==> [1/4] Chest X-ray classification dataset"
python -m src.data.download_dataset ${FORCE}

echo "==> [2/4] Brain MRI classification dataset"
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml ${FORCE}

echo "==> [3/4] Brain tumor segmentation dataset"
python -m src.data.download_segmentation_dataset --problem brain_tumor_seg ${FORCE}

echo "==> [4/4] Chest X-ray segmentation dataset"
python -m src.data.download_segmentation_dataset --problem chest_xray_seg ${FORCE}

echo ""
echo "All datasets available in data/raw/."
