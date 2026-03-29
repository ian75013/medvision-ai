#!/usr/bin/env bash
# Download all datasets (chest X-ray, brain MRI, brain segmentation, chest segmentation).
# Run from the project root with an activated virtual environment.
# Usage: bash scripts/download_dataset.sh [--force]
set -euo pipefail

FORCE=""
for arg in "$@"; do
    [[ "$arg" == "--force" ]] && FORCE="--force"
done

echo "==> [1/6] Chest X-ray classification dataset"
python -m src.data.download_dataset ${FORCE}

echo "==> [2/6] Brain MRI classification dataset"
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml ${FORCE}

echo "==> [3/6] Brain tumor segmentation dataset"
python -m src.data.download_segmentation_dataset --problem brain_tumor_seg ${FORCE}

echo "==> [4/6] Chest X-ray segmentation dataset"
if ! python -m src.data.download_segmentation_dataset --problem chest_xray_seg ${FORCE}; then
    echo "Chest X-ray segmentation download failed once, retrying with --force..."
    python -m src.data.download_segmentation_dataset --problem chest_xray_seg --force
fi

echo "==> [5/6] Prepare brain tumor segmentation manifest"
python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml

echo "==> [6/6] Prepare chest X-ray segmentation manifest"
python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml

echo ""
echo "All datasets available in data/raw/ and manifests in data/processed/."
