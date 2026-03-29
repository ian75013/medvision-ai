#!/usr/bin/env bash
# Download all datasets inside a Docker container.
# Data is written to data/raw/ via the volume mount (./:/app).
#
# The container needs your Kaggle credentials.
# Supported methods (in order of preference):
#   1. KAGGLE_USERNAME + KAGGLE_KEY env vars already set in your shell
#   2. ~/.kaggle/kaggle.json present on the host (auto-mounted)
#
# Usage:
#   bash scripts/download_dataset_docker.sh [--force]
set -euo pipefail

FORCE=""
for arg in "$@"; do
    [[ "$arg" == "--force" ]] && FORCE="--force"
done

# --- Kaggle credentials ---
KAGGLE_MOUNT_FLAG=""
KAGGLE_SETUP_CMD=":"
if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
    if [[ ! -f "$KAGGLE_JSON" ]]; then
        echo "ERROR: Kaggle credentials not found."
        echo "  Either set KAGGLE_USERNAME + KAGGLE_KEY env vars,"
        echo "  or create ~/.kaggle/kaggle.json."
        exit 1
    fi
    KAGGLE_MOUNT_FLAG="-v ${KAGGLE_JSON}:/tmp/kaggle.json:ro"
    KAGGLE_SETUP_CMD="mkdir -p /root/.kaggle && cp /tmp/kaggle.json /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json"
fi

# Isolate each job in its own container to reduce peak memory and avoid OOM kills.
run_download_step() {
    local label="$1"
    local cmd="$2"
    echo "==> ${label}"
    API_DOMAIN="${API_DOMAIN:-api.example.com}" APP_DOMAIN="${APP_DOMAIN:-app.example.com}" \
        docker compose run --rm --no-deps \
        ${KAGGLE_MOUNT_FLAG} \
        api bash -lc "set -e; ${KAGGLE_SETUP_CMD}; ${cmd}"
}

run_download_step "[1/6] Chest X-ray classification dataset" \
    "python -m src.data.download_dataset ${FORCE}"

run_download_step "[2/6] Brain MRI classification dataset" \
    "python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml ${FORCE}"

run_download_step "[3/6] Brain tumor segmentation dataset" \
    "python -m src.data.download_segmentation_dataset --problem brain_tumor_seg ${FORCE}"

run_download_step "[4/6] Chest X-ray segmentation dataset" \
    "python -m src.data.download_segmentation_dataset --problem chest_xray_seg ${FORCE} || (echo 'Chest X-ray segmentation download failed once, retrying with --force...' && python -m src.data.download_segmentation_dataset --problem chest_xray_seg --force)"

run_download_step "[5/6] Prepare brain tumor segmentation manifest" \
    "python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml"

run_download_step "[6/6] Prepare chest X-ray segmentation manifest" \
    "python -m src.data.prepare_segmentation_dataset --config configs/chest_xray_segmentation.yaml"

echo ""
echo "All datasets available in data/raw/ and manifests in data/processed/."
