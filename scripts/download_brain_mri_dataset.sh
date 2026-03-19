#!/usr/bin/env bash
set -euo pipefail
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
