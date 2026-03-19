#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized
