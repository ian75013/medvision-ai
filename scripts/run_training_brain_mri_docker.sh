#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm api bash -lc "python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized"
