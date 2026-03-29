#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm api bash -lc "python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml"
