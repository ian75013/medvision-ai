#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm api bash -lc "python -m src.training.train --config configs/config.yaml --model optimized"
