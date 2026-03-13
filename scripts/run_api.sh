#!/usr/bin/env bash
set -euo pipefail
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
