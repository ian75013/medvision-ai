#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-medvision}"
API_LABEL="${API_LABEL:-app=medvision-api}"
SOURCE_DIR="${SOURCE_DIR:-artifacts/models}"
TARGET_DIR="${TARGET_DIR:-/app/artifacts/models}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

POD_NAME="$(kubectl -n "$NAMESPACE" get pod -l "$API_LABEL" -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "$POD_NAME" ]]; then
  echo "No medvision-api pod found in namespace $NAMESPACE" >&2
  exit 1
fi

echo "Target pod: $POD_NAME"
kubectl -n "$NAMESPACE" exec "$POD_NAME" -- sh -lc "mkdir -p '$TARGET_DIR'"

tar -C "$SOURCE_DIR" -cf - . | kubectl -n "$NAMESPACE" exec -i "$POD_NAME" -- tar -C "$TARGET_DIR" -xf -

echo "Synced local $SOURCE_DIR to $NAMESPACE/$POD_NAME:$TARGET_DIR"
