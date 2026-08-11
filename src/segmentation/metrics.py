
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def dice_coefficient_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    return float((2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth))


def iou_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true > 0.5) == (y_pred > 0.5)))


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
