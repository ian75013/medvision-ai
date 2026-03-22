from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf

from src.utils.config import load_config

DEFAULT_ARTIFACTS_DIR = Path("artifacts")

PROBLEMS: Dict[str, Dict[str, Any]] = {
    "chest_xray": {
        "label": "Chest X-ray Pneumonia Classification",
        "config_path": "configs/config.yaml",
        "model_candidates": {
            "baseline": "baseline_model.keras",
            "optimized": "optimized_model.keras",
        },
        "report_candidates": {
            "baseline": "baseline_classification_report.txt",
            "optimized": "optimized_classification_report.txt",
        },
        "metrics_candidates": {
            "baseline": ["baseline_metrics.json"],
            "optimized": ["optimized_metrics.json"],
        },
        "class_names": ["NORMAL", "PNEUMONIA"],
        "task_type": "binary",
    },
    "brain_mri": {
        "label": "Brain MRI Tumor Classification",
        "config_path": "configs/brain_tumor_mri.yaml",
        "model_candidates": {
            "baseline": "brain_mri_baseline.keras",
            "optimized": "brain_mri_optimized.keras",
        },
        "report_candidates": {
            "baseline": "brain_mri_baseline_classification_report.txt",
            "optimized": "brain_mri_optimized_classification_report.txt",
        },
        "metrics_candidates": {
            "baseline": ["brain_mri_baseline_metrics.json"],
            "optimized": ["brain_mri_metrics.json", "brain_mri_optimized_metrics.json"],
        },
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "task_type": "multiclass",
    },
    "brain_tumor_segmentation": {
        "label": "Brain Tumor Segmentation + Classification",
        "config_path": "configs/brain_tumor_segmentation.yaml",
        "model_candidates": {
            "unet_multitask": "brain_tumor_segmentation_unet.keras",
        },
        "metrics_candidates": {
            "unet_multitask": ["brain_tumor_segmentation_unet_metrics.json"],
        },
        "class_names": ["glioma", "meningioma", "pituitary", "notumor"],
        "task_type": "segmentation_multitask",
    },
    "chest_xray_segmentation": {
        "label": "Chest X-ray Lung Segmentation + Pneumonia Classification",
        "config_path": "configs/chest_xray_segmentation.yaml",
        "model_candidates": {
            "unet_multitask": "chest_xray_segmentation_unet.keras",
        },
        "metrics_candidates": {
            "unet_multitask": ["chest_xray_segmentation_unet_metrics.json"],
        },
        "class_names": ["NORMAL", "PNEUMONIA"],
        "task_type": "segmentation_multitask",
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_first_existing(directory: Path, names: List[str] | tuple[str, ...]) -> Path | None:
    for name in names:
        if not name:
            continue
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def load_registry(artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> Dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    models_dir = artifacts_dir / "models"
    reports_dir = artifacts_dir / "reports"
    registry: Dict[str, Any] = {"problems": {}}

    for problem_key, spec in PROBLEMS.items():
        config = load_config(spec["config_path"]) if Path(spec["config_path"]).exists() else {}
        problem_entry: Dict[str, Any] = {
            "label": spec["label"],
            "task_type": spec["task_type"],
            "class_names": config.get("class_names", spec["class_names"]),
            "models": {},
        }
        for model_key, model_filename in spec["model_candidates"].items():
            model_path = models_dir / model_filename
            metrics_path = _find_first_existing(reports_dir, spec.get("metrics_candidates", {}).get(model_key, []))
            report_path = _find_first_existing(reports_dir, [spec.get("report_candidates", {}).get(model_key, "")])

            problem_entry["models"][model_key] = {
                "model_path": str(model_path),
                "available": model_path.exists(),
                "metrics": _load_json(metrics_path) if metrics_path else {},
                "metrics_path": str(metrics_path) if metrics_path else None,
                "report_path": str(report_path) if report_path and report_path.exists() else None,
                "config_path": spec["config_path"],
            }
        registry["problems"][problem_key] = problem_entry

    return registry


@lru_cache(maxsize=16)
def load_tf_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


def get_model_entry(problem: str, model_name: str, artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> Dict[str, Any]:
    registry = load_registry(artifacts_dir)
    problem_entry = registry["problems"].get(problem)
    if not problem_entry:
        raise KeyError(f"Unknown problem: {problem}")
    model_entry = problem_entry["models"].get(model_name)
    if not model_entry:
        raise KeyError(f"Unknown model '{model_name}' for problem '{problem}'")
    return {**model_entry, "class_names": problem_entry["class_names"], "task_type": problem_entry["task_type"]}


def compare_models(problem: str, artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> List[Dict[str, Any]]:
    registry = load_registry(artifacts_dir)
    problem_entry = registry["problems"].get(problem)
    if not problem_entry:
        raise KeyError(f"Unknown problem: {problem}")

    rows: List[Dict[str, Any]] = []
    for model_name, model_entry in problem_entry["models"].items():
        row = {
            "model_name": model_name,
            "available": model_entry["available"],
        }
        row.update(model_entry.get("metrics", {}))
        rows.append(row)
    return rows
