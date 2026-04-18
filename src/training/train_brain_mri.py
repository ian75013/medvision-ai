from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics_multiclass import (
    build_multiclass_report,
    evaluate_multiclass_predictions,
    save_confusion_matrix_multiclass,
    save_metrics,
)
from src.models.backbones import TF_BACKBONES
from src.models.baseline_model import build_baseline_model
from src.training.transfer_utils import FineTuneConfig, infer_unfreeze_layers, train_with_progressive_finetuning
from src.utils.config import load_config
from src.utils.dataset_multiclass import build_multiclass_datasets
from src.utils.paths import ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brain MRI multi-class classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default="densenet121")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _gather_labels(ds: tf.data.Dataset) -> np.ndarray:
    labels = []
    for _, batch_y in ds.unbatch():
        labels.append(int(batch_y.numpy()))
    return np.array(labels)


def _log_history_metrics(history: dict[str, list[float]]) -> None:
    for metric_name, values in history.items():
        if not values:
            continue
        series = [float(v) for v in values]
        mlflow.log_metric(f"final_{metric_name}", series[-1])
        if "loss" in metric_name:
            mlflow.log_metric(f"best_{metric_name}", float(min(series)))
        else:
            mlflow.log_metric(f"best_{metric_name}", float(max(series)))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    epochs_total = int(args.epochs or cfg.get("epochs", 20))
    warmup_epochs = int(cfg.get("warmup_epochs", max(4, min(8, epochs_total // 3))))
    finetune_epochs = max(0, epochs_total - warmup_epochs)
    warmup_lr = float(cfg.get("warmup_learning_rate", cfg.get("learning_rate", 3e-4)))
    finetune_lr = float(cfg.get("finetune_learning_rate", min(warmup_lr, 3e-5)))
    dataset_dir = cfg["dataset_dir"]
    model_dir = ensure_dir(cfg.get("model_dir", "artifacts/models"))
    reports_dir = ensure_dir(cfg.get("reports_dir", "artifacts/reports"))
    label_smoothing = float(cfg.get("label_smoothing", 0.05))
    backbone_name = args.model.lower()

    train_ds, val_ds, test_ds, class_names = build_multiclass_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=float(cfg.get("validation_split", 0.15)),
        seed=seed,
        training_subdir=cfg.get("training_subdir", "Training"),
        testing_subdir=cfg.get("testing_subdir", "Testing"),
    )

    y_train = _gather_labels(train_ds)
    class_ids = np.arange(len(class_names))
    class_weights_values = compute_class_weight(class_weight="balanced", classes=class_ids, y=y_train)
    class_weights = {int(class_ids[i]): float(class_weights_values[i]) for i in range(len(class_ids))}

    if backbone_name == "baseline":
        model = build_baseline_model(image_size=image_size, num_classes=len(class_names), learning_rate=warmup_lr)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_total,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
            ],
            verbose=1,
        ).history
    else:
        if backbone_name not in TF_BACKBONES:
            raise ValueError(f"Unknown model `{backbone_name}`. Choices: baseline, {', '.join(TF_BACKBONES)}")
        model, history = train_with_progressive_finetuning(
            train_ds=train_ds,
            val_ds=val_ds,
            image_size=image_size,
            num_classes=len(class_names),
            cfg=FineTuneConfig(
                warmup_epochs=warmup_epochs,
                finetune_epochs=finetune_epochs,
                warmup_lr=warmup_lr,
                finetune_lr=finetune_lr,
                backbone_name=backbone_name,
                unfreeze_layers=infer_unfreeze_layers(backbone_name, cfg.get("unfreeze_layers")),
                dropout=cfg.get("dropout"),
                label_smoothing=label_smoothing,
            ),
            class_weight=class_weights,
        )

    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().astype(int).tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics = evaluate_multiclass_predictions(y_true_arr, y_pred_arr, class_names)
    report = build_multiclass_report(y_true_arr, y_pred_arr, class_names)

    model_filename = "brain_mri_baseline.keras" if backbone_name == "baseline" else f"brain_mri_{backbone_name}.keras"
    metrics_filename = "brain_mri_baseline_metrics.json" if backbone_name == "baseline" else f"brain_mri_{backbone_name}_metrics.json"
    report_filename = "brain_mri_baseline_classification_report.txt" if backbone_name == "baseline" else f"brain_mri_{backbone_name}_classification_report.txt"
    cm_filename = "brain_mri_baseline_confusion_matrix.png" if backbone_name == "baseline" else f"brain_mri_{backbone_name}_confusion_matrix.png"
    history_filename = "brain_mri_baseline_history.json" if backbone_name == "baseline" else f"brain_mri_{backbone_name}_history.json"

    model_path = Path(model_dir) / model_filename
    metrics_path = Path(reports_dir) / metrics_filename
    report_path = Path(reports_dir) / report_filename
    cm_path = Path(reports_dir) / cm_filename
    history_path = Path(reports_dir) / history_filename

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.get("mlflow_tracking_uri", "file:./mlruns"))
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", cfg.get("project_name", "medvision-brain-mri"))
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"brain-mri-{backbone_name}"):
        mlflow.log_params(
            {
                "problem": "brain_mri",
                "model": backbone_name,
                "image_size": image_size,
                "batch_size": batch_size,
                "epochs_total": epochs_total,
                "warmup_epochs": warmup_epochs,
                "finetune_epochs": finetune_epochs,
                "warmup_lr": warmup_lr,
                "finetune_lr": finetune_lr,
                "num_classes": len(class_names),
                "label_smoothing": label_smoothing,
            }
        )
        for class_id, weight in class_weights.items():
            mlflow.log_param(f"class_weight_{class_id}", weight)
        _log_history_metrics(history)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        model.save(model_path)
        save_metrics(metrics, metrics_path)
        report_path.write_text(report, encoding="utf-8")
        save_confusion_matrix_multiclass(y_true_arr, y_pred_arr, cm_path, class_names)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(history_path))

    print(json.dumps({"metrics": metrics, "class_names": class_names, "model_path": str(model_path)}, indent=2))


if __name__ == "__main__":
    main()
