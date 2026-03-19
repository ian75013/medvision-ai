from __future__ import annotations

import argparse
import json
import random

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
from src.models.baseline_model import build_baseline_model
from src.models.optimized_model import build_optimized_model
from src.utils.config import load_config
from src.utils.dataset_multiclass import build_multiclass_datasets
from src.utils.paths import ensure_dir


MODEL_FACTORY = {
    "baseline": build_baseline_model,
    "optimized": build_optimized_model,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brain MRI multi-class classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _gather_labels(ds: tf.data.Dataset) -> np.ndarray:
    labels = []
    for _, batch_y in ds.unbatch():
        labels.append(int(batch_y.numpy()))
    return np.array(labels)


def _build_model(model_name: str, image_size: int, learning_rate: float, num_classes: int) -> tf.keras.Model:
    if model_name == "baseline":
        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        x = tf.keras.layers.Rescaling(1.0)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=(image_size, image_size, 3),
            weights="imagenet",
        )
        base.trainable = False
        inputs = tf.keras.Input(shape=(image_size, image_size, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    epochs = int(args.epochs or cfg.get("epochs", 8))
    learning_rate = float(cfg.get("learning_rate", 3e-4))
    dataset_dir = cfg["dataset_dir"]
    model_dir = ensure_dir(cfg.get("model_dir", "artifacts/models"))
    reports_dir = ensure_dir(cfg.get("reports_dir", "artifacts/reports"))

    train_ds, val_ds, test_ds, class_names = build_multiclass_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=float(cfg.get("validation_split", 0.15)),
        seed=seed,
        training_subdir=cfg.get("training_subdir", "Training"),
        testing_subdir=cfg.get("testing_subdir", "Testing"),
    )

    num_classes = len(class_names)
    y_train = _gather_labels(train_ds)
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    class_weights = {int(i): float(v) for i, v in enumerate(class_weights_values)}

    model = _build_model(args.model, image_size, learning_rate, num_classes)

    mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(cfg.get("project_name", "medvision-brain-mri"))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
    ]

    with mlflow.start_run(run_name=f"brain-mri-{args.model}"):
        mlflow.log_params({
            "model_type": args.model,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "class_names": ",".join(class_names),
        })

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        y_true = []
        y_pred = []
        for batch_x, batch_y in test_ds:
            probs = model.predict(batch_x, verbose=0)
            preds = np.argmax(probs, axis=1)
            y_pred.extend(preds.tolist())
            y_true.extend(batch_y.numpy().astype(int).tolist())

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        metrics = evaluate_multiclass_predictions(y_true_arr, y_pred_arr, class_names)
        report = build_multiclass_report(y_true_arr, y_pred_arr, class_names)

        model_path = model_dir / f"brain_mri_{args.model}.keras"
        report_path = reports_dir / f"brain_mri_{args.model}_classification_report.txt"
        cm_path = reports_dir / f"brain_mri_{args.model}_confusion_matrix.png"
        metrics_path = reports_dir / ("brain_mri_metrics.json" if args.model == "optimized" else f"brain_mri_{args.model}_metrics.json")
        history_path = reports_dir / f"brain_mri_{args.model}_history.json"

        model.save(model_path)
        report_path.write_text(report, encoding="utf-8")
        save_confusion_matrix_multiclass(y_true_arr, y_pred_arr, cm_path, class_names)
        save_metrics(metrics, metrics_path)

        history_payload = {"epoch": list(range(1, len(history.history["loss"]) + 1))}
        history_payload.update(history.history)
        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(history_path))

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        print("Training complete")
        print(json.dumps(metrics, indent=2))
        print(report)


if __name__ == "__main__":
    main()
