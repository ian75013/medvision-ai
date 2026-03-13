from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics import build_classification_report, evaluate_predictions, save_confusion_matrix
from src.models.baseline_model import build_baseline_model
from src.models.optimized_model import build_optimized_model
from src.utils.config import load_config
from src.utils.dataset import build_datasets
from src.utils.paths import ensure_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


MODEL_FACTORY = {
    "baseline": build_baseline_model,
    "optimized": build_optimized_model,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a medical image classification model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model", type=str, choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    image_size = int(config.get("image_size", 224))
    batch_size = int(config.get("batch_size", 16))
    epochs = args.epochs or int(config.get("epochs", 3))
    learning_rate = float(config.get("learning_rate", 5e-4))
    dataset_dir = config["dataset_dir"]
    model_dir = ensure_dir(config.get("model_dir", "artifacts/models"))
    reports_dir = ensure_dir(config.get("reports_dir", "artifacts/reports"))

    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(config.get("project_name", "medvision-ai"))

    train_ds, val_ds, test_ds = build_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=float(config.get("validation_split", 0.2)),
        seed=seed,
    )

    if args.model == "baseline":
        model = MODEL_FACTORY[args.model](image_size=image_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        model = MODEL_FACTORY[args.model](image_size=image_size, learning_rate=learning_rate)

    # gather labels for class weights
    y_train_all = []
    for _, labels in train_ds.unbatch():
        y_train_all.append(int(labels.numpy()[0] if hasattr(labels.numpy(), "__len__") else labels.numpy()))
    class_weights_values = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=np.array(y_train_all))
    class_weights = {0: float(class_weights_values[0]), 1: float(class_weights_values[1])}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
    ]

    with mlflow.start_run(run_name=f"{args.model}-training"):
        mlflow.log_params(
            {
                "model_type": args.model,
                "image_size": image_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": seed,
            }
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        y_true = []
        y_prob = []
        for batch_x, batch_y in test_ds:
            probs = model.predict(batch_x, verbose=0).ravel()
            y_prob.extend(probs.tolist())
            y_true.extend(batch_y.numpy().ravel().astype(int).tolist())

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)
        metrics = evaluate_predictions(y_true_arr, y_prob_arr)
        report = build_classification_report(y_true_arr, y_prob_arr)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        model_path = model_dir / f"{args.model}_model.keras"
        model.save(model_path)
        mlflow.log_artifact(str(model_path))

        report_path = reports_dir / f"{args.model}_classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        cm_path = reports_dir / f"{args.model}_confusion_matrix.png"
        save_confusion_matrix(y_true_arr, y_prob_arr, cm_path)
        mlflow.log_artifact(str(cm_path))

        history_path = reports_dir / f"{args.model}_history.json"
        history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(history_path))

        print("Training complete")
        print(f"Saved model to: {model_path}")
        print("Metrics:", json.dumps(metrics, indent=2))
        print("Classification report:\n", report)


if __name__ == "__main__":
    main()
