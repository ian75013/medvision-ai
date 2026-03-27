
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

from src.segmentation.data import build_segmentation_datasets
from src.segmentation.metrics import dice_coefficient_np, iou_np, pixel_accuracy_np, save_metrics
from src.segmentation.models.unet import build_multitask_unet, build_unet
from src.segmentation.overlays import save_overlay
from src.utils.config import load_config
from src.utils.paths import ensure_dir


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train segmentation or multitask segmentation/classification model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=None)
    return parser.parse_args()


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
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    image_size = int(cfg.get('image_size', 256))
    batch_size = int(cfg.get('batch_size', 8))
    epochs = int(args.epochs or cfg.get('epochs', 10))
    learning_rate = float(cfg.get('learning_rate', 1e-3))
    validation_split = float(cfg.get('validation_split', 0.2))
    task_type = cfg.get('task_type', 'multitask')

    train_ds, val_ds, test_ds, class_names = build_segmentation_datasets(
        manifest_path=cfg['manifest_path'],
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        task_type=task_type,
    )

    if task_type == 'multitask':
        model = build_multitask_unet(image_size=image_size, num_classes=max(2, len(class_names)))
        class_loss = 'binary_crossentropy' if len(class_names) <= 2 else 'sparse_categorical_crossentropy'
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={'segmentation_output': 'binary_crossentropy', 'classification_output': class_loss},
            loss_weights={'segmentation_output': float(cfg.get('segmentation_loss_weight', 1.0)), 'classification_output': float(cfg.get('classification_loss_weight', 0.4))},
            metrics={'segmentation_output': [dice_coefficient, iou_score], 'classification_output': ['accuracy']},
        )
    else:
        model = build_unet(image_size=image_size, num_mask_classes=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[dice_coefficient, iou_score],
        )

    model_dir = ensure_dir(cfg.get('model_dir', 'artifacts/models'))
    reports_dir = ensure_dir(cfg.get('reports_dir', 'artifacts/reports'))
    overlays_dir = ensure_dir(cfg.get('overlays_dir', 'artifacts/overlays'))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
    ]

    mlflow.set_tracking_uri(cfg.get('mlflow_tracking_uri', 'file:./mlruns'))
    mlflow.set_experiment(cfg.get('project_name', 'medvision-segmentation'))

    with mlflow.start_run(run_name=cfg.get('run_name', Path(args.config).stem)):
        mlflow.log_params({
            'image_size': image_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'task_type': task_type,
            'class_names': ','.join(class_names),
        })
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

        all_true_masks: list[np.ndarray] = []
        all_pred_masks: list[np.ndarray] = []
        all_class_true: list[int] = []
        all_class_pred: list[int] = []

        sample_image = None
        sample_mask = None

        for batch in test_ds:
            if task_type == 'multitask':
                batch_x, batch_y = batch
                true_masks = batch_y['segmentation_output'].numpy()
                class_true = batch_y['classification_output'].numpy()
                preds = model.predict(batch_x, verbose=0)
                pred_masks = preds['segmentation_output']
                class_preds_raw = preds['classification_output']
                if class_preds_raw.ndim == 1 or class_preds_raw.shape[-1] == 1:
                    class_preds = (class_preds_raw.reshape(-1) >= 0.5).astype(int)
                else:
                    class_preds = np.argmax(class_preds_raw, axis=1)
                all_class_true.extend(class_true.astype(int).tolist())
                all_class_pred.extend(class_preds.astype(int).tolist())
            else:
                batch_x, true_masks = batch
                pred_masks = model.predict(batch_x, verbose=0)

            pred_bin = (pred_masks >= 0.5).astype(np.float32)
            all_true_masks.append(true_masks.astype(np.float32))
            all_pred_masks.append(pred_bin.astype(np.float32))

            if sample_image is None:
                sample_image = batch_x[0].numpy()
                sample_mask = pred_bin[0, ..., 0]

        true_masks_arr = np.concatenate(all_true_masks, axis=0)
        pred_masks_arr = np.concatenate(all_pred_masks, axis=0)

        metrics = {
            'dice': dice_coefficient_np(true_masks_arr, pred_masks_arr),
            'iou': iou_np(true_masks_arr, pred_masks_arr),
            'pixel_accuracy': pixel_accuracy_np(true_masks_arr, pred_masks_arr),
        }
        true_mask_flat = (true_masks_arr > 0.5).reshape(-1).astype(int)
        pred_mask_flat = (pred_masks_arr > 0.5).reshape(-1).astype(int)
        metrics['mask_precision'] = float(precision_score(true_mask_flat, pred_mask_flat, zero_division=0))
        metrics['mask_recall'] = float(recall_score(true_mask_flat, pred_mask_flat, zero_division=0))
        metrics['mask_f1'] = float(f1_score(true_mask_flat, pred_mask_flat, zero_division=0))
        if all_class_true:
            class_true_arr = np.array(all_class_true)
            class_pred_arr = np.array(all_class_pred)
            metrics['classification_accuracy'] = float(np.mean(class_pred_arr == class_true_arr))
            avg_type = 'binary' if len(class_names) <= 2 else 'macro'
            metrics['classification_precision'] = float(
                precision_score(class_true_arr, class_pred_arr, average=avg_type, zero_division=0)
            )
            metrics['classification_recall'] = float(
                recall_score(class_true_arr, class_pred_arr, average=avg_type, zero_division=0)
            )
            metrics['classification_f1'] = float(
                f1_score(class_true_arr, class_pred_arr, average=avg_type, zero_division=0)
            )

        prefix = cfg.get('artifact_prefix', Path(args.config).stem)
        model_path = model_dir / f'{prefix}.keras'
        metrics_path = reports_dir / f'{prefix}_metrics.json'
        history_path = reports_dir / f'{prefix}_history.json'
        overlay_path = overlays_dir / f'{prefix}_sample_overlay.png'

        model.save(model_path)
        save_metrics(metrics, metrics_path)
        history_payload = {'epoch': list(range(1, len(history.history['loss']) + 1))}
        history_payload.update({k: [float(v) for v in vals] for k, vals in history.history.items()})
        history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')

        if sample_image is not None and sample_mask is not None:
            save_overlay(sample_image, sample_mask, overlay_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(history_path))
        if overlay_path.exists():
            mlflow.log_artifact(str(overlay_path))
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        _log_history_metrics(history.history)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
