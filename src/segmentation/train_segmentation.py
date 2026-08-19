

"""Entraînement des modèles de segmentation (et multitâche) sous TensorFlow/Keras.

C'est le point d'entrée appelé par le stage DVC ``train_brain_tumor_segmentation``. Il lit
une config YAML, construit les jeux de données, entraîne, évalue sur le test, écrit les
artefacts (modèle, métriques, historique, superposition d'exemple) et journalise le tout
dans MLflow.

Usage:
    python -m src.segmentation.train_segmentation --config configs/brain_tumor_segmentation.yaml
    python -m src.segmentation.train_segmentation --config <cfg> --epochs 3   # écrase la config

Hyperparamètres lus dans la config (valeurs par défaut entre parenthèses) :

* ``image_size`` (256) — côté du carré auquel les images sont redimensionnées.
* ``batch_size`` (8) — taille des lots.
* ``epochs`` (aucune) — si absent, calculé dynamiquement pour atteindre ~1000 pas
  d'optimisation, afin qu'un petit dataset ne s'entraîne pas en trois pas et qu'un gros ne
  tourne pas des heures.
* ``learning_rate`` (1e-3) — pas de l'optimiseur Adam.
* ``validation_split`` (0.2) — fraction réservée à la validation.
* ``seed`` (42) — graine de reproductibilité (Python, NumPy et TensorFlow).
* ``task_type`` (``multitask``) — ``multitask`` pour le U-Net à deux têtes, autre valeur
  pour la segmentation seule.
* ``segmentation_loss_weight`` (1.0) et ``classification_loss_weight`` (0.4) — le masque
  pèse plus que la classe : la segmentation est la tâche difficile, la classification n'est
  là que pour régulariser l'encodeur.

Pertes : entropie croisée binaire pour le masque ; binaire pour la classification à deux
classes, ``sparse_categorical_crossentropy`` au-delà. Rappels : ``EarlyStopping``
(patience 3, restauration des meilleurs poids) et ``ReduceLROnPlateau`` (facteur 0.2,
patience 2).

POURQUOI la structure ``try/finally`` autour de ``fit`` : un entraînement long peut
échouer ou être interrompu après plusieurs heures. Le bloc ``finally`` sauvegarde ce qui
peut l'être — modèle, historique — avant de relancer l'exception, pour ne pas perdre le
travail déjà fait. Et l'échec de la journalisation MLflow n'est jamais fatal : les
artefacts locaux, eux, sont écrits.
"""
from __future__ import annotations

import argparse
import json
import os
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
    """Coefficient de Dice en TensorFlow, utilisable comme métrique pendant ``fit``.

    Version graphe de :func:`src.segmentation.metrics.dice_coefficient_np`. Les deux
    doivent rester d'accord : celle-ci suit l'entraînement époque par époque, l'autre
    produit le chiffre publié dans le rapport final.

    Args:
        y_true: Masque de vérité terrain (tenseur de forme quelconque, aplati ici).
        y_pred: Masque prédit, **probabilités non seuillées** — c'est voulu : seuiller
            couperait le gradient et rendrait la métrique inutilisable comme perte.
        smooth: Terme de lissage, évite la division par zéro sur les coupes sans lésion.

    Returns:
        Tenseur scalaire : le Dice du lot.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth: float = 1e-6):
    """Intersection sur union en TensorFlow, suivie comme métrique pendant ``fit``.

    Args:
        y_true: Masque de vérité terrain.
        y_pred: Masque prédit, probabilités non seuillées.
        smooth: Terme de lissage.

    Returns:
        Tenseur scalaire : l'IoU du lot.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def set_seed(seed: int) -> None:
    """Fixe la graine des trois générateurs aléatoires en jeu.

    Python, NumPy et TensorFlow tirent chacun de leur côté : n'en fixer qu'un laisse
    l'entraînement irreproductible sans que rien ne le signale.

    Args:
        seed: Graine commune.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``config`` (chemin YAML, obligatoire) et ``epochs``
        (facultatif — fourni, il écrase la valeur de la config ; absent, le nombre d'époques
        est calculé dynamiquement dans :func:`main`).
    """
    parser = argparse.ArgumentParser(description='Train segmentation or multitask segmentation/classification model')
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=None)
    return parser.parse_args()


def _log_history_metrics(history: dict[str, list[float]]) -> None:
    """Reporte dans MLflow, pour chaque courbe d'entraînement, sa valeur finale et sa meilleure.

    « Meilleure » dépend du sens de la métrique : minimum pour tout ce qui contient
    ``loss``, maximum sinon. Sans cette distinction, la comparaison de deux runs classerait
    la pire perte en tête.

    Args:
        history: Historique Keras (``history.history``) : nom de métrique → valeurs par époque.
    """
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
    """Exécute l'entraînement de bout en bout, de la config aux artefacts.

    Déroulé :

    1. Lecture de la config, fixation des graines, construction des trois jeux de données.
    2. Choix du nombre d'époques — celui demandé, ou celui qui donne environ 1000 pas.
    3. Construction et compilation du modèle : U-Net multitâche (deux pertes pondérées) ou
       U-Net de segmentation seule.
    4. ``fit`` sous MLflow, entouré d'un ``try/finally`` qui sauvegarde le modèle et
       l'historique même si l'entraînement échoue, puis relance l'exception.
    5. Évaluation sur le jeu de test : Dice, IoU, exactitude pixel, précision/rappel/F1 du
       masque, et les mêmes pour la classification en multitâche.
    6. Écriture des artefacts — modèle ``.keras``, ``*_metrics.json``, ``*_history.json``,
       superposition d'exemple — puis journalisation MLflow, dont l'échec n'est pas fatal.

    Les métriques finales sont aussi affichées en JSON sur la sortie standard, pour être
    lues par DVC et par l'humain qui regarde le terminal.

    Raises:
        Exception: Toute erreur survenue pendant ``fit`` est relancée après la sauvegarde
            de secours, afin que DVC voie bien l'échec du stage.
    """
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    image_size = int(cfg.get('image_size', 256))
    batch_size = int(cfg.get('batch_size', 8))
    # Set epochs to None for dynamic logic below
    epochs = args.epochs if args.epochs is not None else cfg.get('epochs', None)
    learning_rate = float(cfg.get('learning_rate', 1e-3))
    validation_split = float(cfg.get('validation_split', 0.2))
    task_type = cfg.get('task_type', 'multitask')

    train_ds, val_ds, test_ds, class_names, train_size = build_segmentation_datasets(
        manifest_path=cfg['manifest_path'],
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        task_type=task_type,
    )

    # Dynamically adapt epochs if not set by user or config
    if epochs is None:
        target_steps = 1000  # You can adjust this target as needed
        steps_per_epoch = int(np.ceil(train_size / batch_size))
        epochs = int(np.ceil(target_steps / steps_per_epoch))
        print(f"[INFO] Dynamically setting epochs to {epochs} to reach at least {target_steps} steps (train_size={train_size}, batch_size={batch_size})")

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

    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', cfg.get('mlflow_tracking_uri', 'file:./mlruns'))
    mlflow_experiment = os.getenv('MLFLOW_EXPERIMENT_NAME', cfg.get('project_name', 'medvision-segmentation'))
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)


    # --- Training and evaluation logic ---
    history = None
    train_exception = None
    try:
        with mlflow.start_run(run_name=cfg.get('run_name', Path(args.config).stem)):
            mlflow.log_params({
                'image_size': image_size,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'validation_split': validation_split,
                'task_type': task_type,
                'class_names': ','.join(class_names),
                'seed': seed,
                'segmentation_loss_weight': float(cfg.get('segmentation_loss_weight', 1.0)),
                'classification_loss_weight': float(cfg.get('classification_loss_weight', 0.4)),
                'model_dir': str(model_dir),
                'reports_dir': str(reports_dir),
                'overlays_dir': str(overlays_dir),
            })
            steps_per_epoch = int(np.ceil(train_size / batch_size))
            val_steps = None
            if hasattr(val_ds, '__len__'):
                try:
                    val_steps = int(np.ceil(len(val_ds) / batch_size))
                except Exception:
                    val_steps = None
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
            )
    except Exception as train_exc:
        import traceback
        print("[FATAL] Training crashed with exception:")
        traceback.print_exc()
        train_exception = train_exc
    finally:
        # Always attempt to log artifacts/metrics to MLflow, even if training failed or was interrupted
        try:
            prefix = cfg.get('artifact_prefix', Path(args.config).stem)
            model_path = model_dir / f'{prefix}.keras'
            metrics_path = reports_dir / f'{prefix}_metrics.json'
            history_path = reports_dir / f'{prefix}_history.json'
            overlay_path = overlays_dir / f'{prefix}_sample_overlay.png'

            # Save model and metrics locally (if not already saved)
            if model and not model_path.exists():
                model.save(model_path)
            # Save metrics and history if available
            if history is not None:
                history_payload = {'epoch': list(range(1, len(history.history['loss']) + 1))}
                history_payload.update({k: [float(v) for v in vals] for k, vals in history.history.items()})
                history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')
            # Try MLflow artifact logging
            mlflow.log_artifact(str(model_path))
            if metrics_path.exists():
                mlflow.log_artifact(str(metrics_path))
            if history_path.exists():
                mlflow.log_artifact(str(history_path))
            if overlay_path.exists():
                mlflow.log_artifact(str(overlay_path))
        except Exception as mlflow_artifact_exc:
            print(f"[WARNING] MLflow artifact logging failed in finally block: {mlflow_artifact_exc}\nArtifacts saved locally.")
        if train_exception:
            raise train_exception

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

    # --- Always save model and metrics locally ---
    model.save(model_path)
    save_metrics(metrics, metrics_path)
    history_payload = {'epoch': list(range(1, len(history.history['loss']) + 1))}
    history_payload.update({k: [float(v) for v in vals] for k, vals in history.history.items()})
    history_path.write_text(json.dumps(history_payload, indent=2), encoding='utf-8')

    if sample_image is not None and sample_mask is not None:
        save_overlay(sample_image, sample_mask, overlay_path)

    # --- Try MLflow artifact logging, but do not fail if it errors ---
    try:
        with mlflow.start_run(run_name=cfg.get('run_name', Path(args.config).stem), nested=True):
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(history_path))
            if overlay_path.exists():
                mlflow.log_artifact(str(overlay_path))
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            if 'history' in locals():
                _log_history_metrics(history.history)
    except Exception as mlflow_artifact_exc:
        print(f"[WARNING] MLflow artifact logging failed: {mlflow_artifact_exc}\nArtifacts saved locally.")

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
