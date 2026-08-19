"""Prédiction sur une coupe d'IRM cérébrale, avec un modèle Keras multi-classes.

Voie hors production — voir l'avertissement du paquet :mod:`src.inference`.

POURQUOI la config est relue à l'inférence : les noms de classes et leur **ordre** ne sont
pas dans le modèle. Une sortie de réseau est un vecteur d'indices ; c'est la config qui dit
que l'indice 2 est « pituitary ». Les lire ailleurs qu'à l'endroit d'où l'entraînement les a
tirés, c'est risquer d'annoncer une tumeur pour une autre.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.image_loader import load_and_preprocess_image
from src.utils.config import load_config


def load_brain_mri_model(model_path: str | Path) -> tf.keras.Model:
    """Charge le modèle Keras multi-classes d'IRM cérébrale.

    Args:
        model_path: Chemin du fichier ``.keras``.

    Returns:
        Le modèle prêt à prédire.
    """
    return tf.keras.models.load_model(model_path)


def predict_brain_mri(
    model: tf.keras.Model,
    image_path: str | Path,
    config_path: str | Path = "configs/brain_tumor_mri.yaml",
) -> dict[str, object]:
    """Classe une coupe d'IRM parmi les types de tumeurs déclarés dans la config.

    Args:
        model: Modèle chargé par :func:`load_brain_mri_model`.
        image_path: Chemin de l'image à classer.
        config_path: Config d'où sont lus ``class_names`` et ``image_size`` — **la même que
            celle de l'entraînement**, sans quoi les indices de sortie sont mal traduits.

    Returns:
        ``{"predicted_class": str, "confidence": float, "probabilities": {classe: proba}}``.
        Le détail par classe est conservé volontairement : sur un cas ambigu, savoir que
        deux classes sont à 0.45 et 0.44 vaut bien plus que le seul gagnant.

    Raises:
        KeyError: La config ne déclare pas ``class_names``.
        IndexError: Le modèle a plus de sorties que la config n'a de classes — signe qu'ils
            ne vont pas ensemble.
    """
    cfg = load_config(config_path)
    class_names: list[str] = cfg["class_names"]
    image_size = int(cfg.get("image_size", 224))

    image = load_and_preprocess_image(image_path, image_size=image_size)
    batch = np.expand_dims(image, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
    }
