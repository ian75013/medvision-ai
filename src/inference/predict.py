"""Prédiction sur une radiographie thoracique, avec un modèle Keras (voie hors production).

Voir l'avertissement du paquet :mod:`src.inference` : la production passe par ONNX.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.image_loader import load_and_preprocess_image


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Charge un modèle Keras depuis le disque.

    Contrairement à l'inférence de segmentation, le modèle est chargé **compilé** : les
    classifieurs n'utilisent que des métriques standard, que Keras sait reconstruire seul.

    Args:
        model_path: Chemin du fichier ``.keras``.

    Returns:
        Le modèle prêt à prédire.
    """
    return tf.keras.models.load_model(model_path)


def predict_from_path(model: tf.keras.Model, image_path: str | Path, image_size: int = 224) -> dict[str, float | str]:
    """Prédit « normal » ou « pneumonie » pour une image, et renvoie la probabilité.

    Le modèle est binaire à sortie unique : la valeur lue est la probabilité de la classe
    positive (pneumonie), et le seuil de décision est 0.5 — le même que celui des métriques
    d'évaluation.

    Args:
        model: Modèle chargé par :func:`load_model`.
        image_path: Chemin de l'image à classer.
        image_size: Côté d'entrée du modèle. **Doit valoir la taille vue à l'entraînement.**

    Returns:
        ``{"predicted_class": "PNEUMONIA" | "NORMAL", "probability_pneumonia": float}``.
        La probabilité est toujours celle de la pneumonie, y compris quand la classe prédite
        est « normal » — c'est ce qui permet à l'appelant d'appliquer son propre seuil.
    """
    image = load_and_preprocess_image(image_path, image_size=image_size)
    batch = np.expand_dims(image, axis=0)
    probability = float(model.predict(batch, verbose=0)[0][0])
    predicted_class = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
    return {"predicted_class": predicted_class, "probability_pneumonia": probability}
