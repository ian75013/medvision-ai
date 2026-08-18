"""Métriques de segmentation en NumPy pur, et sauvegarde du rapport JSON.

POURQUOI des versions NumPy alors que ``train_segmentation`` définit déjà les mêmes
métriques en TensorFlow : les versions TF servent *pendant* l'entraînement (elles doivent
être différentiables et vivre dans le graphe) ; celles-ci servent *après*, sur les
prédictions déjà matérialisées du jeu de test. Les garder en NumPy permet de les appeler
— et de les tester — sans importer TensorFlow, ce dont dépendent les smoke tests de la CI.

POURQUOI Dice et IoU plutôt que l'exactitude : sur une IRM, la lésion occupe souvent moins
de 2 % des pixels. Un modèle qui prédit « fond » partout affiche 98 % d'exactitude et un
Dice de 0. Seules les métriques de recouvrement disent quelque chose d'utile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def dice_coefficient_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Coefficient de Dice : deux fois le recouvrement, rapporté à la somme des surfaces.

    Vaut 1 quand les deux masques coïncident, 0 quand ils sont disjoints. C'est la métrique
    de référence en segmentation médicale.

    Args:
        y_true: Masque de vérité terrain, valeurs 0/1 (toute forme, aplatie implicitement).
        y_pred: Masque prédit **déjà binarisé**, même forme que ``y_true``.
        smooth: Terme de lissage ajouté au numérateur et au dénominateur. Il évite la
            division par zéro quand les deux masques sont vides — cas fréquent sur les
            coupes saines, où le score doit alors valoir 1 et non ``NaN``.

    Returns:
        Le Dice, entre 0.0 et 1.0.

    Example:
        >>> dice_coefficient_np(np.ones((4, 4)), np.ones((4, 4)))
        1.0
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    return float((2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth))


def iou_np(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    """Intersection sur union (indice de Jaccard) entre deux masques binaires.

    Toujours inférieur ou égal au Dice, et plus sévère sur les petites erreurs de contour.
    On publie les deux : le Dice pour comparer à la littérature médicale, l'IoU pour
    comparer aux travaux de vision par ordinateur.

    Args:
        y_true: Masque de vérité terrain, valeurs 0/1.
        y_pred: Masque prédit **déjà binarisé**, même forme que ``y_true``.
        smooth: Terme de lissage, même rôle que dans :func:`dice_coefficient_np`.

    Returns:
        L'IoU, entre 0.0 et 1.0.
    """
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def pixel_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Proportion de pixels classés du bon côté du seuil 0.5.

    À lire toujours **en regard du Dice** : sur des lésions petites, cette valeur reste
    proche de 1 même pour un modèle inutile (voir la note du module). Elle n'est conservée
    que pour repérer les régressions grossières.

    Args:
        y_true: Masque de vérité terrain, binaire ou continu.
        y_pred: Masque prédit, binaire ou continu — seuillé à 0.5 comme ``y_true``.

    Returns:
        L'exactitude pixel à pixel, entre 0.0 et 1.0.
    """
    return float(np.mean((y_true > 0.5) == (y_pred > 0.5)))


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Écrit le dictionnaire de métriques en JSON indenté, encodé en UTF-8.

    Ce fichier est l'artefact que DVC suit et que MLflow archive : c'est lui qui permet de
    comparer deux entraînements des mois plus tard.

    Args:
        metrics: Métriques à sérialiser — les valeurs doivent être des types JSON natifs
            (les ``float`` NumPy doivent avoir été convertis par l'appelant).
        path: Chemin du fichier JSON à écrire. Le dossier parent doit exister.
    """
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
