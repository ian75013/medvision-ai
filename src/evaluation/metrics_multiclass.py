"""Métriques de classification multi-classes — IRM cérébrales (types de tumeurs).

POURQUOI publier macro **et** pondéré : les deux répondent à des questions différentes et
l'une sans l'autre induit en erreur.

* **Macro** — chaque classe pèse autant, quelle que soit sa fréquence. C'est la mesure
  pertinente en médecine : rater systématiquement la classe la plus rare doit se voir.
* **Pondéré** — chaque classe pèse son effectif. C'est ce qu'observera un service en
  pratique, sur sa file de patients réelle.

Un écart marqué entre les deux est en soi le diagnostic : le modèle réussit sur les classes
fréquentes et échoue sur les rares.

À la différence du cas binaire, ces fonctions prennent des **prédictions déjà décidées**
(``argmax``) et non des probabilités : sans seuil à régler, il n'y a rien à calculer sur le
score continu.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_multiclass_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Calcule exactitude, précision, rappel et F1 en moyennes macro et pondérée.

    Args:
        y_true: Étiquettes réelles, entiers indexant ``class_names``.
        y_pred: Prédictions, déjà décidées par ``argmax``.
        class_names: Noms des classes, dans l'ordre de leurs indices. Seul leur **nombre**
            est utilisé ici ; il est conservé dans le rapport pour qu'un JSON de métriques
            relu des mois après dise sur combien de classes il porte.

    Returns:
        Les sept métriques et ``num_classes``, en ``float`` natifs sérialisables en JSON.
        Les classes jamais prédites comptent 0 plutôt que de lever (``zero_division=0``).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_classes": len(class_names),
    }


def build_multiclass_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    """Produit le rapport texte de scikit-learn, ligne par classe.

    Args:
        y_true: Étiquettes réelles.
        y_pred: Prédictions décidées.
        class_names: Noms des classes, **dans l'ordre de leurs indices** — un ordre faux
            n'échoue pas, il intervertit silencieusement les lignes du rapport.

    Returns:
        Le rapport formaté, prêt à être écrit dans un ``.txt`` d'artefacts.
    """
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    """Écrit les métriques en JSON indenté, en créant le dossier au besoin.

    C'est cette fonction qu'utilisent aussi les entraînements binaires : elle ne suppose
    rien du contenu du dictionnaire.

    Args:
        metrics: Métriques à sérialiser — valeurs de types JSON natifs.
        output_path: Chemin du JSON à écrire.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_confusion_matrix_multiclass(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path, class_names: list[str]) -> None:
    """Écrit la matrice de confusion multi-classes en PNG, effectifs inscrits dans les cases.

    C'est l'artefact le plus utile du rapport multi-classes : il montre **quelles paires**
    de classes le modèle confond, information qu'aucune moyenne ne restitue. Les étiquettes
    horizontales sont inclinées pour rester lisibles avec des noms de tumeurs longs.

    Args:
        y_true: Étiquettes réelles.
        y_pred: Prédictions décidées.
        output_path: Chemin du PNG ; les dossiers parents sont créés.
        class_names: Noms des classes, dans l'ordre de leurs indices.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm)
    ax.set_title("Brain MRI Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
