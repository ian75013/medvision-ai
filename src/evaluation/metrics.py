"""Métriques de classification binaire — pensées pour un contexte médical.

POURQUOI ne pas se contenter de l'exactitude : sur le dataset de radiographies, environ
trois cas sur quatre sont des pneumonies. Un modèle qui répond toujours « pneumonie »
affiche 75 % d'exactitude et rate tous les cas sains. D'où la batterie retenue :

* **rappel** (sensibilité) — la part des malades effectivement détectés. C'est la métrique
  qui compte le plus : un faux négatif est un patient renvoyé chez lui ;
* **spécificité** — la part des sains correctement rassurés, calculée à la main depuis la
  matrice de confusion (scikit-learn ne l'expose pas directement) ;
* **exactitude équilibrée** — la moyenne des deux précédentes, insensible au déséquilibre ;
* **ROC AUC** et **PR AUC** — la qualité du classement indépendamment du seuil. Sur classes
  déséquilibrées, la PR AUC est la plus honnête des deux ;
* **tp / tn / fp / fn** — les effectifs bruts, conservés pour pouvoir recalculer n'importe
  quelle autre métrique des mois plus tard sans relancer le modèle.

Le seuil de décision est fixé à 0.5 partout dans ce module. Le changer ici sans le changer
côté service ferait diverger les chiffres du rapport et ceux observés en production.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calcule toutes les métriques binaires du projet à partir des probabilités prédites.

    Args:
        y_true: Étiquettes réelles, 0 (normal) ou 1 (pneumonie), tableau 1D.
        y_prob: Probabilités prédites de la classe positive, dans [0, 1], même longueur.
            Ce sont bien des **probabilités**, pas des prédictions binaires : les aires sous
            les courbes ont besoin du score continu.

    Returns:
        Les métriques décrites en tête de module, toutes converties en ``float`` natifs pour
        être sérialisables en JSON — un ``float32`` NumPy ne l'est pas.

    Note:
        ``roc_auc`` et ``pr_auc`` retombent à ``0.0`` quand elles ne sont pas définies,
        c'est-à-dire quand ``y_true`` ne contient qu'une seule classe. Cela arrive sur un
        jeu de test minuscule ou un dataset de démonstration ; on préfère un rapport
        complet avec un zéro visible à une exception qui perdrait toutes les autres
        métriques.

    Example:
        >>> metrics = evaluate_predictions(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8]))
        >>> metrics["recall"]
        1.0
    """
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp + 1e-12)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = 0.0
    return metrics


def build_classification_report(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """Produit le rapport texte de scikit-learn, classe par classe.

    Complète :func:`evaluate_predictions` : là où celle-ci donne des agrégats, ce rapport
    montre précision, rappel et effectif **par classe**, ce qui rend visible d'un coup d'œil
    le modèle qui n'a appris que la classe majoritaire.

    Args:
        y_true: Étiquettes réelles, 0 ou 1.
        y_prob: Probabilités prédites de la classe positive.

    Returns:
        Le rapport formaté, prêt à être écrit dans un ``.txt`` d'artefacts. Les noms de
        classes ``NORMAL`` / ``PNEUMONIA`` sont ceux du dataset de radiographies — ce
        module est spécifique à ce problème.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return str(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], zero_division=0))


def save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, output_path: str | Path) -> None:
    """Écrit la matrice de confusion en PNG, effectifs inscrits dans les cases.

    Les nombres sont écrits en clair par-dessus les cases : sans eux, une matrice où une
    classe domine apparaît comme un aplat de couleur d'où on ne lit rien.

    La figure est fermée après écriture — un entraînement qui en produit plusieurs ne doit
    pas accumuler les figures matplotlib en mémoire.

    Args:
        y_true: Étiquettes réelles, 0 ou 1.
        y_prob: Probabilités prédites de la classe positive.
        output_path: Chemin du PNG ; les dossiers parents sont créés.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
