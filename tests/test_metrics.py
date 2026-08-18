"""Tests du contrat des métriques de classification binaire.

Ces tests ne vérifient pas la *valeur* des métriques — scikit-learn s'en charge — mais leur
**contrat** : la liste des clés produites et la cohérence des effectifs. C'est ce contrat
que consomment les rapports d'artefacts et l'UI ; une clé disparue casserait un affichage
des mois après, sans que rien n'échoue au moment du changement.
"""

import numpy as np

from src.evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_returns_expected_keys() -> None:
    """Toutes les métriques attendues par les rapports sont présentes.

    Si une clé disparaît, les JSON d'artefacts et le tableau de comparaison de l'UI
    perdent silencieusement une colonne.
    """
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = evaluate_predictions(y_true, y_prob)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "balanced_accuracy",
        "specificity",
        "pr_auc",
        "tp",
        "tn",
        "fp",
        "fn",
    }

    assert expected_keys.issubset(metrics.keys())


def test_evaluate_predictions_confusion_counts_are_consistent() -> None:
    """Les quatre effectifs de la matrice de confusion totalisent le nombre d'exemples.

    Vérifie qu'aucun exemple n'est perdu ni compté deux fois — la panne typique d'un
    seuillage ou d'un aplatissement mal fait sur des tableaux de formes différentes.
    """
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = evaluate_predictions(y_true, y_prob)

    assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == len(y_true)
