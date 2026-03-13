import numpy as np

from src.evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_returns_expected_keys() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = evaluate_predictions(y_true, y_prob)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["accuracy"] == 1.0
