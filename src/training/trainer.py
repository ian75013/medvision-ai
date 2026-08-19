"""Boucle d'époque PyTorch, partagée par l'entraînement et l'évaluation.

POURQUOI une seule fonction pour les deux : un ``run_epoch`` d'entraînement et un
``evaluate`` écrits séparément finissent toujours par diverger sur un détail — la façon de
moyenner la perte, le traitement du dernier lot incomplet — et les chiffres cessent d'être
comparables. Ici, la présence ou l'absence d'un optimiseur suffit à basculer d'un mode à
l'autre, tout le reste du calcul est littéralement le même code.

À l'inverse de la voie TensorFlow, où Keras fournit ``fit`` et ``evaluate``, PyTorch laisse
cette boucle à l'appelant : c'est le prix de sa souplesse.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EpochMetrics:
    """Résultat d'une époque, moyenné sur les exemples et non sur les lots.

    Attributes:
        loss: Perte moyenne par exemple. Moyenner par exemple plutôt que par lot évite
            qu'un dernier lot incomplet ne pèse autant qu'un lot plein.
        accuracy: Proportion d'exemples correctement classés, entre 0.0 et 1.0.
    """

    loss: float
    accuracy: float


def _move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfère un lot du chargeur vers le périphérique de calcul.

    Args:
        batch: Lot produit par le ``DataLoader``, avec les clés ``image`` et ``label``.
        device: Périphérique cible (``cuda`` ou ``cpu``).

    Returns:
        Le couple ``(images, labels)`` déjà placé sur le périphérique.
    """
    x = batch["image"].to(device)
    y = batch["label"].to(device)
    return x, y


def run_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """Parcourt le chargeur une fois, en entraînement ou en évaluation.

    Le mode est déterminé par ``optimizer`` : fourni, on rétropropage et on met à jour les
    poids ; absent, on ne fait qu'un passage avant. ``model.train(is_train)`` bascule aussi
    dropout et batch-norm dans le bon régime — l'oublier fausse silencieusement toute
    évaluation.

    Args:
        model: Modèle à entraîner ou à évaluer.
        dataloader: Itérable de lots, chacun un dictionnaire ``{"image": ..., "label": ...}``.
        criterion: Fonction de perte, appliquée aux logits bruts.
        device: Périphérique de calcul.
        optimizer: Optimiseur. **Sa présence seule déclenche l'entraînement** ; passer
            ``None`` fait de cet appel une évaluation.

    Returns:
        Les métriques de l'époque — voir :class:`EpochMetrics`.

    Note:
        Le dénominateur est protégé par ``max(1, ...)`` : un chargeur vide renvoie des
        métriques nulles au lieu de lever une division par zéro, ce qui laisse la boucle
        appelante signaler proprement le problème.

    Example:
        >>> train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        >>> val_metrics = run_epoch(model, val_loader, criterion, device)  # sans optimiseur
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        x, y = _move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((preds == y).sum().item())
        total_examples += int(x.size(0))

    return EpochMetrics(
        loss=total_loss / max(1, total_examples),
        accuracy=total_correct / max(1, total_examples),
    )
