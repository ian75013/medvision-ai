"""Prédiction sur un **volume** IRM (NIfTI), avec le petit CNN PyTorch de démonstration.

POURQUOI l'agrégation par moyenne : le modèle voit des coupes 2D, mais la question posée
porte sur le patient. On prédit donc chaque coupe centrale, puis on moyenne les
probabilités. Moyenner les probabilités plutôt que voter à la majorité conserve le degré de
certitude : trois coupes hésitantes à 0.51 ne valent pas trois coupes affirmatives à 0.99,
et un vote les confondrait.

Voie hors production — voir l'avertissement du paquet :mod:`src.inference`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.dataio.nifti_loader import load_volume
from src.models.classification_2d.simple_cnn import SimpleCNN2D
from src.preprocessing.brain_mri_2d import BrainMRI2DPreprocessor


def predict_volume(
    model_path: str | Path,
    volume_path: str | Path,
    image_size: int = 128,
    k: int = 5,
) -> dict[str, float | str]:
    """Charge le modèle, pré-traite le volume, prédit chaque coupe et agrège le résultat.

    Le calcul est forcé sur CPU : cette fonction sert à vérifier un modèle depuis n'importe
    quelle machine, y compris sans GPU, et le coût est négligeable sur cinq coupes.

    Les coupes forment un seul lot, passé en une fois au modèle. ``torch.no_grad()`` évite
    de construire un graphe de gradients dont on n'a que faire en inférence.

    Args:
        model_path: Chemin des poids ``.pt`` (un ``state_dict`` de
            :class:`src.models.classification_2d.simple_cnn.SimpleCNN2D`).
        volume_path: Chemin du volume NIfTI.
        image_size: Côté d'entrée du modèle — **doit valoir celui de l'entraînement**.
        k: Nombre de coupes centrales à agréger.

    Returns:
        ``{"predicted_class": "tumor" | "normal", "probability_normal": float,
        "probability_tumor": float}``, les probabilités étant moyennées sur les coupes.

    Raises:
        FileNotFoundError: Le modèle ou le volume est introuvable.
        RuntimeError: Les poids ne correspondent pas à l'architecture — typiquement un
            ``image_size`` différent de celui de l'entraînement.
    """
    device = torch.device("cpu")
    model = SimpleCNN2D(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    volume = load_volume(volume_path)
    preprocessor = BrainMRI2DPreprocessor(image_size=image_size)
    slices = preprocessor.preprocess_volume(volume, strategy="central_k", k=k)
    batch = torch.stack([torch.from_numpy(s).float() for s in slices], dim=0)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        mean_probs = probs.mean(dim=0)

    predicted_idx = int(mean_probs.argmax().item())
    return {
        "predicted_class": "tumor" if predicted_idx == 1 else "normal",
        "probability_normal": float(mean_probs[0].item()),
        "probability_tumor": float(mean_probs[1].item()),
    }
