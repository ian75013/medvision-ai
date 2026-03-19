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
