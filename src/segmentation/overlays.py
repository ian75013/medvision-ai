"""Rendu visuel des masques de segmentation : superposition et export PNG.

POURQUOI ce module existe : un score de Dice ne dit pas *où* le modèle s'est trompé. La
superposition du masque prédit sur l'image d'origine est la seule sortie qu'un radiologue
— ou Yann trois mois plus tard — peut juger d'un coup d'œil. Ces images sont produites à
la fin de l'entraînement (``train_segmentation``), à l'inférence
(``predict_segmentation``) et affichées par l'UI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: str | Path, alpha: float = 0.35) -> None:
    """Écrit un PNG où le masque est posé en rouge translucide sur l'image.

    Les dossiers parents manquants sont créés. La figure matplotlib est fermée
    explicitement : sans cela, un entraînement qui produit une superposition par époque
    accumule les figures en mémoire jusqu'à saturer le processus.

    Args:
        image: Image d'origine, tableau ``(H, W, 3)`` de flottants dans [0, 1].
        mask: Masque à superposer, ``(H, W)`` — binaire ou probabilités dans [0, 1].
        output_path: Chemin du PNG à écrire.
        alpha: Opacité du masque, de 0 (invisible) à 1 (masque opaque). 0.35 laisse
            l'anatomie sous-jacente lisible tout en marquant nettement la lésion.

    Example:
        >>> save_overlay(image, pred_mask, "artifacts/overlays/case_042.png")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.imshow(mask, alpha=alpha, cmap="Reds")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Convertit un masque flottant en image PIL en niveaux de gris.

    Les valeurs sont bornées à [0, 1] avant mise à l'échelle : un modèle peut sortir de
    l'intervalle (logits mal calibrés, artefacts numériques), et sans le bornage la
    conversion en ``uint8`` reboucle — un pixel à 1.01 deviendrait noir au lieu de blanc.

    Args:
        mask: Masque ``(H, W)`` de flottants, typiquement dans [0, 1].

    Returns:
        Image PIL en mode « L », où 0 = fond et 255 = lésion.
    """
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8)
