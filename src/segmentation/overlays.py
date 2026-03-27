
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: str | Path, alpha: float = 0.35) -> None:
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
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8)
