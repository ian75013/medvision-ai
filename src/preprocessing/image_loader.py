from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path: str | Path, image_size: int = 224) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr
