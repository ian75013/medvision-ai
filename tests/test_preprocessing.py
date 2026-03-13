from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.image_loader import load_and_preprocess_image


def test_load_and_preprocess_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    image.save(image_path)

    arr = load_and_preprocess_image(image_path, image_size=64)

    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.float32
