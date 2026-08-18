"""Test du pré-traitement d'image partagé par l'inférence et l'entraînement.

Le contrat vérifié ici — forme carrée à la taille demandée, trois canaux, ``float32`` — est
celui sur lequel comptent tous les modèles du projet. Le rompre ne lève pas d'erreur : le
modèle prédit simplement moins bien, sans que rien ne le signale (voir la note du paquet
:mod:`src.preprocessing`).
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing.image_loader import load_and_preprocess_image


def test_load_and_preprocess_image(tmp_path: Path) -> None:
    """Une image quelconque ressort à la taille demandée, en RGB et en float32.

    L'entrée fait 32×32 et la sortie 64×64 : le redimensionnement est donc bien appliqué,
    et non contourné quand l'image ne fait pas déjà la bonne taille.
    """
    image_path = tmp_path / "sample.png"
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    image.save(image_path)

    arr = load_and_preprocess_image(image_path, image_size=64)

    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.float32
