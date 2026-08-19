"""Chargement d'une image 2D et mise en forme pour le modèle.

C'est la définition de référence du pré-traitement d'inférence : l'API, l'UI Streamlit et
les scripts de prédiction passent tous par ici. Voir la note du paquet
(:mod:`src.preprocessing`) sur ce qui arrive quand deux chemins de code divergent sur ce
point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_and_preprocess_image(image_path: str | Path, image_size: int = 224) -> np.ndarray:
    """Ouvre une image, la met en RGB, la redimensionne et la ramène dans [0, 1].

    La conversion en RGB est systématique alors que les images médicales sont en niveaux de
    gris : les dos pré-entraînés attendent trois canaux, et un PNG médical peut aussi
    arriver en RGBA (le canal alpha fausserait alors la forme du tableau).

    Le redimensionnement ne préserve pas les proportions. C'est cohérent avec
    l'entraînement, qui utilise la même déformation — la corriger ici seulement dégraderait
    les prédictions.

    Args:
        image_path: Chemin de l'image.
        image_size: Côté du carré cible, en pixels. **Doit valoir la taille vue à
            l'entraînement** ; la valeur par défaut correspond aux classifieurs, la
            segmentation travaille en 256.

    Returns:
        Tableau ``(image_size, image_size, 3)`` de ``float32`` dans [0, 1], sans dimension
        de lot — c'est à l'appelant d'ajouter l'axe avec ``np.expand_dims``.

    Raises:
        FileNotFoundError: Le fichier n'existe pas.
        PIL.UnidentifiedImageError: Le fichier n'est pas une image lisible.

    Example:
        >>> image = load_and_preprocess_image("data/samples/case_01.png", image_size=256)
        >>> image.shape
        (256, 256, 3)
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr
