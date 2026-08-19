"""Reproductibilité de la voie PyTorch : une graine pour tous les générateurs en jeu.

À ne pas confondre avec les ``set_seed`` locaux des scripts TensorFlow
(:mod:`src.training.train`, :mod:`src.training.train_brain_mri`) : ceux-là fixent la graine
de TensorFlow, celui-ci celle de PyTorch. Les deux existent parce qu'aucun script n'importe
les deux frameworks à la fois — c'est justement ce qu'on cherche à éviter.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixe la graine de tous les générateurs aléatoires et sérialise le calcul PyTorch.

    Quatre sources d'aléa sont couvertes : le ``random`` de Python, NumPy, PyTorch (CPU) et
    les GPU s'il y en a. ``PYTHONHASHSEED`` est posé pour figer l'ordre d'itération des
    ensembles, qui influe sur certains parcours de données.

    ``torch.set_num_threads(1)`` n'est pas là pour la performance mais pour le
    déterminisme : le parallélisme intra-op de PyTorch fait varier l'ordre des sommations
    en virgule flottante, et deux exécutions de la même graine cessent de donner exactement
    le même résultat. Sur une machine d'entraînement où la reproductibilité importe moins
    que la vitesse, c'est la ligne à retirer en premier.

    Args:
        seed: Graine commune.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
