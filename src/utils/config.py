"""Lecture des fichiers de configuration YAML du projet.

POURQUOI toutes les configs passent par ici : chaque entraînement est décrit par un YAML
(``configs/*.yaml``) plutôt que par des arguments en ligne de commande, pour qu'un run soit
rejouable des mois plus tard à partir d'un fichier versionné. Ce module est le seul point
de lecture — et donc le seul endroit où ajouter une validation si le besoin s'en fait sentir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Charge un YAML de configuration et vérifie que sa racine est bien un dictionnaire.

    ``yaml.safe_load`` est utilisé plutôt que ``load`` : la variante complète sait
    instancier des objets Python arbitraires, ce qui donne l'exécution de code à quiconque
    fournit le fichier.

    Args:
        config_path: Chemin du fichier YAML.

    Returns:
        La configuration, sous forme de dictionnaire.

    Raises:
        FileNotFoundError: Le fichier n'existe pas.
        ValueError: La racine du document n'est pas un mapping — cas typique d'un YAML
            vide ou réduit à une liste, qui ferait échouer tous les ``cfg.get()`` plus loin
            avec un message beaucoup moins clair.
        yaml.YAMLError: Le fichier est syntaxiquement invalide.

    Example:
        >>> cfg = load_config("configs/chest_xray.yaml")
        >>> cfg["image_size"]
        224
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at top level.")
    return config
