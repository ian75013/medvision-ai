"""Manipulation des chemins d'artefacts.

Tenu volontairement minuscule : tout ce que le projet fait de commun avec le système de
fichiers, c'est créer le dossier de sortie avant d'y écrire.
"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Crée le dossier s'il n'existe pas et renvoie son ``Path``.

    Renvoyer le chemin permet de l'enchaîner sur place —
    ``model_dir = ensure_dir(cfg["model_dir"]) / "modele.keras"`` — au lieu de séparer la
    création de l'usage, où l'une des deux finit par être oubliée.

    Les parents sont créés au besoin et un dossier déjà présent n'est pas une erreur : un
    entraînement relancé doit repartir sans se plaindre.

    Args:
        path: Dossier à garantir.

    Returns:
        Le dossier, sous forme de ``Path``.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
