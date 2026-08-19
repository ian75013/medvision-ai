"""Lecture des volumes médicaux 3D : NIfTI réels, ou tableaux NumPy de démonstration.

``nibabel`` n'est importé qu'au moment d'ouvrir un fichier NIfTI — voir la note du paquet
:mod:`src.dataio`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Suffixes reconnus. ``.gz`` y figure pour les ``.nii.gz``, dont ``Path.suffix`` ne
#: renvoie que la dernière extension.
SUPPORTED_SUFFIXES = {".nii", ".gz", ".npy"}


def load_volume(path: str | Path) -> np.ndarray:
    """Charge un volume médical 3D et le renvoie en ``float32``.

    Deux formats sont acceptés :

    * ``.npy`` — données synthétiques et essais locaux rapides, sans dépendance ;
    * ``.nii`` / ``.nii.gz`` — vrais volumes IRM, si ``nibabel`` est installé.

    Args:
        path: Chemin du volume.

    Returns:
        Le volume, tableau 3D de ``float32``.

    Raises:
        ImportError: Un NIfTI est demandé mais ``nibabel`` n'est pas installé — le message
            donne la commande d'installation.
        ValueError: L'extension n'est pas reconnue, ou le fichier ne contient pas un
            tableau **à trois dimensions**. Cette dernière vérification attrape tôt les
            volumes 4D (séries temporelles, multi-modalités) que la suite du pipeline
            traiterait de travers sans se plaindre.

    Example:
        >>> volume = load_volume("data/demo/patient_001.npy")
        >>> volume.ndim
        3
    """
    path = Path(path)
    if path.suffix == ".npy":
        volume = np.load(path)
    elif path.suffix == ".nii" or path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "nibabel is required to read NIfTI files. Install it with `pip install nibabel`."
            ) from exc
        nii = nib.load(str(path))
        volume = np.asarray(nii.get_fdata(), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported volume format for {path}")

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape} for {path}")

    return volume.astype(np.float32)
