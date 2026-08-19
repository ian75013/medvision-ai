"""Volume IRM 3D → coupes 2D normalisées, prêtes pour un classifieur d'images.

POURQUOI passer par la 2D plutôt que d'entraîner un réseau 3D : un modèle volumétrique
demande beaucoup plus de mémoire, beaucoup plus de données annotées, et n'est pas
convertible aussi simplement en ONNX. Découper le volume en quelques coupes ramène le
problème à de la classification d'images ordinaire, pour laquelle on dispose de dos
pré-entraînés.

POURQUOI les coupes centrales : dans un volume cérébral cadré normalement, les extrémités
sont majoritairement du crâne et de l'air. Les coupes du milieu portent le tissu — et donc
le signal.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class BrainMRI2DPreprocessor:
    """Chaîne de pré-traitement volume → coupes : normaliser, sélectionner, redimensionner.

    Les réglages sont portés par l'instance pour que la même configuration serve
    l'entraînement et l'inférence : c'est un objet qu'on construit une fois depuis la config
    et qu'on passe, plutôt qu'une suite de fonctions dont chaque appelant re-choisit les
    paramètres.

    Attributes:
        image_size: Côté du carré auquel chaque coupe est ramenée.
        normalization: ``"zscore_nonzero"`` (défaut) ou ``"minmax"``. Toute autre valeur
            laisse le volume inchangé — utile pour un volume déjà normalisé en amont.

    Example:
        >>> prep = BrainMRI2DPreprocessor(image_size=128)
        >>> coupes = prep.preprocess_volume(volume, k=5)
        >>> len(coupes), coupes[0].shape
        (5, (1, 128, 128))
    """

    def __init__(self, image_size: int = 128, normalization: str = "zscore_nonzero") -> None:
        """Mémorise les réglages de la chaîne.

        Args:
            image_size: Côté du carré cible, en pixels.
            normalization: Méthode de normalisation — voir les attributs de la classe.
        """
        self.image_size = image_size
        self.normalization = normalization

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """Ramène les intensités du volume à une échelle comparable d'un examen à l'autre.

        Une IRM n'a pas d'unité absolue : contrairement au scanner et à ses unités
        Hounsfield, deux machines — ou la même à deux réglages — donnent des intensités sans
        rapport pour le même tissu. Sans normalisation, le modèle apprend l'appareil plutôt
        que la pathologie.

        ``zscore_nonzero`` centre et réduit **sur les seuls voxels non nuls**. Le fond d'un
        volume cérébral est exactement zéro et occupe la majeure partie de la boîte : inclus
        dans le calcul, il écraserait la moyenne et l'écart-type vers le vide. Les voxels de
        fond sont ensuite remis à 0 pour qu'ils le restent après centrage.

        ``minmax`` ramène linéairement dans [0, 1] — plus simple, mais sensible à un unique
        voxel aberrant, qui suffit à comprimer tout le reste du volume.

        Args:
            volume: Volume 3D ``(H, W, D)``.

        Returns:
            Le volume normalisé, en ``float32``. Un volume entièrement nul, ou de variance
            négligeable, est renvoyé tel quel plutôt que divisé par zéro.
        """
        volume = volume.astype(np.float32)
        if self.normalization == "zscore_nonzero":
            mask = volume != 0
            if np.any(mask):
                values = volume[mask]
                mean = float(values.mean())
                std = float(values.std())
                std = std if std > 1e-6 else 1.0
                volume = np.where(mask, (volume - mean) / std, 0.0)
        elif self.normalization == "minmax":
            vmin = float(volume.min())
            vmax = float(volume.max())
            if vmax - vmin > 1e-6:
                volume = (volume - vmin) / (vmax - vmin)
        return volume

    def select_slices(self, volume: np.ndarray, strategy: str = "central_k", k: int = 5) -> Sequence[np.ndarray]:
        """Extrait ``k`` coupes axiales autour du centre du volume.

        Les indices sont bornés aux extrémités : sur un volume plus mince que ``k``, la
        même coupe est renvoyée plusieurs fois plutôt que de lever — un volume court reste
        exploitable, avec de la redondance.

        Args:
            volume: Volume 3D ``(H, W, D)``, la profondeur étant le dernier axe.
            strategy: Seule ``"central_k"`` est implémentée à ce jour. L'argument existe
                pour que l'ajout d'une stratégie (coupes équiréparties, sélection guidée par
                le tissu) n'oblige pas à changer les appelants.
            k: Nombre de coupes visé. Le nombre réellement renvoyé est ``2 * max(1, k // 2) + 1``,
                c'est-à-dire toujours impair et centré — 5 pour ``k=5``, mais 3 pour ``k=2``.

        Returns:
            La liste des coupes 2D ``(H, W)``, de la plus basse à la plus haute.

        Raises:
            ValueError: La stratégie demandée n'existe pas.
        """
        depth = volume.shape[2]
        if strategy != "central_k":
            raise ValueError(f"Unsupported slice strategy: {strategy}")
        center = depth // 2
        half = max(1, k // 2)
        indices = [min(max(i, 0), depth - 1) for i in range(center - half, center + half + 1)]
        return [volume[:, :, idx] for idx in indices]

    def resize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Redimensionne une coupe au format d'entrée du modèle, par interpolation bilinéaire.

        L'interpolation passe par PyTorch et non par PIL : les intensités normalisées sont
        des flottants pouvant être négatifs (après centrage-réduction), ce que les
        conversions d'image classiques ne préservent pas.

        Args:
            slice_2d: Coupe ``(H, W)``.

        Returns:
            Tableau ``(1, image_size, image_size)`` — le canal est conservé en tête, format
            attendu par les modèles PyTorch de ce projet.
        """
        tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return tensor.squeeze(0).numpy()

    def preprocess_volume(self, volume: np.ndarray, strategy: str = "central_k", k: int = 5) -> list[np.ndarray]:
        """Enchaîne les trois étapes : normaliser, sélectionner, redimensionner.

        L'ordre compte. Normaliser **avant** de sélectionner assure que toutes les coupes
        d'un même volume partagent la même échelle ; normaliser coupe par coupe rendrait
        deux coupes voisines incomparables. Redimensionner **après** normaliser évite que
        l'interpolation ne fabrique des valeurs sur une échelle qui n'existe plus.

        Args:
            volume: Volume 3D ``(H, W, D)``.
            strategy: Stratégie de sélection — voir :meth:`select_slices`.
            k: Nombre de coupes visé.

        Returns:
            La liste des coupes prêtes pour le modèle, chacune ``(1, image_size, image_size)``.
        """
        volume = self.normalize(volume)
        slices = self.select_slices(volume, strategy=strategy, k=k)
        return [self.resize_slice(slice_2d) for slice_2d in slices]
