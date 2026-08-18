"""``Dataset`` PyTorch qui déplie des volumes IRM 3D en coupes 2D.

Fait le pont entre le CSV de découpage (:mod:`src.datasets.splitters`) et le chargeur
PyTorch de :mod:`src.training.train_classifier`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataio.nifti_loader import load_volume
from src.preprocessing.brain_mri_2d import BrainMRI2DPreprocessor


class BrainMRISliceDataset(Dataset):
    """Un volume par ligne de CSV, K coupes par volume, tout chargé en mémoire à la construction.

    Chaque ligne du CSV désigne un patient et son volume ; la construction du dataset le
    déplie en K coupes représentatives, qui deviennent autant d'exemples.

    POURQUOI tout matérialiser d'emblée plutôt que lire à la demande : décoder un NIfTI est
    lent, et le relire à chaque époque dominerait le temps d'entraînement. Comme on ne garde
    que K coupes réduites par patient, l'empreinte mémoire reste modeste. **La limite est
    là** : ce dataset convient aux quelques centaines de volumes des jeux de démonstration,
    pas à une cohorte complète — au-delà, il faut passer à une lecture paresseuse.

    Conséquence à connaître pour l'évaluation : plusieurs exemples partagent le même
    patient. Le découpage doit donc se faire **avant**, au niveau du patient (voir
    :mod:`src.datasets.splitters`) ; ``patient_id`` est conservé dans chaque exemple pour
    permettre une agrégation par patient au moment de mesurer.

    Attributes:
        csv_path: CSV source.
        df: Le CSV chargé.
        preprocessor: La chaîne de pré-traitement volume → coupes.
        slice_strategy: Stratégie de sélection des coupes.
        k: Nombre de coupes visé par volume.
        samples: Les exemples matérialisés, sous forme ``(coupe, label, patient_id)``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 128,
        normalization: str = "zscore_nonzero",
        slice_strategy: str = "central_k",
        k: int = 5,
    ) -> None:
        """Lit le CSV et matérialise immédiatement toutes les coupes.

        Args:
            csv_path: CSV listant les volumes — colonnes ``path``, ``label``, ``patient_id``.
            image_size: Côté du carré auquel chaque coupe est ramenée.
            normalization: Méthode de normalisation — voir
                :class:`src.preprocessing.brain_mri_2d.BrainMRI2DPreprocessor`.
            slice_strategy: Stratégie de sélection des coupes.
            k: Nombre de coupes visé par volume.

        Raises:
            FileNotFoundError: Le CSV, ou l'un des volumes qu'il désigne, est introuvable.
        """
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.preprocessor = BrainMRI2DPreprocessor(image_size=image_size, normalization=normalization)
        self.slice_strategy = slice_strategy
        self.k = k
        self.samples: list[tuple[np.ndarray, int, str]] = []
        self._materialize_samples()

    def _materialize_samples(self) -> None:
        """Charge chaque volume, le découpe, et empile les coupes dans ``self.samples``.

        Appelée une seule fois, à la construction. Le label et l'identifiant patient sont
        recopiés sur chacune des coupes issues d'un même volume.
        """
        for row in self.df.itertuples(index=False):
            volume = load_volume(row.path)
            slices = self.preprocessor.preprocess_volume(volume, strategy=self.slice_strategy, k=self.k)
            for slice_arr in slices:
                self.samples.append((slice_arr.astype(np.float32), int(row.label), str(row.patient_id)))

    def __len__(self) -> int:
        """Nombre d'exemples, c'est-à-dire de **coupes** — pas de volumes.

        Returns:
            Le nombre de coupes matérialisées, de l'ordre de ``K × nombre de volumes``.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        """Renvoie une coupe sous la forme attendue par la boucle d'entraînement.

        Le dictionnaire porte les clés ``image`` et ``label`` que
        :func:`src.training.trainer.run_epoch` va chercher — les renommer ici casse la
        boucle d'entraînement.

        Args:
            idx: Indice de la coupe.

        Returns:
            ``{"image": FloatTensor, "label": LongTensor, "patient_id": str}``. Le type
            ``long`` du label n'est pas décoratif : ``CrossEntropyLoss`` l'exige.
        """
        image, label, patient_id = self.samples[idx]
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
        }
