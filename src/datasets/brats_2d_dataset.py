from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataio.nifti_loader import load_volume
from src.preprocessing.brain_mri_2d import BrainMRI2DPreprocessor


class BrainMRISliceDataset(Dataset):
    """Turns one 3D volume into several 2D slices.

    Each row in the input CSV is one patient/volume. During dataset construction,
    it expands every volume into K representative slices. That gives a simple but
    practical baseline for Sprint 2.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 128,
        normalization: str = "zscore_nonzero",
        slice_strategy: str = "central_k",
        k: int = 5,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.preprocessor = BrainMRI2DPreprocessor(image_size=image_size, normalization=normalization)
        self.slice_strategy = slice_strategy
        self.k = k
        self.samples: list[tuple[np.ndarray, int, str]] = []
        self._materialize_samples()

    def _materialize_samples(self) -> None:
        for row in self.df.itertuples(index=False):
            volume = load_volume(row.path)
            slices = self.preprocessor.preprocess_volume(volume, strategy=self.slice_strategy, k=self.k)
            for slice_arr in slices:
                self.samples.append((slice_arr.astype(np.float32), int(row.label), str(row.patient_id)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        image, label, patient_id = self.samples[idx]
        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
        }
