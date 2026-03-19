from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.splitters import create_patient_level_splits
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


def make_synthetic_volume(label: int, shape: tuple[int, int, int] = (96, 96, 24)) -> np.ndarray:
    rng = np.random.default_rng()
    volume = rng.normal(loc=0.0, scale=0.2, size=shape).astype(np.float32)

    yy, xx, zz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cy, cx, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2

    brain_mask_2d = ((yy[:, :, 0] - cy) ** 2) / (0.42 * shape[0]) ** 2 + ((xx[:, :, 0] - cx) ** 2) / (0.36 * shape[1]) ** 2 <= 1.0
    volume[brain_mask_2d, :] += 0.8

    if label == 1:
        radius = min(shape) // 8
        tumor_mask = (yy - (cy + 10)) ** 2 + (xx - (cx - 8)) ** 2 + (zz - cz) ** 2 <= radius**2
        volume[tumor_mask] += 1.3

    volume = np.clip(volume, -1.5, 2.5)
    return volume.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic brain MRI dataset for Sprint 2 demo")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/brain_mri_demo"))
    parser.add_argument("--num-patients-per-class", type=int, default=12)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/brain_mri_demo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    processed_dir = ensure_dir(args.processed_dir)

    rows = []
    for label in [0, 1]:
        class_name = "normal" if label == 0 else "tumor"
        class_dir = ensure_dir(output_dir / class_name)
        for idx in range(args.num_patients_per_class):
            patient_id = f"{class_name}_{idx:03d}"
            volume = make_synthetic_volume(label)
            path = class_dir / f"{patient_id}.npy"
            np.save(path, volume)
            rows.append({"patient_id": patient_id, "path": str(path), "label": label})

    metadata_path = output_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    create_patient_level_splits(metadata_path, processed_dir, seed=args.seed)
    print(f"Synthetic dataset generated under {output_dir}")
    print(f"Metadata CSV: {metadata_path}")


if __name__ == "__main__":
    main()
