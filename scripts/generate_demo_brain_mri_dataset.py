"""Génère un dataset d'IRM cérébrales **synthétique** : volumes, métadonnées et découpage.

Usage:
    python scripts/generate_demo_brain_mri_dataset.py
    python scripts/generate_demo_brain_mri_dataset.py --num-patients-per-class 30

POURQUOI ce script existe : il permet de dérouler toute la chaîne volumétrique — lecture
NIfTI/NumPy, découpage par patient, extraction de coupes, entraînement PyTorch — sans
télécharger un dataset médical de plusieurs gigaoctets, et sans manipuler de données de
patients réels. C'est aussi ce qui rend la chaîne testable en intégration continue.

Ce qu'il produit : des volumes de bruit avec une ellipse « cérébrale » plus lumineuse et,
pour la classe 1, une sphère « tumorale » plus lumineuse encore. Un modèle apprend cette
tâche trivialement — **aucun score obtenu sur ces données n'a de sens**. La question à
laquelle ils répondent est « le pipeline tourne-t-il de bout en bout ? », pas « le modèle
est-il bon ? ».

Le découpage est délégué à :func:`src.datasets.splitters.create_patient_level_splits`, donc
par patient — même sur des données jouets, on ne s'entraîne pas à faire fuiter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.splitters import create_patient_level_splits
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


def make_synthetic_volume(label: int, shape: tuple[int, int, int] = (96, 96, 24)) -> np.ndarray:
    """Fabrique un volume 3D de bruit contenant une forme « cérébrale », et une tumeur si besoin.

    Construction : un fond gaussien, une ellipse plus lumineuse étendue sur toute la
    profondeur (le « cerveau »), puis — pour ``label == 1`` — une sphère plus lumineuse
    décentrée (la « tumeur »). Les valeurs sont bornées à [-1.5, 2.5] pour rester dans une
    plage plausible après normalisation.

    Args:
        label: 0 pour un volume sain, 1 pour un volume porteur d'une lésion.
        shape: Dimensions ``(H, W, D)`` du volume.

    Returns:
        Le volume, tableau ``float32`` de la forme demandée.

    Note:
        Le générateur aléatoire est créé **à chaque appel** sans graine, ce qui rend le
        bruit non reproductible même après ``set_seed`` — les volumes diffèrent d'une
        exécution à l'autre. Sans conséquence sur des données jouets, mais à savoir avant
        de vouloir comparer deux exécutions à l'octet près.
    """
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
    """Écrit les volumes, le CSV de métadonnées et les fichiers de découpage.

    Produit ``--num-patients-per-class`` volumes par classe (sain / tumeur) sous
    ``--output-dir``, un ``metadata.csv`` avec ``patient_id``, ``path`` et ``label``, puis
    délègue le découpage train/val/test à
    :func:`src.datasets.splitters.create_patient_level_splits`, qui écrit ses CSV sous
    ``--processed-dir``.

    Ces chemins sont ceux qu'attendent les configs de démonstration : les changer oblige à
    changer la config d'entraînement en conséquence.
    """
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
