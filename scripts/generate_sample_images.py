#!/usr/bin/env python3
"""
Purpose: Generate synthetic PNG sample images for local UI testing.
         Populates data/raw/ with small grayscale images so the Prediction
         Studio can browse a dataset without requiring a Kaggle download.
Usage:   python scripts/generate_sample_images.py [--out-dir data/raw] [--seed 42]
Arguments:
  --out-dir   Root data directory (default: data/raw)
  --seed      RNG seed for reproducibility (default: 42)
Exit codes:
  0  All images written successfully.
  1  Dependency missing (Pillow / numpy not installed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("ERROR: numpy and Pillow are required. Run: pip install numpy Pillow", file=sys.stderr)
    sys.exit(1)


_SIZE = (224, 224)
_SAMPLES_PER_CLASS = 3


def _save_synthetic_image(path: Path, rng: np.random.Generator, patch_value: int = 200) -> None:
    """Generate a grayscale image with noise background and a light square patch."""
    img = rng.integers(20, 60, size=(*_SIZE, 1), dtype=np.uint8)
    cy, cx = _SIZE[0] // 2, _SIZE[1] // 2
    r = _SIZE[0] // 5
    img[cy - r : cy + r, cx - r : cx + r] = patch_value
    Image.fromarray(img.squeeze(), mode="L").save(path)


def _generate_chest_xray(out_dir: Path, rng: np.random.Generator) -> None:
    """Écrit les images synthétiques du problème « radiographie thoracique ».

    Les deux classes se distinguent par la luminosité du carré central (80 contre 180) :
    un modèle réel n'y verrait rien de médical, mais l'UI a de quoi afficher deux classes
    visuellement différentes.

    L'arborescence reproduit celle du vrai dataset (``chest_xray/test/<CLASSE>/``), sans
    quoi le navigateur d'images ne les trouverait pas.

    Args:
        out_dir: Racine des données brutes.
        rng: Générateur aléatoire, pour que la sortie soit reproductible.
    """
    classes = {"NORMAL": 80, "PNEUMONIA": 180}
    for class_name, patch_val in classes.items():
        folder = out_dir / "chest_xray" / "test" / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(_SAMPLES_PER_CLASS):
            _save_synthetic_image(folder / f"sample_{i:03d}.png", rng, patch_val)
        print(f"  {folder}: {_SAMPLES_PER_CLASS} images")


def _generate_brain_mri(out_dir: Path, rng: np.random.Generator) -> None:
    """Écrit les images synthétiques du problème « IRM cérébrale », quatre classes.

    Les noms de classes et le dossier ``Testing/`` reprennent exactement ceux du dataset
    Kaggle : l'UI et le registre s'appuient dessus.

    Args:
        out_dir: Racine des données brutes.
        rng: Générateur aléatoire.
    """
    classes = {
        "glioma": 210,
        "meningioma": 170,
        "notumor": 50,
        "pituitary": 140,
    }
    for class_name, patch_val in classes.items():
        folder = out_dir / "brain_tumor_mri" / "Testing" / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(_SAMPLES_PER_CLASS):
            _save_synthetic_image(folder / f"sample_{i:03d}.png", rng, patch_val)
        print(f"  {folder}: {_SAMPLES_PER_CLASS} images")


def _generate_brain_tumor_segmentation(out_dir: Path, rng: np.random.Generator) -> None:
    """Écrit les images de segmentation **et** le manifeste minimal qui les référence.

    Le navigateur d'images de la segmentation ne lit pas un dossier mais un
    ``manifest.csv`` : générer les seules images ne suffirait pas à rendre le problème
    visible dans l'UI. Le manifeste est écrit dans ``data/processed/`` — à l'endroit où la
    vraie préparation le poserait.

    Ce manifeste de démonstration n'a **pas** de colonne ``mask_path`` : il suffit à
    l'affichage, pas à un entraînement, qui exige le manifeste complet produit par
    :func:`src.segmentation.datasets.manifest.build_manifest`.

    Args:
        out_dir: Racine des données brutes.
        rng: Générateur aléatoire.
    """
    import csv

    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    img_dir = out_dir / "brain_tumor_segmentation" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path("data/processed/brain_tumor_segmentation")
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "manifest.csv"

    rows = []
    for class_idx, class_name in enumerate(classes):
        for i in range(_SAMPLES_PER_CLASS):
            fname = f"{class_name}_sample_{i:03d}.png"
            img_path = img_dir / fname
            _save_synthetic_image(img_path, rng, patch_value=60 + class_idx * 50)
            rows.append({"image_path": str(img_path), "label": class_name, "split": "test"})
    print(f"  {img_dir}: {len(rows)} images")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {manifest_path}: manifest written ({len(rows)} rows)")


def main() -> None:
    """Génère les images synthétiques des trois problèmes sous ``--out-dir``.

    Un seul générateur aléatoire, créé à partir de ``--seed``, est passé aux trois
    fonctions : relancer le script avec la même graine réécrit exactement les mêmes images.

    Rappelle en fin d'exécution que ces données sont synthétiques et donne la commande pour
    récupérer les vraies.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir

    print(f"Generating synthetic sample images under {out_dir}/")
    _generate_chest_xray(out_dir, rng)
    _generate_brain_mri(out_dir, rng)
    _generate_brain_tumor_segmentation(out_dir, rng)
    print("\nDone. These images are synthetic (noise + patch) — for UI testing only.")
    print("To replace with real data: bash scripts/download_dataset.sh")


if __name__ == "__main__":
    main()
