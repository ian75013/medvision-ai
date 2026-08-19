"""Téléchargement du dataset d'IRM cérébrales (types de tumeurs) depuis Kaggle.

Usage:
    python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml

À la différence de :mod:`src.data.download_dataset`, tout est piloté par la config
(``dataset_dir``, ``dataset_slug``, ``archive_name``, ``training_subdir``,
``testing_subdir``) : ce dataset a plusieurs variantes sur Kaggle, et changer de source ne
doit pas demander de toucher au code.

L'archive Kaggle extrait ses dossiers ``Training``/``Testing`` à une profondeur qui varie
selon la variante. Le script les **recherche** puis les recopie à l'emplacement attendu
plutôt que de supposer la structure. La copie est faite avec ``dirs_exist_ok=True`` : une
extraction relancée complète l'existant au lieu d'échouer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``config`` (YAML, obligatoire), ``force`` (retélécharger
        même si le dataset est présent) et ``keep_zip`` (conserver l'archive).
    """
    parser = argparse.ArgumentParser(description="Download brain tumor MRI dataset from Kaggle")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    return parser.parse_args()


def ensure_kaggle_cli() -> None:
    """Vérifie que la CLI Kaggle est installée, et sort proprement sinon.

    Raises:
        SystemExit: La commande ``kaggle`` est introuvable.
    """
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure credentials."
        )


def main() -> None:
    """Télécharge, extrait et range le dataset d'IRM à l'emplacement de la config.

    Sort immédiatement si ``Training/`` et ``Testing/`` sont déjà en place (sauf
    ``--force``) : la vérification a lieu **avant** le contrôle de la CLI Kaggle, pour qu'un
    pipeline puisse rejouer ce stage sur une machine sans identifiants Kaggle.

    Si l'archive attendue n'est pas là après téléchargement — Kaggle nomme parfois le
    fichier autrement — la plus récente du dossier est retenue.

    Affiche pour finir les classes trouvées : c'est le contrôle qui permet de voir tout de
    suite une extraction partielle.

    Raises:
        KeyError: La config ne déclare pas ``dataset_dir`` ou ``dataset_slug``.
        SystemExit: Le téléchargement s'est terminé sans produire d'archive.
        subprocess.CalledProcessError: La commande ``kaggle`` a échoué.
    """
    args = parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg["dataset_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    slug = cfg["dataset_slug"]
    archive_name = cfg.get("archive_name", "brain-tumor-mri-dataset.zip")
    zip_path = raw_dir.parent / archive_name

    training_dir = raw_dir / cfg.get("training_subdir", "Training")
    testing_dir = raw_dir / cfg.get("testing_subdir", "Testing")
    if training_dir.exists() and testing_dir.exists() and not args.force:
        print(f"Dataset already present at {raw_dir}")
        return

    ensure_kaggle_cli()

    cmd = [
        "kaggle", "datasets", "download",
        "-d", slug,
        "-p", str(raw_dir.parent),
        "-o",
    ]
    subprocess.run(cmd, check=True)

    if not zip_path.exists():
        candidates = sorted(raw_dir.parent.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit("Kaggle download completed but no zip archive was found.")
        zip_path = candidates[0]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir.parent)

    # Move extracted root if needed
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)

    extracted_training = list(raw_dir.parent.rglob(cfg.get("training_subdir", "Training")))
    extracted_testing = list(raw_dir.parent.rglob(cfg.get("testing_subdir", "Testing")))
    if extracted_training and extracted_testing:
        train_src = extracted_training[0]
        test_src = extracted_testing[0]
        if train_src.parent != raw_dir:
            shutil.copytree(train_src, raw_dir / train_src.name, dirs_exist_ok=True)
        if test_src.parent != raw_dir:
            shutil.copytree(test_src, raw_dir / test_src.name, dirs_exist_ok=True)

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()

    print(f"Dataset ready at {raw_dir}")
    print(f"Classes found in training dir: {[p.name for p in sorted((raw_dir / cfg.get('training_subdir', 'Training')).iterdir()) if p.is_dir()]}")


if __name__ == "__main__":
    main()
