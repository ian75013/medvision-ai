"""Téléchargement du dataset de radiographies thoraciques depuis Kaggle.

Usage:
    python -m src.data.download_dataset
    python -m src.data.download_dataset --force          # retélécharge et écrase
    python -m src.data.download_dataset --keep-zip       # conserve l'archive

Prérequis : la CLI ``kaggle`` installée et authentifiée (``~/.kaggle/kaggle.json`` ou les
variables ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``). Le script vérifie sa présence avant toute
chose et sort avec un message actionnable plutôt que d'échouer plus loin.

POURQUOI la structure extraite est *cherchée* plutôt que supposée : Kaggle a changé
l'arborescence de cette archive au fil des versions — parfois ``chest_xray/``, parfois
``chest_xray/chest_xray/``. :func:`locate_dataset_root` essaie donc trois pistes, de la plus
précise à la plus générale, pour que le script survive à la prochaine réorganisation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

#: Identifiant Kaggle du dataset de référence (Kermany et al.).
DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_RAW_DIR = Path("data/raw")
#: Nom du dossier attendu après extraction.
EXPECTED_DIRNAME = "chest_xray"


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``raw_dir`` (destination), ``dataset`` (slug Kaggle),
        ``force`` (retélécharger même si présent) et ``keep_zip`` (garder l'archive, utile
        pour éviter un second téléchargement sur connexion lente).
    """
    parser = argparse.ArgumentParser(description="Download the MedVision dataset from Kaggle.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory where the zip will be extracted.")
    parser.add_argument("--dataset", type=str, default=DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing files.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip archive.")
    return parser.parse_args()


def ensure_kaggle_cli() -> None:
    """Vérifie que la CLI Kaggle est installée, et sort proprement sinon.

    Raises:
        SystemExit: La commande ``kaggle`` est introuvable. Le message donne la commande
            d'installation et rappelle les deux façons de fournir les identifiants — c'est
            l'erreur numéro un sur une machine fraîche.
    """
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI not found. Install it with: pip install kaggle\n"
            "Then configure credentials with ~/.kaggle/kaggle.json or the KAGGLE_USERNAME / KAGGLE_KEY environment variables."
        )


def locate_dataset_root(raw_dir: Path) -> Path | None:
    """Retrouve la racine du dataset extrait, quelle que soit la forme de l'archive.

    Trois pistes, dans cet ordre : le dossier attendu à la racine ; le même nom trouvé
    n'importe où dessous ; enfin, tout dossier contenant à la fois ``train``, ``test`` et
    ``val`` (ou ``validation``) — la signature structurelle du dataset, indépendante des
    noms.

    Args:
        raw_dir: Dossier dans lequel l'archive a été extraite.

    Returns:
        La racine du dataset, ou ``None`` s'il n'est pas là.
    """
    direct = raw_dir / EXPECTED_DIRNAME
    if direct.exists():
        return direct

    for candidate in raw_dir.rglob(EXPECTED_DIRNAME):
        if candidate.is_dir():
            return candidate

    for candidate in raw_dir.rglob("train"):
        parent = candidate.parent
        if (parent / "test").exists() and ((parent / "val").exists() or (parent / "validation").exists()):
            return parent

    return None


def download_zip(dataset: str, raw_dir: Path, force: bool) -> Path:
    """Télécharge l'archive Kaggle, ou renvoie celle déjà présente.

    Args:
        dataset: Slug Kaggle du dataset.
        raw_dir: Dossier de destination, créé au besoin.
        force: Retélécharger même si l'archive est déjà là.

    Returns:
        Le chemin de l'archive.

    Raises:
        subprocess.CalledProcessError: La commande ``kaggle`` a échoué — le plus souvent des
            identifiants absents ou périmés, ou les conditions du dataset non acceptées sur
            le site.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "chest-xray-pneumonia.zip"

    if zip_path.exists() and not force:
        return zip_path

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(raw_dir),
    ]
    subprocess.run(cmd, check=True)
    return zip_path


def extract_zip(zip_path: Path, raw_dir: Path, force: bool) -> Path:
    """Extrait l'archive si nécessaire, et renvoie la racine du dataset.

    Args:
        zip_path: Archive téléchargée.
        raw_dir: Dossier d'extraction.
        force: Réextraire même si le dataset semble déjà en place.

    Returns:
        La racine du dataset extrait.

    Raises:
        SystemExit: L'extraction a réussi mais aucune des trois structures connues n'a été
            reconnue — signe que Kaggle a encore changé l'arborescence, et qu'il faut
            compléter :func:`locate_dataset_root`.
        zipfile.BadZipFile: L'archive est corrompue (téléchargement interrompu) — la
            supprimer et relancer.
    """
    dataset_root = locate_dataset_root(raw_dir)
    if dataset_root is not None and not force:
        return dataset_root

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    dataset_root = locate_dataset_root(raw_dir)
    if dataset_root is None:
        raise SystemExit("Download succeeded but the extracted dataset structure was not recognized.")
    return dataset_root


def main() -> None:
    """Télécharge et extrait le dataset, sauf s'il est déjà là.

    L'archive est supprimée après extraction (sauf ``--keep-zip``) : elle pèse plus d'un
    gigaoctet et ne sert plus à rien une fois le dataset en place.

    Affiche en fin de parcours la commande d'entraînement à enchaîner — c'est ce qu'on
    cherche quand on revient sur le projet après des mois.
    """
    args = parse_args()
    ensure_kaggle_cli()

    dataset_root = locate_dataset_root(args.raw_dir)
    if dataset_root is not None and not args.force:
        print(f"Dataset already available at: {dataset_root}")
        return

    zip_path = download_zip(args.dataset, args.raw_dir, args.force)
    dataset_root = extract_zip(zip_path, args.raw_dir, args.force)

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()

    print("Dataset ready.")
    print(f"Kaggle slug: {args.dataset}")
    print(f"Dataset path: {dataset_root}")
    print("You can now train with:")
    print("python -m src.training.train --config configs/config.yaml --model optimized")


if __name__ == "__main__":
    main()
