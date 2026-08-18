
"""Téléchargement des datasets de segmentation depuis Kaggle, avec repli CLI → API.

Usage:
    python -m src.data.download_segmentation_dataset --problem brain_tumor_seg
    python -m src.data.download_segmentation_dataset --problem chest_xray_seg --force

POURQUOI deux chemins de téléchargement : la CLI ``kaggle`` est un sous-processus, et sur
une machine à mémoire juste, le noyau la tue (OOM killer) sur les archives volumineuses —
en laissant un code de retour peu parlant. Le client Python prend alors le relais dans le
processus courant. Si les deux échouent, le message final rappelle les commandes de
diagnostic (``free -h``, ``df -h``, ``dmesg | grep -i oom``), parce que la cause réelle est
presque toujours la mémoire ou le disque, pas Kaggle.

``--force`` **supprime** le dossier cible avant de retélécharger : une extraction par-dessus
une précédente laisserait cohabiter des fichiers de deux versions du dataset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

#: Les datasets de segmentation connus : slug Kaggle et dossier de destination.
#: Ajouter une entrée ici suffit à rendre un nouveau dataset téléchargeable — le choix
#: ``--problem`` est construit à partir de ce dictionnaire.
SEGMENTATION_DATASETS = {
    'brain_tumor_seg': {
        'slug': 'indk214/brain-tumor-dataset-segmentation-and-classification',
        'target_dir': 'data/raw/brain_tumor_segmentation',
    },
    'chest_xray_seg': {
        'slug': 'nikhilpandey360/chest-xray-masks-and-labels',
        'target_dir': 'data/raw/chest_xray_segmentation',
    },
}


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``problem`` (clé de :data:`SEGMENTATION_DATASETS`,
        obligatoire), ``output_dir`` (remplace la destination par défaut) et ``force``
        (supprime la destination avant de retélécharger).
    """
    parser = argparse.ArgumentParser(description='Download segmentation datasets from Kaggle')
    parser.add_argument('--problem', choices=sorted(SEGMENTATION_DATASETS.keys()), required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _extract_all(zip_path: Path, dest: Path) -> None:
    """Extrait toute l'archive dans le dossier de destination.

    Args:
        zip_path: Archive à extraire.
        dest: Dossier de destination, supposé exister.

    Raises:
        zipfile.BadZipFile: Archive corrompue — téléchargement interrompu ou disque plein.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)


def _download_with_cli(slug: str, download_dir: Path) -> Path:
    """Télécharge via la CLI ``kaggle`` (sous-processus) et renvoie l'archive obtenue.

    Args:
        slug: Slug Kaggle du dataset.
        download_dir: Dossier de téléchargement.

    Returns:
        Le chemin de l'archive — celle au nom attendu, ou à défaut la plus récente du
        dossier, Kaggle ne nommant pas toujours le fichier d'après le slug.

    Raises:
        FileNotFoundError: La CLI n'est pas installée — l'appelant bascule alors sur l'API.
        RuntimeError: La CLI a rendu un code non nul, ou n'a produit aucune archive.
    """
    if shutil.which('kaggle') is None:
        raise FileNotFoundError('Kaggle CLI is not installed')

    cmd = ['kaggle', 'datasets', 'download', '-d', slug, '-p', str(download_dir), '-o']
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'Kaggle CLI download failed with exit code {result.returncode}')

    zip_path = download_dir / f"{slug.split('/')[-1]}.zip"
    if zip_path.exists():
        return zip_path

    candidates = sorted(download_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError('Kaggle CLI reported success but no zip archive was found')
    return candidates[0]


def _download_with_api(slug: str, download_dir: Path) -> Path:
    """Télécharge via le client Python Kaggle, dans le processus courant.

    Chemin de repli quand la CLI échoue — voir la note du module. L'import est local à la
    fonction : le paquet ``kaggle`` n'est pas requis tant qu'on n'a pas besoin de ce repli.

    Args:
        slug: Slug Kaggle du dataset.
        download_dir: Dossier de téléchargement.

    Returns:
        Le chemin de l'archive téléchargée.

    Raises:
        ImportError: Le paquet ``kaggle`` n'est pas installé.
        RuntimeError: Le client n'a produit aucune archive.
        OSError: Les identifiants Kaggle sont absents ou invalides.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(download_dir), unzip=False)

    zip_path = download_dir / f"{slug.split('/')[-1]}.zip"
    if zip_path.exists():
        return zip_path

    candidates = sorted(download_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError('Kaggle API reported success but no zip archive was found')
    return candidates[0]


def main() -> None:
    """Télécharge et extrait le dataset de segmentation demandé.

    Ne fait rien si la destination existe et n'est pas vide, sauf ``--force``, qui la
    supprime d'abord. Essaie la CLI, puis l'API en cas d'échec ; si les deux tombent, sort
    avec un message reprenant les **deux** erreurs et les commandes de diagnostic système.

    Raises:
        SystemExit: Les deux chemins de téléchargement ont échoué.
    """
    args = parse_args()
    spec = SEGMENTATION_DATASETS[args.problem]
    target_dir = Path(args.output_dir or spec['target_dir'])
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        print(f'{target_dir} already exists and is not empty; skipping download.')
        return
    if target_dir.exists() and args.force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    download_dir = target_dir.parent
    try:
        zip_path = _download_with_cli(spec['slug'], download_dir)
    except Exception as cli_err:
        print(f'Kaggle CLI path failed ({cli_err}); falling back to Kaggle API client...')
        try:
            zip_path = _download_with_api(spec['slug'], download_dir)
        except Exception as api_err:
            raise SystemExit(
                f'Failed to download dataset {spec["slug"]}. '
                f'CLI error: {cli_err}. API error: {api_err}.\n'
                'If the process is being killed by the OS, check memory/disk on the host '
                '(e.g. `free -h`, `df -h`, `dmesg -T | grep -Ei "killed process|oom" | tail`).'
            ) from api_err

    _extract_all(zip_path, target_dir)
    print(f'Downloaded and extracted {spec["slug"]} to {target_dir}')


if __name__ == '__main__':
    main()
