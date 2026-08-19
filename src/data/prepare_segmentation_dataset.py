
"""Construction du manifeste de segmentation à partir d'un dossier brut.

Stage DVC placé entre le téléchargement et l'entraînement. Il délègue tout le travail
d'appariement à :func:`src.segmentation.datasets.manifest.build_manifest` et se charge du
reste : lire la config, écrire le manifeste, et publier un **résumé**.

Usage:
    python -m src.data.prepare_segmentation_dataset --config configs/brain_tumor_segmentation.yaml

POURQUOI le résumé JSON compte : l'appariement image/masque ignore silencieusement ce qu'il
ne comprend pas (voir la note de ``manifest.py``). Le nombre de lignes et la liste des
labels retenus sont donc le seul signal disponible pour repérer une préparation qui a
« réussi » en ne produisant presque rien. Le lire avant de lancer l'entraînement fait gagner
des heures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.segmentation.datasets.manifest import build_manifest
from src.utils.config import load_config

#: Labels de repli quand la config ne déclare pas ``class_names``. La clé ``brain`` sert de
#: défaut historique ; ``chest`` liste les deux casses rencontrées selon les datasets.
DEFAULT_LABELS = {
    'brain': ['glioma', 'meningioma', 'pituitary', 'notumor'],
    'chest': ['NORMAL', 'ABNORMAL', 'normal', 'abnormal'],
}


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``config``, chemin du YAML du problème de segmentation.
    """
    parser = argparse.ArgumentParser(description='Prepare a segmentation manifest from a raw dataset tree')
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main() -> None:
    """Construit le manifeste, écrit le résumé, et l'affiche.

    Clés lues dans la config : ``raw_dataset_dir`` et ``manifest_path`` (obligatoires),
    ``class_names`` (sinon les labels cérébraux par défaut) et ``dataset_summary_path``
    (sinon ``dataset_summary.json``, à côté du manifeste).

    Le résumé est écrit **et** affiché sur la sortie standard : le fichier sert à DVC et
    aux comparaisons ultérieures, l'affichage sert à celui qui regarde le terminal.

    Raises:
        KeyError: La config ne déclare pas ``raw_dataset_dir`` ou ``manifest_path``.
    """
    args = parse_args()
    cfg = load_config(args.config)
    raw_dir = Path(cfg['raw_dataset_dir'])
    manifest_path = Path(cfg['manifest_path'])
    known_labels = cfg.get('class_names') or DEFAULT_LABELS['brain']
    df = build_manifest(raw_dir=raw_dir, output_csv=manifest_path, known_labels=known_labels)
    summary = {
        'rows': int(len(df)),
        'labels': sorted(df['label'].dropna().unique().tolist()) if not df.empty else [],
        'manifest_path': str(manifest_path),
    }
    summary_path = Path(cfg.get('dataset_summary_path', manifest_path.with_name('dataset_summary.json')))
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
