
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

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
    parser = argparse.ArgumentParser(description='Download segmentation datasets from Kaggle')
    parser.add_argument('--problem', choices=sorted(SEGMENTATION_DATASETS.keys()), required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _extract_all(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)


def main() -> None:
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

    api = KaggleApi()
    api.authenticate()
    download_dir = target_dir.parent
    api.dataset_download_files(spec['slug'], path=str(download_dir), unzip=False)
    zip_name = spec['slug'].split('/')[-1] + '.zip'
    zip_path = download_dir / zip_name
    _extract_all(zip_path, target_dir)
    print(f'Downloaded and extracted {spec["slug"]} to {target_dir}')


if __name__ == '__main__':
    main()
