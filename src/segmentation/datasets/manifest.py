
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:_mask|mask|_seg|_annotation|_label)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def _label_from_path(path: Path, known_labels: Iterable[str]) -> str | None:
    parts = [p.lower() for p in path.parts]
    for label in known_labels:
        if label.lower() in parts:
            return label
    return None


def build_manifest(raw_dir: str | Path, output_csv: str | Path, known_labels: list[str]) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)
    images = []
    masks = []
    for path in raw_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            low = str(path).lower()
            if 'mask' in low or 'seg' in low or 'label' in low or path.parent.name.lower() in {'masks', 'mask'}:
                masks.append(path)
            else:
                images.append(path)

    mask_index = {}
    for mask in masks:
        mask_index.setdefault(_normalize_stem(mask), []).append(mask)

    rows = []
    for image in images:
        key = _normalize_stem(image)
        candidates = mask_index.get(key, [])
        if not candidates:
            continue
        label = _label_from_path(image, known_labels)
        rows.append({
            'image_path': str(image),
            'mask_path': str(candidates[0]),
            'label': label or 'unknown',
            'split': 'test' if 'test' in str(image).lower() else 'train',
        })
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
