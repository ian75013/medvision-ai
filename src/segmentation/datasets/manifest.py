from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTS_PRIORITY = {".png", ".bmp", ".tif", ".tiff"}


def _normalize_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(?:_mask|mask|_seg|_annotation|_label)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def _label_from_path(path: Path, known_labels: Iterable[str]) -> str | None:
    parts = [p.lower() for p in path.parts]
    normalized_parts = [p.replace("_", " ").strip() for p in parts]

    for label in known_labels:
        lbl = label.lower().replace("_", " ").strip()
        for part in normalized_parts:
            if lbl == part or lbl in part:
                return label
    return None


def _looks_like_mask(path: Path) -> bool:
    name = path.stem.lower()
    parent = path.parent.name.lower()
    return (
        "mask" in name
        or name.endswith("_seg")
        or "label" in name
        or parent in {"mask", "masks", "seg", "segs", "labels"}
    )


def build_manifest(raw_dir: str | Path, output_csv: str | Path, known_labels: list[str]) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)

    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_normalize_stem(path), []).append(path)

    rows = []

    for key, candidates in grouped.items():
        if len(candidates) < 2:
            continue

        explicit_masks = [p for p in candidates if _looks_like_mask(p)]
        explicit_images = [p for p in candidates if not _looks_like_mask(p)]

        image_path = None
        mask_path = None

        if explicit_masks and explicit_images:
            mask_path = explicit_masks[0]
            image_path = explicit_images[0]
        else:
            jpgs = [p for p in candidates if p.suffix.lower() in {".jpg", ".jpeg"}]
            mask_like = [p for p in candidates if p.suffix.lower() in MASK_EXTS_PRIORITY]

            if jpgs and mask_like:
                image_path = jpgs[0]
                mask_path = next((p for p in mask_like if p != image_path), None)

        if image_path is None or mask_path is None:
            continue

        label = _label_from_path(image_path, known_labels)
        split = "test" if "test" in str(image_path).lower() else "train"

        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": label or "unknown",
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df