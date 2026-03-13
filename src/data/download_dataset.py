from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_RAW_DIR = Path("data/raw")
EXPECTED_DIRNAME = "chest_xray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MedVision dataset from Kaggle.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory where the zip will be extracted.")
    parser.add_argument("--dataset", type=str, default=DATASET_SLUG, help="Kaggle dataset slug.")
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing files.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip archive.")
    return parser.parse_args()


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI not found. Install it with: pip install kaggle\n"
            "Then configure credentials with ~/.kaggle/kaggle.json or the KAGGLE_USERNAME / KAGGLE_KEY environment variables."
        )


def locate_dataset_root(raw_dir: Path) -> Path | None:
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
        "-f",
        "chest-xray-pneumonia.zip",
    ]
    subprocess.run(cmd, check=True)
    return zip_path


def extract_zip(zip_path: Path, raw_dir: Path, force: bool) -> Path:
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
