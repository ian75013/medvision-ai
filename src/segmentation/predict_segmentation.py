
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.image_loader import load_and_preprocess_image
from src.segmentation.overlays import mask_to_pil, save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run segmentation inference')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-dir', default='artifacts/inference')
    parser.add_argument('--image-size', type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path, compile=False)
    image = load_and_preprocess_image(args.image_path, image_size=args.image_size)
    batch = np.expand_dims(image, axis=0)
    raw = model.predict(batch, verbose=0)
    if isinstance(raw, dict):
        mask = raw['segmentation_output'][0, ..., 0]
        cls = raw.get('classification_output')
        class_probs = cls[0].tolist() if cls is not None else None
    else:
        mask = raw[0, ..., 0]
        class_probs = None
    pred = (mask >= 0.5).astype(np.float32)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    mask_path = outdir / 'predicted_mask.png'
    overlay_path = outdir / 'predicted_overlay.png'
    mask_to_pil(pred).save(mask_path)
    save_overlay(image, pred, overlay_path)
    payload = {'mask_path': str(mask_path), 'overlay_path': str(overlay_path), 'classification_probabilities': class_probs}
    (outdir / 'prediction.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
