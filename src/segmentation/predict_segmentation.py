"""Inférence de segmentation en ligne de commande : une image entre, un masque sort.

POURQUOI ce script existe à côté de l'API et de l'UI : il rejoue un modèle **sans rien
d'autre** — ni FastAPI, ni Streamlit, ni ONNX Runtime. C'est l'outil de vérification
manuelle après un entraînement (« le modèle que je viens de sauvegarder segmente-t-il
vraiment quelque chose ? ») et le point de comparaison quand l'API renvoie un résultat
douteux : si le ``.keras`` d'origine donne le bon masque et pas l'API, le problème est dans
la conversion ONNX ou le pré-traitement de l'API, pas dans le modèle.

Il charge donc un ``.keras`` via TensorFlow, là où la production sert de l'ONNX.

Usage:
    python -m src.segmentation.predict_segmentation \\
        --model-path artifacts/models/brain_tumor_segmentation_unet.keras \\
        --image-path data/raw/brain_tumor_segmentation/images/case_042.png \\
        --output-dir artifacts/inference

Écrit dans ``--output-dir`` : ``predicted_mask.png`` (le masque seul),
``predicted_overlay.png`` (le masque posé sur l'image) et ``prediction.json`` (les chemins
produits et, pour un modèle multitâche, les probabilités de classe).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.image_loader import load_and_preprocess_image
from src.segmentation.overlays import mask_to_pil, save_overlay


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``model_path`` et ``image_path`` (obligatoires),
        ``output_dir`` (défaut ``artifacts/inference``) et ``image_size`` (défaut 256, qui
        doit correspondre à la taille vue à l'entraînement, sinon le modèle refuse l'entrée).
    """
    parser = argparse.ArgumentParser(description='Run segmentation inference')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--output-dir', default='artifacts/inference')
    parser.add_argument('--image-size', type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Charge le modèle, segmente une image et écrit masque, superposition et JSON.

    Le modèle est chargé avec ``compile=False`` : les métriques personnalisées
    (``dice_coefficient``, ``iou_score``) ne sont pas enregistrables telles quelles, et
    Keras échouerait à reconstruire l'optimiseur alors qu'on ne veut qu'un passage avant.

    Deux formes de sortie sont acceptées, ce qui permet d'utiliser le même script pour les
    deux architectures : un dictionnaire (U-Net multitâche, dont on lit
    ``segmentation_output`` et, si présent, ``classification_output``) ou un tableau nu
    (U-Net de segmentation seule).

    Le masque est binarisé au seuil 0.5 avant écriture — c'est le même seuil que celui des
    métriques d'évaluation, et le déplacer ici sans le déplacer là-bas rendrait les scores
    incomparables.

    Le résultat est écrit sur disque **et** affiché sur la sortie standard, pour être
    directement exploitable dans un script appelant.
    """
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
