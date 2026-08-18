"""Du ``manifest.csv`` aux ``tf.data.Dataset`` d'entraînement, validation et test.

POURQUOI ce module existe : il est la frontière entre « des fichiers sur le disque » et
« des tenseurs par lots ». Tout ce qui concerne la lecture des images, la binarisation des
masques et le découpage en trois jeux est ici, et nulle part ailleurs — l'entraînement ne
connaît que les ``Dataset`` qu'il reçoit.

Deux formes de sortie, selon ``task_type`` :

* ``'multitask'`` — ``(image, {'segmentation_output': masque, 'classification_output': label})``,
  pour le U-Net à deux têtes ;
* toute autre valeur — ``(image, masque)``, pour le U-Net de segmentation seule.

Les noms de clés ne sont pas décoratifs : Keras apparie les sorties du modèle aux cibles
**par nom**. Les renommer ici sans les renommer dans ``models/unet.py`` casse
l'entraînement avec un message peu clair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE


def _read_image(path: str, image_size: int) -> np.ndarray:
    """Charge une image en RGB, la redimensionne et la ramène dans [0, 1].

    La conversion en RGB est systématique même pour des images médicales en niveaux de
    gris : les encodeurs pré-entraînés attendent trois canaux.

    Args:
        path: Chemin de l'image sur le disque.
        image_size: Côté du carré cible, en pixels.

    Returns:
        Tableau ``(image_size, image_size, 3)`` de ``float32`` dans [0, 1].
    """
    arr = np.asarray(Image.open(path).convert('RGB').resize((image_size, image_size)), dtype=np.float32) / 255.0
    return arr


def _read_mask(path: str, image_size: int) -> np.ndarray:
    """Charge un masque, le redimensionne et le **rebinarise** à 0 ou 1.

    Le seuillage après redimensionnement est indispensable : l'interpolation crée des
    valeurs intermédiaires sur les contours, et un masque à 0.4 n'est ni du fond ni de la
    lésion pour une perte binaire.

    Args:
        path: Chemin du masque sur le disque (niveaux de gris ou couleur).
        image_size: Côté du carré cible, en pixels — doit valoir celui de l'image associée.

    Returns:
        Tableau ``(image_size, image_size, 1)`` de ``float32`` valant exactement 0.0 ou 1.0.
    """
    mask = Image.open(path).convert('L').resize((image_size, image_size))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return arr[..., None]


def _make_dataset(df: pd.DataFrame, image_size: int, batch_size: int, class_to_idx: dict[str, int], task_type: str, repeat: bool = True):
    """Fabrique un ``tf.data.Dataset`` par lots à partir d'un sous-ensemble du manifeste.

    Les images sont lues paresseusement par un générateur Python plutôt que chargées d'un
    bloc : les datasets d'IRM dépassent la mémoire d'une machine de développement.

    Args:
        df: Lignes du manifeste à servir — colonnes ``image_path``, ``mask_path``, ``label``.
        image_size: Côté du carré cible, en pixels.
        batch_size: Taille des lots produits.
        class_to_idx: Table nom de classe → entier. Un label absent de la table tombe sur 0,
            de sorte qu'un manifeste partiellement étiqueté n'interrompt pas l'entraînement.
        task_type: ``'multitask'`` pour produire les deux cibles nommées, sinon le masque seul.
        repeat: Si vrai, le dataset boucle indéfiniment.

    Returns:
        Un ``tf.data.Dataset`` par lots, avec préchargement (``prefetch``).

    Note:
        Seul le jeu d'entraînement doit être répété, car il est couplé à
        ``steps_per_epoch``. Répéter la validation ou le test les rend infinis : la
        validation et la boucle d'évaluation qui suit ``fit`` ne se terminent jamais.
    """
    image_paths = df['image_path'].tolist()
    mask_paths = df['mask_path'].tolist()
    labels = [class_to_idx.get(lbl, 0) for lbl in df['label'].tolist()]

    def gen():
        """Générateur paresseux : un triplet image / masque / label à la fois."""
        for image_path, mask_path, label in zip(image_paths, mask_paths, labels, strict=False):
            image = _read_image(image_path, image_size)
            mask = _read_mask(mask_path, image_size)
            if task_type == 'multitask':
                yield image, {'segmentation_output': mask, 'classification_output': np.array(label, dtype=np.int32)}
            else:
                yield image, mask

    if task_type == 'multitask':
        output_signature = (
            tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
            {
                'segmentation_output': tf.TensorSpec(shape=(image_size, image_size, 1), dtype=tf.float32),
                'classification_output': tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(image_size, image_size, 1), dtype=tf.float32),
        )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    # Seul le train est répété (couplé à steps_per_epoch). Répéter val/test les rend
    # infinis → validation et boucle d'éval post-fit ne se terminent jamais.
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_segmentation_datasets(manifest_path: str | Path, image_size: int, batch_size: int, validation_split: float = 0.2, seed: int = 42, task_type: str = 'multitask'):
    """Lit le manifeste et en tire les trois jeux, la liste des classes et la taille du train.

    Découpage : la colonne ``split`` du manifeste fait foi pour isoler le test. Si elle ne
    désigne aucune ligne de test, un découpage stratifié en fabrique un — mieux vaut un
    test tiré au sort qu'une évaluation faite sur les données d'entraînement. La validation
    est ensuite prélevée sur ce qui reste, également de façon stratifiée pour que les
    classes rares ne disparaissent pas d'un des jeux.

    Args:
        manifest_path: Chemin du ``manifest.csv`` produit par
            :func:`src.segmentation.datasets.manifest.build_manifest`.
        image_size: Côté du carré cible, en pixels.
        batch_size: Taille des lots.
        validation_split: Fraction prélevée pour la validation (et plancher de 0.15 pour le
            test de secours).
        seed: Graine des découpages, pour que deux exécutions donnent les mêmes jeux.
        task_type: ``'multitask'`` ou segmentation seule — voir :func:`_make_dataset`.

    Returns:
        Le quintuplet ``(train_ds, val_ds, test_ds, class_names, train_size)``, où
        ``class_names`` est la liste ordonnée des classes (elle fixe les indices) et
        ``train_size`` le nombre d'exemples d'entraînement, nécessaire pour calculer
        ``steps_per_epoch`` puisque le dataset d'entraînement est infini.

    Raises:
        FileNotFoundError: Le manifeste n'existe pas — la préparation des données n'a pas
            été lancée.
        ValueError: Le manifeste est vide, c'est-à-dire que la préparation s'est exécutée
            mais n'a apparié aucune image à un masque. C'est un échec silencieux classique :
            l'entraînement partirait sur zéro exemple sans cette vérification.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.stat().st_size == 0:
        raise ValueError(
            f"Manifest is empty: {manifest_path}. "
            "prepare_segmentation_dataset produced zero rows."
        )

    df = pd.read_csv(manifest_path)

    if df.empty:
        raise ValueError(f'No rows found in manifest {manifest_path}')
    labels = sorted([lbl for lbl in df['label'].dropna().unique().tolist() if lbl != 'unknown'])
    if not labels:
        labels = ['negative', 'positive']
    class_to_idx = {name: idx for idx, name in enumerate(labels)}

    train_df = df[df['split'] != 'test'].copy()
    test_df = df[df['split'] == 'test'].copy()
    if test_df.empty:
        train_df, test_df = train_test_split(train_df, test_size=max(validation_split, 0.15), random_state=seed, stratify=train_df['label'])
    train_df, val_df = train_test_split(train_df, test_size=validation_split, random_state=seed, stratify=train_df['label'])

    train_size = len(train_df)

    return (
        _make_dataset(train_df, image_size, batch_size, class_to_idx, task_type, repeat=True),
        _make_dataset(val_df, image_size, batch_size, class_to_idx, task_type, repeat=False),
        _make_dataset(test_df, image_size, batch_size, class_to_idx, task_type, repeat=False),
        labels,
        train_size,
    )
