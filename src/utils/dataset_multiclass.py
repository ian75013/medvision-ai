"""Jeux de données de classification **multi-classes** à partir d'un dataset partitionné.

Alimente :mod:`src.training.train_brain_mri`. À la différence de :mod:`src.utils.dataset`,
ce module **exige** que le dataset soit déjà partitionné à la source — ``Training/`` et
``Testing/`` chez Kaggle — et lève si ce n'est pas le cas plutôt que de se rabattre sur un
découpage improvisé : sur un problème à quatre classes, un test tiré au sort n'est pas
comparable aux résultats publiés sur ce même dataset.

La validation est prélevée sur ``Training/``, jamais sur ``Testing/``, qui doit rester
intouché jusqu'à la mesure finale.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _normalize(images, labels):
    """Ramène les pixels de [0, 255] à [0, 1], sans toucher aux labels.

    Args:
        images: Lot d'images en entiers.
        labels: Lot de labels entiers, transmis tel quel.

    Returns:
        Le couple ``(images normalisées, labels)``.
    """
    return tf.cast(images, tf.float32) / 255.0, labels


def build_multiclass_datasets(
    dataset_dir: str | Path,
    image_size: int,
    batch_size: int,
    validation_split: float = 0.15,
    seed: int = 42,
    training_subdir: str = "Training",
    testing_subdir: str = "Testing",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Construit les trois jeux multi-classes et la liste ordonnée des classes.

    Entraînement et validation lisent le **même** dossier avec la **même** graine et le
    même ``validation_split`` : c'est ce qui garantit que Keras les découpe sans
    recouvrement. Modifier la graine d'un seul des deux appels ferait fuiter des exemples
    d'entraînement dans la validation, sans qu'aucune erreur ne soit levée.

    Args:
        dataset_dir: Racine du dataset extrait.
        image_size: Côté du carré cible, en pixels.
        batch_size: Taille des lots.
        validation_split: Fraction de ``Training/`` réservée à la validation.
        seed: Graine du mélange et du découpage — doit rester identique entre les deux
            sous-ensembles.
        training_subdir: Nom du dossier d'entraînement (``Training`` chez Kaggle).
        testing_subdir: Nom du dossier de test (``Testing`` chez Kaggle).

    Returns:
        Le quadruplet ``(train_ds, val_ds, test_ds, class_names)``. ``class_names`` est
        l'ordre alphabétique des dossiers, et **c'est lui qui fixe les indices** de sortie
        du modèle : le relire depuis ce dataset est la seule façon sûre d'interpréter une
        prédiction. Renommer un dossier de classe change silencieusement cet ordre.

    Raises:
        FileNotFoundError: Un des deux sous-dossiers manque — typiquement une extraction
            Kaggle incomplète ou un ``dataset_dir`` pointant un niveau trop haut.
    """
    dataset_dir = Path(dataset_dir)
    train_root = dataset_dir / training_subdir
    test_root = dataset_dir / testing_subdir

    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Expected `{train_root}` and `{test_root}` to exist. Check dataset_dir and Kaggle extraction."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_root,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_root,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        label_mode="int",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_root,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )

    class_names = train_ds.class_names

    train_ds = train_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names
