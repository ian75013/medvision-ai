"""Jeux de données de classification **binaire** à partir d'un dossier d'images.

Alimente :mod:`src.training.train` (radiographies thoraciques). L'arborescence attendue est
celle que produisent les datasets Kaggle : un dossier par classe.

Deux dispositions sont acceptées, dans cet ordre :

1. ``train/``, ``val/`` et ``test/`` existent — on respecte la partition officielle du
   dataset, qui est la seule comparable à la littérature ;
2. sinon — un découpage validation est tiré au sort, et **le test est alors le même jeu que
   la validation**. Ce repli existe pour qu'un dossier d'images improvisé (démonstration,
   petit essai) fonctionne sans réorganisation, mais les métriques de test qui en sortent
   ne sont pas des métriques de test : elles mesurent des données déjà vues par la
   sélection de modèle. Ne pas publier de chiffre obtenu dans ce mode.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(
    dataset_dir: str | Path,
    image_size: int,
    batch_size: int,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Construit les jeux d'entraînement, de validation et de test, normalisés dans [0, 1].

    Seul le jeu d'entraînement est mélangé : mélanger la validation ou le test ne change
    aucun score mais rend les journaux incomparables d'une exécution à l'autre.

    Args:
        dataset_dir: Racine du dataset — soit contenant ``train``/``val``/``test``, soit
            contenant directement un dossier par classe (voir le repli décrit en tête de
            module).
        image_size: Côté du carré cible, en pixels.
        batch_size: Taille des lots.
        validation_split: Fraction de validation, utilisée **uniquement** dans le mode de
            repli.
        seed: Graine du mélange et du découpage.

    Returns:
        Le triplet ``(train_ds, val_ds, test_ds)``, chacun par lots, avec des images en
        ``float32`` dans [0, 1] et des labels binaires.

    Note:
        Dans le mode de repli, ``test_ds`` **est** ``val_ds``.
    """
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            label_mode="binary",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=False,
            label_mode="binary",
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=False,
            label_mode="binary",
        )
    else:
        full_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            label_mode="binary",
            validation_split=validation_split,
            subset="both",
        )
        train_ds, val_ds = full_ds
        test_ds = val_ds

    def prepare(ds: tf.data.Dataset) -> tf.data.Dataset:
        """Ramène les pixels de [0, 255] à [0, 1] et active le préchargement.

        Args:
            ds: Dataset brut sorti de ``image_dataset_from_directory``.

        Returns:
            Le dataset normalisé, avec recouvrement du calcul et de la lecture.
        """
        return ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return prepare(train_ds), prepare(val_ds), prepare(test_ds)
