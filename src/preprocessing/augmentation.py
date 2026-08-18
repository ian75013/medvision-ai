"""Augmentation de données Keras — active à l'entraînement seulement.

POURQUOI des amplitudes aussi faibles : une image médicale a une orientation et une échelle
anatomiques. Un miroir horizontal reste plausible (l'anatomie thoracique est quasi
symétrique), mais une rotation franche ou un zoom violent fabriquent une image qu'aucun
appareil ne produirait, et le modèle apprend alors à reconnaître des cas qui n'existent
pas. On augmente pour couvrir les variations réelles de positionnement du patient — pas
au-delà.
"""

from __future__ import annotations

import tensorflow as tf


def build_augmentation() -> tf.keras.Sequential:
    """Construit le bloc d'augmentation à insérer en tête du modèle.

    Quatre transformations, toutes de faible amplitude : miroir horizontal, rotation de
    ±3 %, zoom de ±10 % et variation de contraste de ±10 %.

    Ces couches Keras sont **inactives en inférence** : elles ne s'appliquent que lorsque
    ``training=True``. C'est ce qui permet de les intégrer au modèle plutôt qu'au pipeline
    de données — l'augmentation part alors avec le modèle sauvegardé, et l'inférence ne
    risque pas de l'appliquer par erreur.

    Returns:
        Un ``tf.keras.Sequential`` nommé ``augmentation``.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
