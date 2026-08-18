"""Types de base décrivant un exemple de dataset.

Volontairement minuscule : cette structure sert de vocabulaire commun aux découpeurs et
aux ``Dataset`` PyTorch, sans les coupler entre eux.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MRIExample:
    """Un exemple d'IRM : de qui, où, quoi, et dans quel jeu.

    Figé (``frozen=True``) à dessein : un exemple décrit une donnée déjà sur le disque.
    Le rendre modifiable ouvrirait la porte à un ``split`` changé en cours de route — c'est
    exactement la mécanique par laquelle un exemple de test se retrouve dans
    l'entraînement.

    Attributes:
        patient_id: Identifiant du patient. C'est **lui** qui doit gouverner le découpage,
            jamais le fichier : un même patient a souvent plusieurs images, et les répartir
            entre train et test fait fuiter de l'information (le modèle reconnaît le
            patient, pas la pathologie).
        path: Chemin du volume ou de l'image sur le disque.
        label: Étiquette entière de la classe.
        split: ``"train"``, ``"val"`` ou ``"test"``.
    """

    patient_id: str
    path: Path
    label: int
    split: str
