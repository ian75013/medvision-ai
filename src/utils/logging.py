"""Fabrique de loggers : un format unique, et jamais de message en double.

POURQUOI passer par ce module plutôt que par ``logging.getLogger`` directement : la
configuration des handlers est faite **une seule fois par logger**. Sans cette garde, un
module importé deux fois — ce qui arrive avec Streamlit, qui réexécute le script à chaque
interaction — empile les handlers et chaque ligne de journal apparaît deux, puis trois, puis
dix fois.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Renvoie un logger nommé, configuré au premier appel seulement.

    Le format retenu — horodatage, niveau, nom du module, message — est celui qui reste
    lisible dans ``kubectl logs``, où les lignes de plusieurs pods se mélangent : le nom du
    module dit d'où vient chaque ligne.

    Args:
        name: Nom du logger, par convention ``__name__`` de l'appelant.

    Returns:
        Le logger, avec un handler console au niveau INFO. Les appels suivants pour le même
        nom renvoient le même objet sans rien reconfigurer.

    Example:
        >>> LOGGER = get_logger(__name__)
        >>> LOGGER.info("Modèle chargé depuis %s", chemin)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
