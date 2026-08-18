"""Architectures de classification : dos pré-entraînés et petit modèle de référence.

* :mod:`src.models.backbones` — le catalogue :data:`TF_BACKBONES` (DenseNet, ResNet,
  EfficientNet…), la construction d'un modèle de transfert et le dégel du dos ;
* :mod:`src.models.baseline_model` — un CNN simple entraîné de zéro.

POURQUOI garder un modèle « baseline » alors que le transfert fait mieux : il donne le
plancher. Un dos pré-entraîné qui ne bat pas un petit CNN entraîné de zéro signale un
problème — données trop peu nombreuses, pré-traitement incohérent, ou dos mal dégelé — que
les seules courbes d'entraînement ne révèlent pas.

Les architectures de segmentation sont ailleurs : :mod:`src.segmentation.models`.
"""

from src.models.backbones import TF_BACKBONES, build_transfer_model, set_backbone_trainable
from src.models.baseline_model import build_baseline_model

__all__ = ["TF_BACKBONES", "build_transfer_model", "set_backbone_trainable", "build_baseline_model"]
