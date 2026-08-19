"""Transfert progressif : chauffer la tête, puis dégeler le haut du dos pré-entraîné.

POURQUOI ne pas entraîner tout le réseau d'emblée : au premier pas, la tête de
classification est initialisée au hasard et produit d'énormes gradients. Rétropropagés
dans un dos pré-entraîné, ils effacent en quelques lots les représentations apprises sur
ImageNet — c'est le « catastrophic forgetting », et le modèle finit moins bon qu'un modèle
entraîné de zéro.

La parade, en deux temps :

1. **Chauffe** — dos gelé, seule la tête apprend, à un pas d'apprentissage normal (1e-3).
   La tête devient raisonnable sans rien casser derrière elle.
2. **Affinage** — on dégèle les N dernières couches du dos et on repart à un pas beaucoup
   plus petit (3e-5 au plus). Les couches basses, qui codent bords et textures, restent
   gelées : elles sont valables sur une radiographie comme sur une photo.

Le nombre de couches dégelées dépend de l'architecture, il est déclaré dans
:data:`src.models.backbones.TF_BACKBONES`.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from src.models.backbones import TF_BACKBONES, build_transfer_model, set_backbone_trainable


@dataclass
class FineTuneConfig:
    """Réglages d'un entraînement par transfert progressif.

    Attributes:
        warmup_epochs: Époques de chauffe, dos gelé. Zéro saute la phase.
        finetune_epochs: Époques d'affinage, dos partiellement dégelé. Zéro saute la phase.
        warmup_lr: Pas d'apprentissage de la chauffe (typiquement 1e-3).
        finetune_lr: Pas d'apprentissage de l'affinage. Doit rester bien plus petit que
            ``warmup_lr`` — c'est tout l'intérêt de la manœuvre.
        backbone_name: Clé du dos dans :data:`src.models.backbones.TF_BACKBONES`.
        unfreeze_layers: Nombre de couches terminales du dos rendues entraînables.
        dropout: Taux de dropout de la tête ; ``None`` laisse la valeur par défaut du dos.
        label_smoothing: Lissage des étiquettes, de 0 (aucun) à ~0.1. Il évite qu'un modèle
            sûr de lui à 100 % sur un dataset médical bruité s'arrête d'apprendre.
    """

    warmup_epochs: int
    finetune_epochs: int
    warmup_lr: float
    finetune_lr: float
    backbone_name: str
    unfreeze_layers: int
    dropout: float | None = None
    label_smoothing: float = 0.0


def default_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Rappels Keras communs à toutes les phases d'entraînement.

    ``EarlyStopping`` (patience 4, restauration des meilleurs poids) coupe dès que la
    validation cesse de progresser et **rend les poids du meilleur passage**, pas ceux de
    la dernière époque. ``ReduceLROnPlateau`` (facteur 0.2, patience 2) divise le pas par
    cinq sur un plateau, ce qui débloque souvent une descente devenue trop grossière.

    Returns:
        Une liste fraîche de rappels — à ne pas partager entre deux ``fit`` concurrents,
        les rappels portent un état.
    """
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
    ]


def train_with_progressive_finetuning(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    image_size: int,
    num_classes: int,
    cfg: FineTuneConfig,
    class_weight: dict[int, float] | None = None,
) -> tuple[tf.keras.Model, dict[str, list[float]]]:
    """Entraîne un modèle de transfert en deux phases : chauffe puis affinage.

    L'affinage reprend la numérotation d'époques là où la chauffe s'est arrêtée
    (``initial_epoch``), pour que les courbes MLflow forment une seule série continue et
    non deux courbes superposées démarrant à zéro.

    Args:
        train_ds: Jeu d'entraînement, déjà par lots.
        val_ds: Jeu de validation, qui pilote les deux rappels.
        image_size: Côté des images en entrée, en pixels.
        num_classes: Nombre de classes de sortie.
        cfg: Réglages du transfert — voir :class:`FineTuneConfig`.
        class_weight: Poids par classe, pour compenser un dataset déséquilibré (les
            datasets médicaux le sont presque toujours). ``None`` laisse toutes les classes
            à poids égal.

    Returns:
        Le couple ``(modèle entraîné, historique)``, où l'historique concatène les deux
        phases par nom de métrique.

    Note:
        La recompilation avant l'affinage passe ``metrics=["accuracy"]`` en clair plutôt
        que de réutiliser ``model.metrics``. Réutiliser les objets ``Metric`` déjà
        construits rend le conteneur de métriques non sérialisable, et ``model.save()``
        échoue ensuite sur un ``NotImplementedError`` de ``compile_utils.get_config``
        (Keras 3). Le symptôme apparaît des heures après, au moment de sauvegarder.
    """
    model, base_model = build_transfer_model(
        backbone_name=cfg.backbone_name,
        image_size=image_size,
        num_classes=num_classes,
        learning_rate=cfg.warmup_lr,
        trainable_backbone=False,
        dropout=cfg.dropout,
        label_smoothing=cfg.label_smoothing,
    )

    history: dict[str, list[float]] = {}
    callbacks = default_callbacks()

    if cfg.warmup_epochs > 0:
        warmup_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.warmup_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        ).history
        for k, v in warmup_history.items():
            history.setdefault(k, []).extend(v)

    set_backbone_trainable(base_model, cfg.unfreeze_layers)
    # metrics=["accuracy"] (et non model.metrics) : réutiliser les objets Metric déjà
    # construits rend le conteneur de métriques non-sérialisable → model.save() plante
    # (Keras 3 : compile_utils get_config NotImplementedError).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.finetune_lr),
        loss=model.loss,
        metrics=["accuracy"],
    )

    if cfg.finetune_epochs > 0:
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.warmup_epochs + cfg.finetune_epochs,
            initial_epoch=cfg.warmup_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        ).history
        for k, v in fine_history.items():
            history.setdefault(k, []).extend(v)

    return model, history


def infer_unfreeze_layers(backbone_name: str, override: int | None = None) -> int:
    """Combien de couches terminales du dos rendre entraînables à l'affinage.

    Chaque architecture a sa profondeur de bloc : dégeler 20 couches n'a pas le même sens
    sur un DenseNet que sur un EfficientNet. La valeur de référence est donc déclarée avec
    le dos, et la config du projet peut la remplacer ponctuellement.

    Args:
        backbone_name: Clé du dos dans :data:`src.models.backbones.TF_BACKBONES`.
        override: Valeur imposée par la config, ou ``None`` pour prendre celle du dos.

    Returns:
        Le nombre de couches à dégeler.

    Raises:
        KeyError: Le dos est inconnu et aucune valeur n'a été imposée.
    """
    if override is not None:
        return override
    return TF_BACKBONES[backbone_name].unfreeze_layers
