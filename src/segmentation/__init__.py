"""Segmentation d'images médicales : données, modèle U-Net, entraînement, inférence.

Ce paquet couvre la chaîne complète de la tâche « segmentation » du projet, celle qui
répond à *où* se trouve la lésion, là où le paquet ``src/inference`` répond seulement à
*est-ce qu'il y en a une*.

Le parcours, dans l'ordre où on l'exécute :

1. :mod:`src.segmentation.datasets.manifest` — apparie chaque image à son masque et écrit
   un ``manifest.csv`` (une ligne = image + masque + label + split).
2. :mod:`src.segmentation.data` — transforme ce manifeste en ``tf.data.Dataset`` prêts à
   entraîner (train / val / test).
3. :mod:`src.segmentation.models.unet` — construit le U-Net, simple ou multitâche
   (segmentation + classification sur le même encodeur).
4. :mod:`src.segmentation.train_segmentation` — entraîne, évalue, journalise dans MLflow.
5. :mod:`src.segmentation.predict_segmentation` — rejoue un modèle entraîné sur une image.
6. :mod:`src.segmentation.metrics` et :mod:`src.segmentation.overlays` — mesurent (Dice,
   IoU) et donnent à voir (masque superposé à l'image).

POURQUOI un paquet séparé de ``src/training`` : la segmentation a ses propres métriques
(Dice, IoU — l'exactitude pixel par pixel ne veut rien dire sur des masques creux), son
propre format de données (image *et* masque appariés) et sa propre sortie (une image, pas
un score). Les mélanger avec la classification rendait les deux illisibles.
"""
