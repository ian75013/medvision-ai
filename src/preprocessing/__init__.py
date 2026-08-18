"""Pré-traitement des images : mise en forme avant le modèle, augmentation à l'entraînement.

* :mod:`src.preprocessing.image_loader` — une image 2D sur disque → tableau normalisé prêt
  pour l'inférence ;
* :mod:`src.preprocessing.brain_mri_2d` — un volume IRM 3D → une poignée de coupes 2D
  normalisées ;
* :mod:`src.preprocessing.augmentation` — les couches d'augmentation Keras, réservées à
  l'entraînement.

POURQUOI il est vital que ce paquet reste la seule définition du pré-traitement : le modèle
apprend sur des images préparées d'une certaine façon. Si l'inférence les prépare
différemment — un redimensionnement dans un autre ordre, une normalisation oubliée — le
modèle ne se plaint pas, il se contente de prédire moins bien, et rien dans les journaux ne
le signale. C'est le mode de panne le plus coûteux du projet.
"""
