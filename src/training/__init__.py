"""Entraînement des modèles de classification : points d'entrée et briques partagées.

Un fichier par tâche, parce que chacune a son dataset, ses classes et ses artefacts :

* :mod:`src.training.train` — radiographies thoraciques (2 classes, transfert progressif) ;
* :mod:`src.training.train_classifier` — variante générique pilotée par la config seule ;
* :mod:`src.training.train_brain_mri` — IRM cérébrales, 4 classes, en TensorFlow ;
* :mod:`src.training.train_brain_mri_torch` — la même tâche en PyTorch.

Et deux briques communes :

* :mod:`src.training.transfer_utils` — le transfert progressif (chauffe tête gelée, puis
  dégel partiel du dos), utilisé par tous les entraînements TensorFlow ;
* :mod:`src.training.trainer` — la boucle d'époque PyTorch, partagée par les variantes Torch.

POURQUOI deux frameworks cohabitent : les modèles ont d'abord été écrits en TensorFlow,
puis la variante PyTorch a été ajoutée pour l'IRM. Depuis la migration ONNX (juin 2026),
**aucun des deux n'est requis en production** — l'inférence passe par ONNX Runtime. TF et
Torch ne sont installés que sur la machine d'entraînement, via ``requirements-train.txt``.
La segmentation, elle, vit dans :mod:`src.segmentation`.
"""
