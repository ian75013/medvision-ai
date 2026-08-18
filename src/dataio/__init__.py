"""Lecture des formats de fichiers médicaux volumétriques.

Contient :mod:`src.dataio.nifti_loader`, seul point du projet où un volume 3D est lu depuis
le disque. Les images 2D ordinaires, elles, passent par
:mod:`src.preprocessing.image_loader`.

POURQUOI isoler la lecture : NIfTI dépend de ``nibabel``, une bibliothèque que ni l'API ni
l'image de production n'installent. Cantonner l'import à ce module permet au reste du code
de s'exécuter sans elle, et de n'échouer — avec un message clair — que si l'on tente
réellement d'ouvrir un volume.
"""
