"""Tests de fumée : rapides, sans TensorFlow ni PyTorch — la barrière de la CI.

Règle d'admission dans ce dossier : le test doit s'exécuter avec les seules dépendances de
l'image de production (``requirements/base.txt`` — FastAPI, ONNX Runtime, NumPy, Pillow,
pandas). Un test qui a besoin d'entraîner ou de construire un modèle TensorFlow va à la
racine de ``tests/``.

Ce qu'ils couvrent : le registre de modèles, les routes de l'API (historiques et v2), le
navigateur d'images, la logique de segmentation pure et la veille DVC — c'est-à-dire tout
ce qui peut casser en production.
"""
