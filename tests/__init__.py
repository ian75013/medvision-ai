"""Suite de tests de MedVision-AI, en deux étages.

* ``tests/smoke/`` — les tests **sans TensorFlow** : ils tournent en quelques secondes et
  constituent la barrière de la CI (``scripts/ci_local.sh``). Tout ce qui vit ici doit
  rester importable dans l'environnement de production, c'est-à-dire sans TF ni PyTorch ;
* ``tests/`` (racine) — les tests qui **exigent** TensorFlow : construction d'architectures,
  métriques d'évaluation. Ils ne sont collectés que par le job CI « test-tf ».

POURQUOI cette séparation : l'image déployée ne contient qu'ONNX Runtime. Une suite qui
importerait TensorFlow au démarrage ne dirait plus rien de ce qui tourne réellement en
production, et ne serait plus exécutable dans la CI rapide. La frontière est donc un
invariant du projet, pas une commodité.
"""
