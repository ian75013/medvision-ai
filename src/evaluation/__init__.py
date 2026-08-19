"""Mesure de la qualité des modèles : métriques, rapports, matrices de confusion.

Deux modules, selon la nature du problème :

* :mod:`src.evaluation.metrics` — classification **binaire** (radiographie thoracique),
  avec seuil, spécificité et aires sous les courbes ;
* :mod:`src.evaluation.metrics_multiclass` — classification **multi-classes** (IRM
  cérébrale), avec moyennes macro et pondérées.

Les métriques de segmentation vivent à part, dans :mod:`src.segmentation.metrics` : elles
comparent des masques, pas des étiquettes.

POURQUOI cette séparation entre entraînement et évaluation : les mêmes fonctions servent
après ``fit``, dans les tests, et pour re-mesurer un modèle déjà entraîné. Un chiffre publié
dans un rapport doit venir du même code que celui des tests, sans quoi les deux dérivent.
"""
