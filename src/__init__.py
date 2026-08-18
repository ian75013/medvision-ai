"""MedVision-AI — imagerie médicale : entraînement, évaluation et service de modèles.

Carte du paquet, dans l'ordre où les données le traversent :

* :mod:`src.data` — téléchargement et préparation des datasets bruts ;
* :mod:`src.datasets` — description, découpage par patient, indexation pour l'UI ;
* :mod:`src.preprocessing` — mise en forme des images et des volumes ;
* :mod:`src.models` — architectures de classification (dos pré-entraînés et modèle simple) ;
* :mod:`src.training` — entraînement des classifieurs, TensorFlow et PyTorch ;
* :mod:`src.segmentation` — la chaîne complète de segmentation, données comprises ;
* :mod:`src.evaluation` — métriques, rapports, matrices de confusion ;
* :mod:`src.registry` — catalogue des modèles ONNX servis, et chargement ;
* :mod:`src.inference` — rejeu d'un modèle dans son format d'origine (hors production) ;
* :mod:`src.api` — l'API FastAPI ; ``streamlit_app.py``, à la racine, est l'autre interface.

Deux frontières à garder en tête pour ne pas s'y perdre des mois plus tard :

1. **Entraînement et production ne partagent pas leurs dépendances.** TensorFlow et PyTorch
   ne vivent que sur la machine d'entraînement (``requirements-train.txt``) ; l'image
   déployée ne contient qu'ONNX Runtime. Tout module importé par l'API doit le rester.
2. **Les artefacts ne sont pas dans l'image.** Modèles et rapports sont versionnés par DVC
   sur S3 et récupérés au démarrage du pod. Le code doit donc supporter leur absence —
   c'est ce que fait :mod:`src.registry.model_registry` avec son drapeau ``available``.
"""
