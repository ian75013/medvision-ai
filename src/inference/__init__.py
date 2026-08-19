"""Inférence hors production : rejouer un modèle entraîné, dans son format d'origine.

* :mod:`src.inference.predict` — radiographie thoracique, modèle Keras binaire ;
* :mod:`src.inference.predict_brain_mri` — IRM cérébrale, modèle Keras multi-classes ;
* :mod:`src.inference.predict_classifier` — volume NIfTI, modèle PyTorch, avec agrégation
  des coupes.

ATTENTION — ce n'est **pas** le chemin de la production. Depuis la migration ONNX (juin
2026), l'application sert ses prédictions via ONNX Runtime, dans
:mod:`src.registry.model_registry` et ``streamlit_app.py`` ; ni TensorFlow ni PyTorch ne
sont installés dans l'image déployée. Ce paquet suppose ``requirements-train.txt`` et n'est
utilisable que sur la machine d'entraînement.

À quoi il sert, alors : à vérifier un modèle fraîchement entraîné avant conversion, et à
arbitrer quand l'API renvoie un résultat douteux. Si le modèle d'origine donne la bonne
réponse et pas l'ONNX, le défaut est dans la conversion ou dans le pré-traitement du
service — pas dans le modèle.
"""
