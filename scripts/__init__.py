"""Outils exécutables : conversion de modèles, génération de données de démonstration.

Ce paquet n'est pas importé par l'application. Il rassemble ce qu'on lance à la main ou
depuis un pipeline :

* ``convert_to_onnx.py`` — convertit les modèles ``.keras`` / ``.pt`` en ONNX sur la machine
  d'entraînement. C'est l'étape sans laquelle un modèle fraîchement entraîné n'est pas
  servable : la production ne charge que de l'ONNX.
* ``generate_sample_images.py`` — fabrique des PNG synthétiques pour que l'UI soit
  navigable sans télécharger de dataset.
* ``generate_demo_brain_mri_dataset.py`` — fabrique des volumes 3D synthétiques et leur
  découpage, pour dérouler la chaîne PyTorch de bout en bout.

Les deux générateurs produisent du **bruit avec des formes**, pas des images médicales :
ils servent à vérifier que la tuyauterie fonctionne, jamais à mesurer une qualité de
modèle.
"""
