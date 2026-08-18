"""Briques transverses : config, journalisation, chemins, graines, jeux de données Keras.

Ce paquet ne connaît rien du domaine médical. Il ne contient que ce dont tous les autres
modules ont besoin et qui n'appartient à aucun d'eux en particulier :

* :mod:`src.utils.config` — lecture des configs YAML ;
* :mod:`src.utils.logging` — un logger configuré, sans doublon de sortie ;
* :mod:`src.utils.paths` — création de dossiers d'artefacts ;
* :mod:`src.utils.seed` — reproductibilité côté PyTorch ;
* :mod:`src.utils.dataset` et :mod:`src.utils.dataset_multiclass` — construction des
  ``tf.data.Dataset`` de classification, binaire et multi-classes.

Règle de ce paquet : rien de ce qui vit ici ne doit importer ``src.training``,
``src.segmentation`` ni ``src.api``. Une dépendance dans ce sens créerait un cycle et,
surtout, ferait remonter TensorFlow ou FastAPI dans des modules censés rester légers.
"""
