"""Acquisition des datasets : téléchargement depuis Kaggle, puis mise en forme.

Un script par dataset, chacun exécutable en module et appelé par un stage DVC :

* :mod:`src.data.download_dataset` — radiographies thoraciques (Kermany) ;
* :mod:`src.data.download_brain_mri_dataset` — IRM cérébrales multi-classes ;
* :mod:`src.data.download_segmentation_dataset` — dataset de segmentation ;
* :mod:`src.data.prepare_segmentation_dataset` — construit le manifeste image/masque.

POURQUOI les données ne sont pas dans le dépôt : elles pèsent plusieurs gigaoctets et leurs
licences ne permettent pas toujours la redistribution. Le dépôt versionne donc la
**procédure** d'obtention ; DVC versionne l'empreinte du résultat.

Ces scripts sont **idempotents** : relancés sur un dataset déjà présent, ils ne
retéléchargent rien et se contentent de le signaler. C'est ce qui permet de les mettre dans
un pipeline DVC ou un entrypoint de conteneur sans crainte.
"""
