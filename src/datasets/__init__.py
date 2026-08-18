"""Description et découpage des jeux de données, côté PyTorch et côté navigation UI.

* :mod:`src.datasets.base` — les types décrivant un exemple ;
* :mod:`src.datasets.splitters` — découpage **par patient** d'un CSV de métadonnées ;
* :mod:`src.datasets.brats_2d_dataset` — ``Dataset`` PyTorch de coupes issues de volumes ;
* :mod:`src.datasets.sample_browser` — indexation des images pour le navigateur de l'UI.

À ne pas confondre avec :mod:`src.data`, qui **télécharge et prépare** les datasets bruts :
ici, on suppose les fichiers déjà là et on décrit comment on les lit.
"""
