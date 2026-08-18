"""Construction des jeux de données de segmentation à partir des dossiers bruts.

Contient :mod:`src.segmentation.datasets.manifest`, qui parcourt un dossier téléchargé
(Kaggle, Montgomery, ...) et en tire le ``manifest.csv`` que le reste du pipeline consomme.

POURQUOI cette étape existe : chaque dataset médical range ses images et ses masques
autrement — parfois dans deux dossiers frères, parfois côte à côte avec un suffixe
``_mask``, parfois avec le diagnostic dans un rapport texte à part. Le manifeste est le
point où toute cette diversité est ramenée à un tableau unique ; tout ce qui suit
(``data.py``, l'entraînement) ne connaît plus que ce tableau.
"""
