# Known gaps and technical debt

## 1. Deux lignes de produit cohabitent

Le dépôt contient à la fois :
- un flux historique **chest X-ray** ;
- un flux récent **brain MRI**.

Cela rend la lecture moins immédiate pour un nouveau contributeur.

## 2. Modules de modèles historiques manquants dans ce zip

Les scripts historiques `src/training/train.py` et `src/training/train_brain_mri.py` importent :
- `src.models.baseline_model`
- `src.models.optimized_model`

Or le dossier `src/models/` n'est pas présent dans cette archive. Cela signifie que :
- le dépôt n'est pas entièrement auto-cohérent dans l'état du zip ;
- la partie Brain MRI est conceptuellement la plus pertinente, mais nécessiterait soit la restauration de `src/models/`, soit une refactorisation pour utiliser uniquement les modèles définis localement.

## 3. Serving pas encore aligné avec Brain MRI

- `src/api/main.py` charge un modèle historique `optimized_model.keras`
- `streamlit_app.py` parle encore de chest X-ray / pneumonia

Pour une cohérence produit, il faudrait :
- une API Brain MRI dédiée ;
- une UI Brain MRI dédiée ;
- une convention de nommage des modèles plus claire.

## 4. DVC minimal

Le pipeline DVC actuel est utile mais reste minimal. Une version plus robuste devrait séparer :
- `download`
- `prepare`
- `train`
- `evaluate`
- `package`

## 5. MRI encore 2D Kaggle

Le dépôt montre une bonne base de travail, mais il ne s'agit pas encore d'une stack IRM volumique clinique complète.

## 6. Recommandation prioritaire

Si tu veux rendre le dépôt plus propre rapidement :
1. recréer `src/models/` ou adapter les scripts pour supprimer ces imports ;
2. créer une API/UI Brain MRI ;
3. isoler le flux chest X-ray dans un sous-dossier ou une branche ;
4. enrichir le pipeline DVC.
