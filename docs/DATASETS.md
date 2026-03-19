# Datasets

## 1. Dataset historique

Le dépôt conserve une configuration historique `configs/config.yaml` orientée **chest X-ray pneumonia**.

## 2. Dataset recommandé dans ce zip

Le workflow le plus cohérent est aujourd'hui le dataset Kaggle :

```text
masoudnickparvar/brain-tumor-mri-dataset
```

### Structure attendue

```text
data/raw/brain_tumor_mri/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 3. Démo synthétique

`brain_mri_2d_demo.yaml` et certains fichiers de démo correspondent à un jeu synthétique pour valider l'architecture logicielle, pas à une source clinique réelle.

## 4. Futur recommandé

Pour un projet plus sérieux de vision médicale MRI, la suite logique serait :
- migration vers des volumes NIfTI ;
- pipeline 3D ou 2.5D ;
- segmentation + classification ;
- éventuellement BraTS / MONAI.
