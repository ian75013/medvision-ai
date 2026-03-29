#!/bin/bash
set -e

# 1. Vérifie que kaggle CLI est installé
if ! command -v kaggle &> /dev/null; then
    echo "Erreur : kaggle CLI n'est pas installé. Installe-le avec : pip install kaggle"
    exit 1
fi

# 2. Vérifie la présence de la clé API
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Erreur : clé API Kaggle manquante. Va sur https://www.kaggle.com/settings pour la générer et place-la dans ~/.kaggle/kaggle.json"
    exit 1
fi

# 3. Crée le dossier de destination
DEST="data/raw/chest_xray_segmentation"
mkdir -p "$DEST"

# 4. Télécharge le dataset (si pas déjà fait)
if [ ! -f "$DEST/chest-xray-masks-and-labels.zip" ]; then
    echo "Téléchargement du dataset depuis Kaggle..."
    kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels -p "$DEST"
else
    echo "Archive déjà présente, on saute le téléchargement."
fi

# 5. Dézippe (si pas déjà fait)
if [ ! -d "$DEST/Lung Segmentation" ]; then
    echo "Décompression de l'archive..."
    unzip -q "$DEST/chest-xray-masks-and-labels.zip" -d "$DEST"
else
    echo "Dossier déjà dézippé."
fi

if [ -d "$DEST/Lung Segmentation" ]; then
    if [ "$(ls -A "$DEST/Lung Segmentation" 2>/dev/null)" ]; then
        echo "Déplacement des fichiers..."
        mv "$DEST/Lung Segmentation"/* "$DEST"/ 2>/dev/null || true
        mv "$DEST/Lung Segmentation"/.[!.]* "$DEST"/ 2>/dev/null || true  # déplace aussi les fichiers cachés
    else
        echo "Dossier 'Lung Segmentation' déjà vide, rien à déplacer."
    fi
    rm -rf "$DEST/Lung Segmentation"
fi

echo "✅ Dataset chest_xray_segmentation prêt dans $DEST"