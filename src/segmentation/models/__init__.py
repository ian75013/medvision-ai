"""Architectures de segmentation — expose les constructeurs de U-Net.

``build_unet`` produit un U-Net classique (une seule sortie : le masque) ;
``build_multitask_unet`` y greffe une tête de classification qui partage l'encodeur, de
sorte qu'un même passage avant répond à la fois « où » et « quoi ».
"""

from src.segmentation.models.unet import build_multitask_unet, build_unet

__all__ = ["build_unet", "build_multitask_unet"]
