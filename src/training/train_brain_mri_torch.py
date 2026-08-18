"""Classification d'IRM cérébrales en PyTorch — variante torchvision de ``train_brain_mri``.

POURQUOI une seconde implémentation de la même tâche : elle donne accès à des
architectures que Keras n'expose pas (notamment les Swin Transformers) et sert de contrôle
croisé — deux piles indépendantes qui convergent vers les mêmes scores rassurent bien plus
qu'une seule. Les deux produisent, après conversion ONNX, un modèle servable de la même
façon.

Usage:
    python -m src.training.train_brain_mri_torch --config configs/brain_mri.yaml
    python -m src.training.train_brain_mri_torch --config configs/brain_mri.yaml --model swin_v2_s_torch

Deux points de vigilance, propres à cette voie :

* **Le modèle retenu est celui du meilleur F1 macro de validation**, conservé en mémoire
  (``deepcopy``) et rechargé avant le test. PyTorch n'a pas d'équivalent de
  ``restore_best_weights`` : sans cette copie, on évaluerait la dernière époque, qui est
  rarement la meilleure.
* **Le F1 macro, et non l'exactitude, arbitre**. Sur quatre classes déséquilibrées,
  l'exactitude suit la classe majoritaire ; le F1 macro traite les quatre à égalité.

Le point de contrôle sauvegardé embarque les classes, le nom du modèle et la taille
d'image en plus des poids : un ``.pt`` nu ne dit pas dans quel ordre lire ses sorties, et
c'est exactement ce dont on a besoin des mois plus tard pour le convertir en ONNX.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.utils.config import load_config
from src.utils.paths import ensure_dir


def set_seed(seed: int) -> None:
    """Fixe la graine de Python, NumPy et PyTorch (CPU et tous les GPU).

    Args:
        seed: Graine commune.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Déclare et lit les arguments de la ligne de commande.

    Returns:
        Les arguments analysés : ``config`` (YAML, obligatoire), ``model`` parmi les trois
        dos torchvision câblés dans :func:`build_model`, et ``epochs`` (facultatif).
    """
    parser = argparse.ArgumentParser(description="PyTorch transfer learning for brain MRI classification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="densenet121_torch", choices=["densenet121_torch", "resnet50_torch", "swin_v2_s_torch"])
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def build_transforms(image_size: int):
    """Compose les transformations d'entrée : une pour l'entraînement, une pour l'évaluation.

    L'augmentation (miroir, rotation ≤ 10°, légère variation de luminosité et de contraste)
    ne s'applique **qu'à l'entraînement**. L'appliquer aussi en évaluation rendrait les
    scores bruités et non reproductibles. Les amplitudes restent faibles : une IRM
    cérébrale a une orientation anatomique, et une rotation franche fabriquerait une image
    qui n'existe pas en clinique.

    La normalisation utilise les moyennes et écarts-types d'ImageNet, parce que les dos sont
    pré-entraînés dessus — s'en écarter décale les activations dès la première couche.

    Args:
        image_size: Côté du carré cible, en pixels.

    Returns:
        Le couple ``(transformations d'entraînement, transformations d'évaluation)``.
    """
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf


def build_model(model_name: str, num_classes: int):
    """Charge un dos torchvision pré-entraîné, remplace sa tête, et gèle tout le reste.

    Chaque architecture nomme sa tête différemment — ``classifier`` chez DenseNet, ``fc``
    chez ResNet, ``head`` chez Swin — d'où les trois branches. La tête d'origine prédit les
    1000 classes d'ImageNet ; on la remplace par une couche linéaire aux dimensions du
    problème, seule partie initialisée au hasard.

    Le gel est ensuite général puis relâché sur la seule tête : c'est l'état de départ de
    la phase de chauffe (voir la note de :mod:`src.training.transfer_utils`, le raisonnement
    est le même ici).

    Args:
        model_name: ``densenet121_torch``, ``resnet50_torch`` ou ``swin_v2_s_torch``.
            Toute autre valeur retombe sur le Swin — les choix sont contraints en amont par
            ``argparse``.
        num_classes: Nombre de classes de sortie.

    Returns:
        Le modèle prêt pour la chauffe : dos gelé, tête entraînable.
    """
    if model_name == "densenet121_torch":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50_torch":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        weights = models.Swin_V2_S_Weights.DEFAULT
        model = models.swin_v2_s(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(key in name for key in ["classifier", "fc", "head"]):
            param.requires_grad = True
    return model


def unfreeze_last_layers(model: nn.Module, model_name: str, unfreeze_blocks: int = 2):
    """Rend entraînables les derniers blocs du dos, pour la phase d'affinage.

    On dégèle par **bloc** et non par couche : dans ces architectures, un bloc est l'unité
    cohérente (convolutions, normalisation et connexion résiduelle), et n'en dégeler qu'une
    partie déséquilibre l'apprentissage.

    Chez ResNet, on écarte le dernier enfant du module — c'est la tête, déjà dégelée par
    :func:`build_model`.

    Args:
        model: Modèle renvoyé par :func:`build_model`.
        model_name: Nom de l'architecture, qui détermine où trouver les blocs.
        unfreeze_blocks: Nombre de blocs terminaux à dégeler. Au-delà de 3, sur un petit
            dataset médical, le sur-apprentissage l'emporte.
    """
    if model_name == "densenet121_torch":
        modules = list(model.features.children())
    elif model_name == "resnet50_torch":
        modules = list(model.children())[:-1]
    else:
        modules = list(model.features.children())
    for module in modules[-unfreeze_blocks:]:
        for param in module.parameters():
            param.requires_grad = True


def evaluate(model, loader, device):
    """Évalue le modèle sur un chargeur et renvoie les quatre métriques macro.

    ``model.eval()`` et ``torch.no_grad()`` sont tous deux nécessaires et ne font pas la
    même chose : le premier bascule dropout et batch-norm en mode inférence, le second
    empêche la construction du graphe de gradients. Oublier le premier fausse les
    résultats ; oublier le second fait exploser la mémoire.

    Args:
        model: Modèle à évaluer.
        loader: Chargeur produisant des couples ``(images, labels)``.
        device: Périphérique de calcul.

    Returns:
        Dictionnaire ``accuracy``, ``precision_macro``, ``recall_macro``, ``f1_macro``. Les
        moyennes sont macro : chaque classe pèse autant, quelle que soit sa fréquence.
    """
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_p.extend(preds.tolist())
            all_y.extend(y.numpy().tolist())
    return {
        "accuracy": float(accuracy_score(all_y, all_p)),
        "precision_macro": float(precision_score(all_y, all_p, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_y, all_p, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(all_y, all_p, average="macro", zero_division=0)),
    }


def train_epoch(model, loader, optimizer, criterion, device):
    """Fait une passe d'entraînement complète et renvoie la perte moyenne par exemple.

    La perte est pondérée par la taille réelle du lot avant d'être moyennée : un dernier
    lot incomplet ne doit pas peser autant qu'un lot plein.

    Args:
        model: Modèle à entraîner.
        loader: Chargeur d'entraînement.
        optimizer: Optimiseur, construit sur les seuls paramètres entraînables.
        criterion: Fonction de perte, appliquée aux logits.
        device: Périphérique de calcul.

    Returns:
        La perte moyenne par exemple sur l'époque.
    """
    model.train()
    running_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size
    return running_loss / max(1, count)


def main():
    """Exécute l'entraînement PyTorch de bout en bout, de la config aux artefacts.

    Déroulé : config et graines → transformations → jeux ``ImageFolder`` → découpage
    validation → chauffe (tête seule) → affinage (derniers blocs dégelés) → rechargement
    des meilleurs poids → évaluation sur le test → écriture du point de contrôle, des
    métriques et de l'historique, puis journalisation MLflow.

    Deux détails qui évitent des erreurs silencieuses :

    * le dossier d'entraînement est ouvert **deux fois**, une fois avec les transformations
      d'entraînement et une fois avec celles d'évaluation, avant d'être découpé par
      indices. Sans cela, le sous-ensemble de validation hériterait de l'augmentation
      aléatoire et ses scores danseraient d'une époque à l'autre ;
    * l'optimiseur est reconstruit après le dégel, sur les paramètres redevenus
      entraînables. Réutiliser l'ancien laisserait les blocs fraîchement dégelés hors de
      son champ, et l'affinage n'affinerait rien.
    """
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    epochs_total = int(args.epochs or cfg.get("epochs", 20))
    warmup_epochs = int(cfg.get("warmup_epochs", max(4, min(8, epochs_total // 3))))
    finetune_epochs = max(0, epochs_total - warmup_epochs)
    warmup_lr = float(cfg.get("warmup_learning_rate", 1e-3))
    finetune_lr = float(cfg.get("finetune_learning_rate", 3e-5))
    dataset_dir = Path(cfg["dataset_dir"])
    training_subdir = cfg.get("training_subdir", "Training")
    testing_subdir = cfg.get("testing_subdir", "Testing")
    model_dir = Path(ensure_dir(cfg.get("model_dir", "artifacts/models")))
    reports_dir = Path(ensure_dir(cfg.get("reports_dir", "artifacts/reports")))
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.get("mlflow_tracking_uri", "file:./mlruns"))
    mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", cfg.get("project_name", "medvision-brain-mri"))
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    train_tf, eval_tf = build_transforms(image_size)
    train_full = datasets.ImageFolder(dataset_dir / training_subdir, transform=train_tf)
    eval_full = datasets.ImageFolder(dataset_dir / training_subdir, transform=eval_tf)
    test_ds = datasets.ImageFolder(dataset_dir / testing_subdir, transform=eval_tf)
    num_classes = len(train_full.classes)

    indices = np.arange(len(train_full))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * float(cfg.get("validation_split", 0.15))))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_subset = torch.utils.data.Subset(train_full, train_idx.tolist())
    val_subset = torch.utils.data.Subset(eval_full, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_accuracy": [], "val_f1_macro": []}
    best_model = deepcopy(model.state_dict())
    best_score = -1.0

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=warmup_lr)
    for _ in range(warmup_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1_macro"].append(metrics["f1_macro"])
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = deepcopy(model.state_dict())

    unfreeze_last_layers(model, args.model, unfreeze_blocks=int(cfg.get("torch_unfreeze_blocks", 2)))
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=finetune_lr)
    for _ in range(finetune_epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_f1_macro"].append(metrics["f1_macro"])
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    test_metrics = evaluate(model, test_loader, device)

    model_path = model_dir / f"brain_mri_{args.model}.pt"
    metrics_path = reports_dir / f"brain_mri_{args.model}_metrics.json"
    history_path = reports_dir / f"brain_mri_{args.model}_history.json"

    torch.save(
        {"state_dict": model.state_dict(), "classes": train_full.classes, "model_name": args.model, "image_size": image_size},
        model_path,
    )
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"brain-mri-{args.model}"):
        mlflow.log_params({
            "problem": "brain_mri",
            "framework": "pytorch",
            "model": args.model,
            "image_size": image_size,
            "epochs_total": epochs_total,
            "warmup_epochs": warmup_epochs,
            "finetune_epochs": finetune_epochs,
            "warmup_lr": warmup_lr,
            "finetune_lr": finetune_lr,
        })
        for k, v in test_metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(history_path))

    print(json.dumps({"model_path": str(model_path), "metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
