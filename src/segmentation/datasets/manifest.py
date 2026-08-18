"""Appariement image ↔ masque ↔ diagnostic, et écriture du ``manifest.csv``.

POURQUOI ce module existe : aucun dataset médical public n'est rangé comme le précédent.
Certains posent ``CHNCXR_0001.png`` et ``CHNCXR_0001_mask.png`` côte à côte ; d'autres
séparent ``images/`` et ``masks/`` ; d'autres encore ne donnent le diagnostic que dans un
rapport texte (Montgomery, dossier ``ClinicalReadings``). Plutôt que d'écrire un
chargeur par dataset, on ramène tout ici à un tableau unique — image, masque, label,
split — et le reste du pipeline n'a plus jamais à connaître la forme d'origine.

Stratégie d'appariement, dans l'ordre :

1. **Regrouper par nom normalisé** — on retire les suffixes ``_mask``, ``_seg``,
   ``_annotation``, ``_label`` et toute ponctuation, de sorte que ``CHNCXR_0001.png`` et
   ``CHNCXR-0001_mask.png`` tombent dans le même groupe.
2. **Dans chaque groupe, désigner l'image et le masque** — par le nom quand il est
   explicite, sinon par l'extension (le JPEG est l'image, le PNG/TIFF sans perte est le
   masque : personne n'encode un masque binaire en JPEG).
3. **Trouver le label** — d'abord dans les rapports cliniques s'il y en a, sinon dans le
   nom des dossiers traversés.

Un groupe dont l'image ou le masque manque est **ignoré sans erreur** : les datasets
contiennent des fichiers orphelins, et échouer sur le premier rendrait la préparation
inutilisable. La contrepartie est qu'un manifeste vide est un échec silencieux — c'est
``build_segmentation_datasets`` qui le rattrape et lève.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

#: Extensions considérées comme des images candidates lors du balayage.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
#: Extensions sans perte, privilégiées comme masque quand le nom ne tranche pas :
#: un masque binaire compressé en JPEG serait bruité sur les contours.
MASK_EXTS_PRIORITY = {".png", ".bmp", ".tif", ".tiff"}


def _normalize_stem(path: Path) -> str:
    """Réduit un nom de fichier à une clé d'appariement image/masque.

    Passe en minuscules, retire un éventuel suffixe de masque et toute ponctuation, pour
    que les variantes d'un même cas se retrouvent dans le même groupe.

    Args:
        path: Chemin du fichier.

    Returns:
        Clé alphanumérique, par exemple ``"chncxr0001"`` pour ``CHNCXR-0001_mask.png``.
    """
    stem = path.stem.lower()
    stem = re.sub(r"(?:_mask|mask|_seg|_annotation|_label)$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def _label_from_path(path: Path, known_labels: Iterable[str]) -> str | None:
    """Cherche un label connu dans les dossiers traversés par le chemin.

    Convention la plus répandue : le diagnostic est le nom du dossier
    (``.../pneumonia/img_12.png``). La comparaison tolère les underscores et accepte
    l'inclusion, pour attraper aussi ``.../pneumonia_bacterial/``.

    Args:
        path: Chemin de l'image.
        known_labels: Labels attendus pour ce dataset, tels que déclarés dans la config.

    Returns:
        Le label reconnu, ou ``None`` si aucun ne figure dans le chemin.
    """
    parts = [p.lower() for p in path.parts]
    normalized_parts = [p.replace("_", " ").strip() for p in parts]

    for label in known_labels:
        lbl = label.lower().replace("_", " ").strip()
        for part in normalized_parts:
            if lbl == part or lbl in part:
                return label
    return None


def _resolve_binary_labels(known_labels: Iterable[str]) -> tuple[str | None, str | None]:
    """Devine lequel des labels déclarés est le « sain » et lequel est le « pathologique ».

    Sert à interpréter les rapports cliniques, qui parlent de maladies et non des noms de
    classes du projet. Le label normal se reconnaît au mot « normal » ; l'anormal à un
    vocabulaire de pathologie. À défaut, on prend simplement l'autre label — un dataset
    binaire n'a que deux possibilités.

    Args:
        known_labels: Labels déclarés pour le dataset.

    Returns:
        Le couple ``(label_normal, label_anormal)``, chaque terme pouvant être ``None`` si
        rien ne correspond.
    """
    labels = list(known_labels)
    normal_label = next((label for label in labels if "normal" in label.lower()), None)
    abnormal_label = next(
        (
            label
            for label in labels
            if any(token in label.lower() for token in ("abnormal", "pneumonia", "positive", "tb", "tuberculosis"))
        ),
        None,
    )
    if abnormal_label is None:
        abnormal_label = next((label for label in labels if label != normal_label), None)
    return normal_label, abnormal_label


def _infer_label_from_report_text(text: str, known_labels: list[str]) -> str | None:
    """Lit un compte rendu radiologique et en déduit « sain » ou « pathologique ».

    Règle de décision, volontairement prudente : un rapport n'est classé normal que s'il
    ne contient **aucun** marqueur de pathologie ; dès qu'un marqueur apparaît, le cas est
    anormal. Un doute est donc résolu du côté pathologique, ce qui est le sens sûr en
    médecine — et un rapport qui ne dit ni l'un ni l'autre renvoie ``None`` plutôt qu'un
    label inventé.

    Args:
        text: Contenu brut du compte rendu.
        known_labels: Labels déclarés pour le dataset, servant de cible à la traduction.

    Returns:
        Le label déduit, ou ``None`` si le texte ne tranche pas.
    """
    normal_label, abnormal_label = _resolve_binary_labels(known_labels)
    if normal_label is None and abnormal_label is None:
        return None

    content = " ".join(text.lower().split())
    normal_markers = (
        "normal",
        "no active disease",
        "no acute disease",
        "no focal infiltrate",
        "clear lungs",
        "normal chest radiograph",
        "heart size is normal",
    )
    abnormal_markers = (
        "abnormal",
        "tuberculosis",
        "tb",
        "infiltrate",
        "opacity",
        "opacities",
        "consolidation",
        "effusion",
        "lesion",
        "calcification",
        "cavity",
        "fibrosis",
        "disease",
        "pneumonia",
        "cardiopulmonary abnormality",
    )

    has_normal = any(marker in content for marker in normal_markers)
    has_abnormal = any(marker in content for marker in abnormal_markers)

    if has_normal and not has_abnormal:
        return normal_label
    if has_abnormal:
        return abnormal_label
    return None


def _build_report_label_map(raw_dir: Path, known_labels: list[str]) -> dict[str, str]:
    """Indexe les comptes rendus cliniques du dataset : clé de cas → label déduit.

    Ne concerne que les datasets qui livrent un dossier ``ClinicalReadings`` (Montgomery,
    par exemple). Pour les autres, la table revient vide et l'appelant se rabat sur le nom
    des dossiers.

    Un rapport illisible (encodage exotique, fichier corrompu) est ignoré : perdre un cas
    est préférable à interrompre la préparation de tout le dataset.

    Args:
        raw_dir: Racine du dataset téléchargé.
        known_labels: Labels déclarés pour le dataset.

    Returns:
        Table ``clé normalisée du cas → label``, vide s'il n'y a pas de rapports.
    """
    report_dirs = [raw_dir / "ClinicalReadings", raw_dir / "clinicalreadings"]
    report_dir = next((path for path in report_dirs if path.exists()), None)
    if report_dir is None:
        return {}

    labels: dict[str, str] = {}
    for report_path in report_dir.rglob("*"):
        if not report_path.is_file():
            continue
        try:
            content = report_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        label = _infer_label_from_report_text(content, known_labels)
        if label is not None:
            labels[_normalize_stem(report_path)] = label
    return labels


def _looks_like_mask(path: Path) -> bool:
    """Vrai si le nom du fichier ou de son dossier annonce un masque.

    Args:
        path: Chemin du fichier image candidat.

    Returns:
        True pour ``..._mask.png``, ``..._seg.png``, ``...label...`` ou tout fichier rangé
        dans ``mask/``, ``masks/``, ``seg/``, ``segs/`` ou ``labels/``.
    """
    name = path.stem.lower()
    parent = path.parent.name.lower()
    return (
        "mask" in name
        or name.endswith("_seg")
        or "label" in name
        or parent in {"mask", "masks", "seg", "segs", "labels"}
    )


def build_manifest(raw_dir: str | Path, output_csv: str | Path, known_labels: list[str]) -> pd.DataFrame:
    """Balaie un dataset brut et écrit le ``manifest.csv`` du pipeline de segmentation.

    Applique la stratégie décrite en tête de module : regroupement par nom normalisé,
    désignation image/masque dans chaque groupe, puis étiquetage par rapport clinique ou
    par nom de dossier. Le split est déduit du chemin (``test`` s'il contient ce mot,
    ``train`` sinon), afin de respecter la partition officielle quand le dataset en fournit
    une.

    Args:
        raw_dir: Racine du dataset téléchargé, parcourue récursivement.
        output_csv: Chemin du CSV à écrire ; les dossiers parents sont créés.
        known_labels: Labels attendus pour ce dataset, tels que déclarés dans la config.

    Returns:
        Le manifeste, avec les colonnes ``image_path``, ``mask_path``, ``label`` et
        ``split``. Un cas dont le label reste indéterminé est étiqueté ``"unknown"``,
        **sauf** si le dataset fournit des rapports cliniques : dans ce cas un cas non
        identifié est écarté, car on préfère un jeu plus petit à des étiquettes fausses.

    Example:
        >>> df = build_manifest("data/raw/montgomery", "data/processed/manifest.csv",
        ...                     ["normal", "tuberculosis"])
        >>> sorted(df.columns.tolist())
        ['image_path', 'label', 'mask_path', 'split']
    """
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)

    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    report_labels = _build_report_label_map(raw_dir, known_labels)

    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_normalize_stem(path), []).append(path)

    rows = []

    for _key, candidates in grouped.items():
        if len(candidates) < 2:
            continue

        explicit_masks = [p for p in candidates if _looks_like_mask(p)]
        explicit_images = [p for p in candidates if not _looks_like_mask(p)]

        image_path = None
        mask_path = None

        if explicit_masks and explicit_images:
            mask_path = explicit_masks[0]
            image_path = explicit_images[0]
        else:
            jpgs = [p for p in candidates if p.suffix.lower() in {".jpg", ".jpeg"}]
            mask_like = [p for p in candidates if p.suffix.lower() in MASK_EXTS_PRIORITY]

            if jpgs and mask_like:
                image_path = jpgs[0]
                mask_path = next((p for p in mask_like if p != image_path), None)

        if image_path is None or mask_path is None:
            continue

        label = report_labels.get(_normalize_stem(image_path)) or _label_from_path(image_path, known_labels)
        if report_labels and label is None:
            continue
        split = "test" if "test" in str(image_path).lower() else "train"

        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "label": label or "unknown",
                "split": split,
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
