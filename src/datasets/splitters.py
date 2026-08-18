"""Découpage train / validation / test **au niveau du patient**.

POURQUOI par patient et non par image : un patient contribue souvent plusieurs images
(coupes, incidences, examens successifs). Découper au hasard sur les images place presque
sûrement des images d'un même patient des deux côtés de la frontière. Le modèle apprend
alors à reconnaître ce patient précis, ses scores de test explosent, et rien de tout cela ne
survit au premier patient inconnu. C'est la fuite de données la plus banale — et la plus
coûteuse — en imagerie médicale.

Le découpage est stratifié sur le label, pour que chaque jeu garde la même proportion de
cas positifs, et déterministe à graine fixée.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.paths import ensure_dir

#: Colonnes que le CSV de métadonnées doit fournir. ``patient_id`` est la clé du découpage,
#: ``path`` désigne le fichier, ``label`` sert à stratifier.
REQUIRED_COLUMNS = {"patient_id", "path", "label"}


def create_patient_level_splits(
    metadata_csv: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> tuple[Path, Path, Path, Path]:
    """Découpe un CSV de métadonnées en trois jeux, sans jamais séparer un patient.

    Le découpage se fait en deux temps sur la table **dédupliquée des patients** : d'abord
    l'entraînement contre le reste, puis le reste en validation et test. La seconde
    proportion est recalculée relativement à ce reste (``test_size / (val_size +
    test_size)``), sans quoi les fractions finales ne seraient pas celles demandées. Les
    images sont ensuite rattachées à leur patient par jointure.

    Args:
        metadata_csv: CSV décrivant les images — doit contenir au moins ``patient_id``,
            ``path`` et ``label``.
        output_dir: Dossier de sortie, créé au besoin.
        seed: Graine du découpage. Deux appels de même graine donnent la même partition.
        val_size: Fraction des **patients** en validation.
        test_size: Fraction des **patients** en test.

    Returns:
        Le quadruplet de chemins ``(splits.csv, train.csv, val.csv, test.csv)``.
        ``splits.csv`` réunit tout avec une colonne ``split`` — c'est la vue à archiver,
        celle qui permet de prouver des mois après quelle image était de quel côté.

    Raises:
        ValueError: Une colonne obligatoire manque, ou une classe est trop peu représentée
            pour être stratifiée (scikit-learn exige au moins deux patients par classe dans
            chaque découpage).

    Note:
        Les fractions portent sur le nombre de **patients**, pas d'images. Si les patients
        n'ont pas tous le même nombre d'images, les jeux résultants ne respectent pas
        exactement ces proportions en nombre d'images — c'est le prix, assumé, de l'absence
        de fuite.
    """
    metadata_csv = Path(metadata_csv)
    output_dir = ensure_dir(output_dir)
    df = pd.read_csv(metadata_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    patient_df = df[["patient_id", "label"]].drop_duplicates().reset_index(drop=True)

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=patient_df["label"],
    )

    relative_test_size = test_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_patients["label"],
    )

    split_frames = {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients,
    }

    merged_frames: dict[str, pd.DataFrame] = {}
    for split_name, patient_split in split_frames.items():
        merged = df.merge(patient_split[["patient_id"]], on="patient_id", how="inner").copy()
        merged["split"] = split_name
        merged_frames[split_name] = merged

    split_csv = output_dir / "splits.csv"
    pd.concat(merged_frames.values(), ignore_index=True).to_csv(split_csv, index=False)

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"
    merged_frames["train"].to_csv(train_csv, index=False)
    merged_frames["val"].to_csv(val_csv, index=False)
    merged_frames["test"].to_csv(test_csv, index=False)
    return split_csv, train_csv, val_csv, test_csv
