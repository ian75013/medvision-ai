# Components

## `src/models/`
Classification model builders.

## `src/segmentation/models/`
Segmentation and multitask segmentation/classification model builders.

## `src/segmentation/data.py`
Builds TensorFlow datasets from a segmentation manifest.

## `src/data/download_segmentation_dataset.py`
Downloads segmentation datasets from Kaggle using problem-specific shortcuts.

## `src/data/prepare_segmentation_dataset.py`
Transforms a raw dataset tree into a manifest suitable for training.

## `src/registry/model_registry.py`
Creates a normalized registry consumed by FastAPI and Streamlit for all tasks.
