> **Legal notice**  
> This repository is proprietary software owned by Doctum Consilium.  
> No use, reproduction, modification, or redistribution is permitted without prior written authorization.
# MedVision AI

MedVision is a computer-vision and MLOps playground for two medical imaging tasks:

- **Chest X-ray pneumonia classification**
- **Brain MRI tumor classification**

The repository combines:
- TensorFlow/Keras training scripts
- MLflow experiment tracking
- DVC pipelines and remote support
- FastAPI inference
- Streamlit comparison UI
- Terraform for provisioning an S3 bucket used as a DVC remote

## What is aligned now

FastAPI and Streamlit are now aligned on the same model registry so they can expose **all trained models for both problems** and compare their saved metrics.

Supported logical model slots:

### Chest X-ray
- `baseline`
- `optimized`

### Brain MRI
- `baseline`
- `optimized`

The serving layer reads artifacts from `artifacts/models` and `artifacts/reports` and builds one normalized registry.

## Project layout

```text
src/
  api/            # FastAPI app
  data/           # dataset download scripts
  evaluation/     # metrics and reports
  inference/      # task-specific prediction helpers
  models/         # Keras model builders
  preprocessing/  # image loading and preprocessing
  registry/       # model registry used by API and UI
  training/       # training entry points
  utils/          # config and dataset helpers
streamlit_app.py  # comparison interface
configs/          # YAML configs per task
terraform/        # infra for DVC remote
```

## Installation

> **Prérequis** : [uv](https://docs.astral.sh/uv/getting-started/installation/) — gestionnaire de paquets Python ultra-rapide.
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
> # winget install --id=astral-sh.uv -e             # Windows
> ```

### Créer l'environnement et installer les dépendances

```bash
uv venv .venv-wsl
source .venv-wsl/bin/activate   # Linux/macOS
# .venv-wsl\Scripts\activate    # Windows PowerShell

uv pip install -r requirements.txt
```

> **Note GPU** : le `requirements.txt` référence l'index PyTorch CUDA 12.4
> (`https://download.pytorch.org/whl/cu124`). Si vous n'avez pas de GPU NVIDIA,
> retirez la ligne `--extra-index-url` du fichier avant l'installation.

### Ajouter / supprimer un paquet

```bash
uv pip install <paquet>          # installer
uv pip uninstall <paquet>        # désinstaller
uv pip compile requirements.txt  # verrouiller les versions dans requirements.lock
```

## Train models

### Chest X-ray

```bash
python -m src.training.train --config configs/config.yaml --model baseline
python -m src.training.train --config configs/config.yaml --model optimized
```

### Brain MRI

```bash
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model baseline
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized
```

## Run FastAPI

```bash
uvicorn src.api.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /models`
- `GET /models?problem=brain_mri`
- `GET /compare?problem=chest_xray`
- `POST /predict?problem=brain_mri&model_name=optimized`

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

The Streamlit app lets you:
- pick one problem,
- see available trained models,
- compare stored metrics,
- upload one image and compare predictions across models.

## DVC

Use DVC for dataset and artifact versioning, not as a container runtime.

```bash
dvc init
dvc repro
dvc exp run
dvc exp show
```

See `README_DVC.md` for details.

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/FASTAPI_STREAMLIT_ALIGNMENT.md`
- `README_DVC.md`

## Notes

The comparison layer depends on expected artifact names in `artifacts/models` and metrics JSON files in `artifacts/reports`.
If you introduce new filenames, extend `src/registry/model_registry.py`.
