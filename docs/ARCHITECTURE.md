# Architecture

This document is the technical architecture reference for MedVision AI.
It explains the system as an engineering platform, not only as a collection of training scripts.

Audience: engineers extending or operating the repository.

## 1. Architectural goals

MedVision AI is designed around five practical goals:

1. Reproducibility: a run can be reconstructed from code, config, and data lineage.
2. Comparability: models across tasks can be compared in a consistent way.
3. Operability: API and UI can serve whichever models are actually available.
4. Extensibility: adding a new track should not require reworking all layers.
5. Traceability: data preparation, training decisions, and outcomes are discoverable.

## 2. System overview

The system is organized as layered modules with explicit interfaces.

### 2.1 Data layer

Responsibilities:

- Download raw datasets from Kaggle.
- Normalize heterogeneous segmentation datasets into manifest-driven structure.
- Provide deterministic inputs for training stages.

Core modules:

- src/data/download_dataset.py
- src/data/download_brain_mri_dataset.py
- src/data/download_segmentation_dataset.py
- src/data/prepare_segmentation_dataset.py
- src/segmentation/datasets/manifest.py

### 2.2 Training layer

Responsibilities:

- Train classification models for chest X-ray and brain MRI.
- Train multitask segmentation models (mask + class prediction).
- Emit artifacts, reports, and metrics with stable naming.

Core modules:

- src/training/train.py
- src/training/train_brain_mri.py
- src/segmentation/train_segmentation.py
- src/models/*
- src/segmentation/models/unet.py
- src/segmentation/metrics.py
- src/segmentation/overlays.py

### 2.3 Serving layer

Responsibilities:

- Expose model inference over HTTP (FastAPI).
- Support visual comparison workflows (Streamlit).
- Keep output semantics aligned between programmatic and UI paths.

Core modules:

- src/api/main.py
- streamlit_app.py
- src/registry/model_registry.py

### 2.4 MLOps layer

Responsibilities:

- Define pipeline dependencies and outputs using DVC.
- Track experiments and artifacts with MLflow.
- Optionally provision remote artifact storage through Terraform.

Core files:

- dvc.yaml
- params.yaml
- mlruns/
- terraform/aws_dvc_remote/

## 3. Data contracts

### 3.1 Raw data contract

Raw datasets are stored under data/raw.
Each track expects a known folder structure or a known preparation path.

### 3.2 Processed data contract for segmentation

Segmentation training is manifest-driven.
The preparation stage builds a CSV with a stable schema:

- image_path
- mask_path
- label
- split

This contract decouples training code from dataset-specific directory layouts.

### 3.3 Artifact contract

Training outputs are expected under artifacts/ with stable naming patterns:

- artifacts/models/*.keras
- artifacts/reports/*metrics*.json
- artifacts/reports/*history*.json
- artifacts/overlays/*.png

The registry relies on these naming conventions to discover available models.

## 4. Pipeline topology

The repository defines explicit DVC stages for each flow:

- download_chest_xray
- train_chest_xray
- download_brain_mri
- train_brain_mri
- download_brain_tumor_segmentation
- prepare_brain_tumor_segmentation
- train_brain_tumor_segmentation
- download_chest_xray_segmentation
- prepare_chest_xray_segmentation
- train_chest_xray_segmentation

Each stage declares dependencies, outputs, and in some cases metrics.
This makes stage invalidation predictable and reproducibility auditable.

## 5. End-to-end flow diagrams (textual)

### 5.1 Classification flow

raw data -> dataset builder -> classifier training -> metrics/reports/model -> registry discovery -> API/UI inference

### 5.2 Segmentation flow

raw data -> manifest preparation -> multitask U-Net training -> mask and class metrics + overlays -> registry discovery -> API/UI inference

## 6. Registry-centric serving design

A core architectural decision is to centralize model discovery and metadata in src/registry/model_registry.py.

Benefits:

- FastAPI and Streamlit read the same source of truth.
- New models can be exposed by artifact availability plus registry config.
- Serving behavior remains consistent across tracks.

Risks and constraints:

- Incorrect artifact naming can hide valid models.
- Registry specs must be updated when introducing new task families.

## 7. Multitask segmentation rationale

Instead of training pure segmentation-only models, the segmentation branch predicts:

- a dense segmentation mask
- an image-level class

Why this is useful in this project:

- Produces more informative demos and evaluations.
- Improves parity with classification-only tracks.
- Enables shared comparison views in Streamlit and API payloads.

## 8. Observability and traceability model

### 8.1 DVC answers pipeline lineage

DVC captures:

- Which dependencies were used.
- Which stages produced which outputs.
- Which outputs are out of date.

### 8.2 MLflow answers run performance

MLflow captures:

- Training parameters
- Scalar metrics
- Artifacts (models, reports, visual outputs)

Together they provide both lineage and performance context.

## 9. Configuration architecture

Configuration is split across two levels:

- Task-specific configs in configs/.
- Pipeline-oriented hyperparameters in params.yaml.

Guiding principle:

- Keep behavior changes explicit and versioned in config files.
- Avoid hidden constants inside training scripts unless unavoidable.

## 10. Operational architecture

### 10.1 Local run mode

Engineers can run services independently:

- FastAPI (uvicorn)
- Streamlit
- MLflow UI

This is preferred for debugging due to clearer logs and tighter iteration loops.

### 10.2 Composed run mode

docker-compose.yml runs:

- mlflow service
- api service
- streamlit service

This mode is useful for demo-style environments where integrated startup matters more than granular debugging.

## 11. Design trade-offs

1. Heuristic segmentation matching vs strict curated manifests:
	Heuristic matching accelerates onboarding but requires validation on new datasets.

2. 2D segmentation workflows vs full 3D volumetric pipelines:
	2D is simpler and faster to operate, but less expressive for volumetric MRI use cases.

3. Artifact naming conventions vs explicit model registration database:
	Naming-based discovery is lightweight, but strict conventions are mandatory.

## 12. Failure modes and mitigations

### 12.1 Manifest mismatch

Symptom:

- Good training loss behavior but poor qualitative overlays.

Mitigation:

- Inspect manifest rows.
- Visualize random image/mask samples before long runs.

### 12.2 Missing model in API/UI

Symptom:

- Model trained but not listed.

Mitigation:

- Verify artifact file names match registry candidates.
- Verify expected metrics/report paths.

### 12.3 Non-reproducible reruns

Symptom:

- Stage results differ unexpectedly across runs.

Mitigation:

- Check params.yaml and task configs for drift.
- Re-run with DVC status and stage-specific repro.

## 13. Extension blueprint for new tracks

To add a new task cleanly:

1. Add data acquisition/prep logic in src/data or src/segmentation.
2. Add training entry point with stable artifact outputs.
3. Add DVC stages with explicit deps/outs/metrics.
4. Add registry entries in src/registry/model_registry.py.
5. Validate inference payloads in API and Streamlit.
6. Update docs (README + architecture + operations).

## 14. Summary

MedVision AI is architected as a reproducible ML engineering platform where:

- data preparation is explicit,
- training outputs are contract-driven,
- serving is registry-centered,
- and experiment lifecycle is tracked through DVC and MLflow.

This architecture is intentionally pragmatic: simple enough to evolve quickly, structured enough to remain reliable as tracks expand.
