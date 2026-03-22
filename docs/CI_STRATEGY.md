# MedVision AI CI strategy

This pack is designed to work with the current repository **without modifying the source code**.

## Why multiple dependency files?

The current `requirements.txt` mixes together:
- lightweight scientific/runtime packages,
- TensorFlow,
- PyTorch,
- notebooks,
- DVC / cloud tooling,
- Streamlit / MLflow.

For CI, that is too much for every job. A clean strategy is to split the dependency graph by responsibility.

## Proposed dependency layers

- `requirements/base.txt`:
  shared runtime packages used by the current unit tests.
- `requirements/ci-fast.txt`:
  very fast checks for metrics and preprocessing.
- `requirements/ci-tf.txt`:
  lightweight TensorFlow profile required because `tests/test_api.py`
  imports `src.api.main`, which imports `src.registry.model_registry`, which imports TensorFlow.
- `requirements/ci-full.txt`:
  a slightly larger but still reasonable profile to run the current `tests/` directory as a whole.
- `requirements/dev-lite.txt`:
  a local developer profile for notebooks and experiments without installing the full production stack.

## Proposed workflow split

### 1. `ci-fast.yml`
Runs on every push/PR and validates the cheapest tests:
- `tests/test_metrics.py`
- `tests/test_preprocessing.py`

### 2. `ci-tf.yml`
Runs on every push/PR and validates:
- TensorFlow installation/import
- `tests/test_api.py`

### 3. `ci-suite.yml`
Runs on `main`/`master` and manually via `workflow_dispatch`, and executes the current full `tests/` folder.

## Why this is compatible with the current repo

No source changes are required.
The key point is that `tests/test_api.py` currently needs TensorFlow to be installed because of the import chain:

`tests/test_api.py -> src.api.main -> src.registry.model_registry -> tensorflow`

Therefore, the API CI job includes TensorFlow.
