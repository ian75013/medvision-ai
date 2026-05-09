# INFRASTRUCTURE

## Purpose
medvision-ai manages medical AI training/inference pipelines with reproducible data workflows.

## Main Components
- Core project code and configs (params.yaml, dvc.yaml)
- Docker runtime (docker-compose.yml and docker-compose.prod.yml)
- Python dependencies from requirements.txt

## Local Run
1. Install dependencies or use Docker runtime.
2. Start services with docker compose up -d --build.
3. Run project-specific checks before model/data operations.

## Deployment
- Use production compose for controlled environment rollout.
- Keep model/data versions aligned with DVC metadata.

## Operations and Validation
- Validate container health and data-path access.
- Verify model-serving endpoints after updates.

## Rollback
- Revert to previous image and DVC revision.
- Restart services with known-good configuration.
