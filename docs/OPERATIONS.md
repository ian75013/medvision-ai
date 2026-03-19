# Operations / Runbook

## Commandes locales utiles

### Installer
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Télécharger les données MRI
```bash
python -m src.data.download_brain_mri_dataset --config configs/brain_tumor_mri.yaml
```

### Entraîner
```bash
python -m src.training.train_brain_mri --config configs/brain_tumor_mri.yaml --model optimized --epochs 10
```

### Lancer MLflow
```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

### Lancer l'API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Lancer Streamlit
```bash
streamlit run streamlit_app.py
```

### DVC
```bash
dvc init
dvc repro
dvc exp run
dvc exp show
```

### Docker Compose
```bash
docker compose up --build
```

## Vérifications rapides

- `pytest -q`
- vérifier que `kaggle.json` est en place
- vérifier que `artifacts/models/` existe après entraînement
- vérifier que `mlruns/` se remplit

## Conseils

- utiliser le workflow Brain MRI comme flux principal ;
- garder le flux chest X-ray comme référence historique ;
- documenter explicitement les modèles manquants si tu continues la branche historique.
