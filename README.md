# Customer Churn Engine

Predict telecom customer churn with XGBoost, serve predictions over FastAPI, and query risk from Cursor via MCP.

## Highlights

- **Real data** — IBM Telco Customer Churn (7K+ records); Bank dataset supported
- **Tuned pipeline** — Optuna hyperparameter search + SMOTE for class imbalance
- **Explainable output** — churn probability, risk tier, and top retention drivers
- **Production-ready** — REST API, Docker, MCP tools for LLM workflows

## Model Performance (Telco test set)

| Metric    | Score |
|-----------|-------|
| F1        | 0.65  |
| ROC-AUC   | 0.85  |
| Precision | 0.57  |
| Recall    | 0.76  |

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/download_data.py --telco
python scripts/run_eda.py
$env:PYTHONPATH = "."
python -m src.models.train

uvicorn src.serving.app:app --reload --port 8000
pytest tests/ -v
```

**Alternate datasets**

```powershell
python scripts/download_data.py --bank          # Bank churn CSV
python scripts/download_data.py --telco --kaggle  # Kaggle CLI
```

## API

| Method | Endpoint         | Description              |
|--------|------------------|--------------------------|
| GET    | `/health`        | Service health check     |
| GET    | `/model/info`    | Model metrics & features |
| POST   | `/predict`       | Single customer prediction |
| POST   | `/predict/batch` | Batch predictions      |

Example:

```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"customer_id":"C001","gender":"Female","senior_citizen":0,"partner":"Yes","dependents":"No","tenure":12,"phone_service":"Yes","multiple_lines":"No","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"Yes","streaming_movies":"No","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":89.5,"total_charges":1080.0}'
```

## MCP (Cursor)

1. Start the API on port `8000`
2. Add `mcp.json` to **Cursor Settings → MCP**
3. Tools: `predict_customer_churn`, `get_churn_model_info`

## Docker

```powershell
docker compose up --build
```

## Project Layout

```
src/
  data/          # Schema & CSV loading
  features/      # Engineering & preprocessing
  models/        # Train, evaluate, explain
  serving/       # FastAPI app
  mcp_server/    # MCP tools
scripts/         # Download data, EDA, demo
artifacts/       # Model, metrics, schema (generated)
tests/           # API & schema tests
```

## Config

All paths and training settings live in `config.yaml`. Artifacts are written to `artifacts/` after training.
