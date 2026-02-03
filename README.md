# Context-Aware Predictive Customer Churn Engine

End-to-end churn prediction with XGBoost, FastAPI, Docker, and MCP integration for live LLM inference.

**Dataset:** Real [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle (7,043 customers).

> **Full run & showcase guide:** see [SHOWCASE.md](SHOWCASE.md)

## Model Performance (`artifacts/metrics.json`)

| Metric | Value |
|--------|-------|
| Test F1 (churn class) | **0.650** |
| ROC-AUC | **0.851** |
| Precision | 0.566 |
| Recall | 0.764 |

On real-world Telco data, **ROC-AUC exceeds 85%**; F1 on the minority churn class is ~65% on a strict held-out test split — typical for this benchmark.

## Quick Start

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download real Kaggle dataset (mirror; or use --kaggle with Kaggle CLI)
python scripts/download_data.py --telco
python scripts/run_eda.py

$env:PYTHONPATH = "."
python -m src.models.train

uvicorn src.serving.app:app --reload --port 8000
```

### Download from Kaggle directly (optional)

1. Install Kaggle CLI: `pip install kaggle`
2. Place `kaggle.json` in `%USERPROFILE%\.kaggle\`
3. Run: `python scripts/download_data.py --telco --kaggle`

## Sample Predict Request

```json
{
  "customer_id": "7590-VHVEG",
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 1,
  "phone_service": "No",
  "multiple_lines": "No phone service",
  "internet_service": "DSL",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 29.85,
  "total_charges": 29.85
}
```

## MCP Integration

See `mcp.json` and [SHOWCASE.md](SHOWCASE.md) for Cursor setup.

## Other real datasets

```powershell
python scripts/download_data.py --bank   # Bank Customer Churn (10K rows)
```
