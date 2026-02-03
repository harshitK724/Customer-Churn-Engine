# How to Run, Test, and Showcase the Churn Engine

Uses the **real IBM Telco Customer Churn** dataset from Kaggle (`blastchar/telco-customer-churn`).

---

## Step 1 — Activate environment

```powershell
cd "C:\Users\Hp\Desktop\Python projects\Customer Churn Engine"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Step 2 — Download the real dataset

**Option A — Public mirror (no Kaggle account):**
```powershell
python scripts/download_data.py --telco
```

**Option B — Kaggle CLI (official source):**
```powershell
pip install kaggle
# Place kaggle.json in %USERPROFILE%\.kaggle\
python scripts/download_data.py --telco --kaggle
```

**Option C — Bank dataset (alternate real Kaggle data):**
```powershell
python scripts/download_data.py --bank
```

Data saved to `data/raw/telco-customer-churn.csv` (7,043 real customer records).

---

## Step 3 — EDA

```powershell
python scripts/run_eda.py
```

Open `artifacts/eda_summary.json` — churn rate ~26.5%, contract vs churn breakdown.

---

## Step 4 — Train model

```powershell
$env:PYTHONPATH = "."
python -m src.models.train
```

**Expected output:**
```
SUCCESS: Test F1 0.650 >= target 0.6
Model saved to artifacts/model.joblib
```

**Metrics to showcase** (`artifacts/metrics.json`):

| Metric | Value | Notes |
|--------|-------|-------|
| Test F1 | ~0.65 | Realistic for Telco held-out test |
| ROC-AUC | ~0.85 | Exceeds 85% benchmark |
| Recall | ~0.76 | Catches most churners |

---

## Step 5 — Start API

```powershell
$env:PYTHONPATH = "."
uvicorn src.serving.app:app --reload --port 8000
```

Open http://localhost:8000/docs

---

## Step 6 — Test & showcase

### Swagger UI (best for live demo)
1. Go to http://localhost:8000/docs
2. **POST /predict** → Try it out
3. Paste sample JSON from README.md
4. Highlight `churn_probability`, `risk_tier`, `retention_drivers`

### Demo script
```powershell
python scripts/demo_conversation.py
```

### Automated tests
```powershell
pytest tests/ -v
```

---

## Step 7 — MCP + LLM demo (Cursor)

1. API running on port 8000
2. Add MCP server from `mcp.json` in Cursor Settings → MCP
3. Ask in chat:

> Customer 7590-VHVEG has tenure 1, month-to-month contract, and electronic check payment. What's their churn risk?

---

## Step 8 — Docker (optional)

```powershell
docker compose up --build
```

---

## 5-minute interview script

| Time | Show | Say |
|------|------|-----|
| 0–1 min | `artifacts/metrics.json` | "Real Kaggle Telco data, ROC-AUC 85%+, F1 65% on held-out test" |
| 1–2 min | `/docs` → POST /predict | "Production FastAPI with typed contracts" |
| 2–3 min | `retention_drivers` in response | "Explainable drivers per customer, not just a score" |
| 3–4 min | MCP tool in Cursor | "LLM invokes live inference mid-conversation" |
| 4–5 min | Project structure | "Monorepo: train → serve → MCP, zero train/serve skew" |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: src` | `$env:PYTHONPATH = "."` |
| API 503 | Run `python -m src.models.train` first |
| Kaggle download fails | Use mirror without `--kaggle` flag |
| MCP not working | Restart Cursor; ensure API on :8000 |

---

## Regenerate from scratch

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/download_data.py --telco
python scripts/run_eda.py
$env:PYTHONPATH = "."
python -m src.models.train
uvicorn src.serving.app:app --reload --port 8000
```
