import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.model_service import model_service

# Real customer from IBM Telco dataset (Kaggle)
SAMPLE = {
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
    "total_charges": 29.85,
}


@pytest.fixture(scope="module", autouse=True)
def load_model():
    if not model_service.is_ready:
        model_service.load()


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "telco-customer-churn"
    assert body["f1_score"] >= 0.60
    assert body["roc_auc"] >= 0.85


def test_predict():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert len(body["retention_drivers"]) <= 5
