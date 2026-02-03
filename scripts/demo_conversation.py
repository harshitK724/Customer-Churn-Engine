"""End-to-end demo with a real IBM Telco customer profile."""
import json
import sys

import httpx

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

API_URL = "http://localhost:8000"


def main() -> None:
    print("=== Customer Churn Engine — End-to-End Demo ===\n")
    print("Dataset: IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)")
    print("Scenario: New customer, month-to-month, electronic check payment.\n")

    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{API_URL}/health")
        health.raise_for_status()
        print(f"API health: {health.json()}\n")

        info = client.get(f"{API_URL}/model/info")
        info.raise_for_status()
        print(f"Model info: {json.dumps(info.json(), indent=2)}\n")

        response = client.post(f"{API_URL}/predict", json=SAMPLE)
        response.raise_for_status()
        result = response.json()

    print("Prediction result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("ERROR: API not running. Start it with:")
        print("  uvicorn src.serving.app:app --reload --port 8000")
        sys.exit(1)
