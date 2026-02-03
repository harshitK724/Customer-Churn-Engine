"""MCP server — run: python -m src.mcp_server.server"""
import json
import os

import httpx
from mcp.server import MCPServer

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

mcp = MCPServer("churn-engine")


@mcp.tool()
async def predict_customer_churn(
    customer_id: str,
    gender: str,
    senior_citizen: int,
    partner: str,
    dependents: str,
    tenure: int,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    paperless_billing: str,
    payment_method: str,
    monthly_charges: float,
    total_charges: float,
) -> str:
    """
    Predict whether a telecom customer is likely to churn and explain why.

    Call this when:
    - A customer shows declining engagement or service usage
    - Support issues, contract type, or payment method are discussed
    - You need churn risk before offering retention actions

    Returns churn probability, risk tier, and top retention drivers.
    """
    payload = {
        "customer_id": customer_id,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{API_BASE}/predict", json=payload)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


@mcp.tool()
async def get_churn_model_info() -> str:
    """Get churn model metadata: F1, ROC-AUC, dataset, threshold, and features."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{API_BASE}/model/info")
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)


if __name__ == "__main__":
    mcp.run()
