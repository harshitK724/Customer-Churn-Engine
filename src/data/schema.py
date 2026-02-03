"""Pydantic schema — single contract for train, API, and MCP."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    customer_id: str
    gender: Literal["Male", "Female"]
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, description="Months as customer")
    phone_service: Literal["Yes", "No"]
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: str
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)


FEATURE_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
]

DATASET_NAME = "telco-customer-churn"
KAGGLE_SOURCE = "blastchar/telco-customer-churn"
