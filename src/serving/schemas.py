from typing import Literal

from pydantic import BaseModel, Field

from src.data.schema import CustomerFeatures


class PredictRequest(CustomerFeatures):
    pass


class RetentionDriver(BaseModel):
    feature: str
    impact: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_tier: Literal["low", "medium", "high"]
    retention_drivers: list[RetentionDriver]


class ModelInfoResponse(BaseModel):
    model_version: str
    dataset: str = "telco-customer-churn"
    kaggle_source: str = "blastchar/telco-customer-churn"
    f1_score: float
    roc_auc: float = 0.0
    threshold: float
    features: list[str]
