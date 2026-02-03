import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.config import CONFIG
from src.data.schema import DATASET_NAME, FEATURE_COLUMNS, KAGGLE_SOURCE
from src.features.engineering import engineer_features
from src.models.explain import explain_prediction
from src.serving.schemas import PredictResponse, RetentionDriver


class ModelService:
    def __init__(self):
        self.pipeline = None
        self.threshold = 0.5
        self.feature_names: list[str] = []
        self.metrics: dict = {}

    def load(self) -> None:
        bundle = joblib.load(CONFIG["paths"]["model_file"])
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle["threshold"]
        artifacts_dir = Path(CONFIG["paths"]["artifacts_dir"])
        self.feature_names = joblib.load(artifacts_dir / "feature_names.joblib")
        with open(CONFIG["paths"]["metrics_file"], encoding="utf-8") as f:
            self.metrics = json.load(f)

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def _to_dataframe(self, payload: dict) -> pd.DataFrame:
        row = {
            "gender": payload["gender"],
            "senior_citizen": payload["senior_citizen"],
            "partner": payload["partner"],
            "dependents": payload["dependents"],
            "tenure": payload["tenure"],
            "phone_service": payload["phone_service"],
            "multiple_lines": payload["multiple_lines"],
            "internet_service": payload["internet_service"],
            "online_security": payload["online_security"],
            "online_backup": payload["online_backup"],
            "device_protection": payload["device_protection"],
            "tech_support": payload["tech_support"],
            "streaming_tv": payload["streaming_tv"],
            "streaming_movies": payload["streaming_movies"],
            "contract": payload["contract"],
            "paperless_billing": payload["paperless_billing"],
            "payment_method": payload["payment_method"],
            "monthly_charges": payload["monthly_charges"],
            "total_charges": payload["total_charges"],
        }
        return engineer_features(pd.DataFrame([row]))

    def predict(self, payload: dict) -> PredictResponse:
        start = time.perf_counter()
        df = self._to_dataframe(payload)
        proba = float(self.pipeline.predict_proba(df)[:, 1][0])
        pred = proba >= self.threshold

        if proba >= 0.7:
            tier = "high"
        elif proba >= 0.4:
            tier = "medium"
        else:
            tier = "low"

        transformed = self.pipeline.named_steps["preprocessor"].transform(df)
        classifier = self.pipeline.named_steps["classifier"]
        drivers_raw = explain_prediction(
            classifier, transformed, self.feature_names, top_k=5
        )
        drivers = [RetentionDriver(**d) for d in drivers_raw]

        latency_ms = (time.perf_counter() - start) * 1000
        print(
            f"predict customer={payload.get('customer_id')} "
            f"proba={proba:.3f} latency={latency_ms:.1f}ms"
        )

        return PredictResponse(
            customer_id=payload["customer_id"],
            churn_probability=round(proba, 4),
            churn_prediction=pred,
            risk_tier=tier,
            retention_drivers=drivers,
        )

    def model_info(self) -> dict:
        return {
            "model_version": "2.2.0",
            "dataset": DATASET_NAME,
            "kaggle_source": KAGGLE_SOURCE,
            "f1_score": self.metrics.get("f1", 0),
            "roc_auc": self.metrics.get("roc_auc", 0),
            "threshold": self.threshold,
            "features": FEATURE_COLUMNS,
        }


model_service = ModelService()
