import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG
from src.data.schema import DATASET_NAME, FEATURE_COLUMNS, KAGGLE_SOURCE
from src.features.engineering import engineer_features

ENGINEERED_COLUMNS = [
    "service_count",
    "avg_monthly_spend",
    "is_month_to_month",
    "is_electronic_check",
    "has_fiber",
]

TELCO_COLUMN_MAP = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    path = path or CONFIG["paths"]["raw_data"]
    df = pd.read_csv(path)
    df = df.rename(columns=TELCO_COLUMN_MAP)
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"])
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)
    return df


def _encode_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return (series.astype(str).str.strip() == "Yes").astype(int)


def prepare_xy(df: pd.DataFrame):
    enriched = engineer_features(df)
    feature_cols = FEATURE_COLUMNS + ENGINEERED_COLUMNS
    X = enriched[feature_cols].copy()
    y = _encode_target(df["churn"])
    return X, y


def split_data(X, y):
    rs = CONFIG["training"]["random_state"]
    test_size = CONFIG["training"]["test_size"]
    val_size = CONFIG["training"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rs
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=rs
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def export_schema(out_path: str | None = None) -> None:
    out_path = out_path or CONFIG["paths"]["schema_file"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET_NAME,
        "kaggle_source": KAGGLE_SOURCE,
        "features": FEATURE_COLUMNS,
        "engineered_features": ENGINEERED_COLUMNS,
        "target": "churn",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
