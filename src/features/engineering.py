import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    service_cols = [
        "phone_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    out["service_count"] = out[service_cols].eq("Yes").sum(axis=1)
    out["avg_monthly_spend"] = out["total_charges"] / out["tenure"].clip(lower=1)
    out["is_month_to_month"] = (out["contract"] == "Month-to-month").astype(int)
    out["is_electronic_check"] = out["payment_method"].str.contains(
        "Electronic check", regex=False
    ).astype(int)
    out["has_fiber"] = out["internet_service"].eq("Fiber optic").astype(int)
    return out
