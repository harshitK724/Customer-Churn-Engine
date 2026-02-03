import numpy as np


def explain_prediction(
    classifier,
    transformed_row: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Use XGBoost pred_contribs for per-prediction SHAP-style drivers."""
    import xgboost as xgb

    dmatrix = xgb.DMatrix(transformed_row)
    contribs = classifier.get_booster().predict(dmatrix, pred_contribs=True)
    values = contribs[0][:-1]

    pairs = sorted(
        zip(feature_names, values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_k]

    drivers = []
    for name, impact in pairs:
        drivers.append({
            "feature": name,
            "impact": round(float(abs(impact)), 4),
            "direction": "increases_risk" if impact > 0 else "decreases_risk",
        })
    return drivers
