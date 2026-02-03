import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.005):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    preds = (y_proba >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
