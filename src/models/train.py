"""Train XGBoost churn model. Run: python -m src.models.train"""
import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import CONFIG
from src.data.load_data import export_schema, load_raw_data, prepare_xy, split_data
from src.features.preprocess import build_preprocessor, get_feature_names
from src.models.evaluate import compute_metrics, find_best_threshold


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            enable_categorical=False,
            random_state=CONFIG["training"]["random_state"],
            n_jobs=-1,
        )),
    ])


def objective(trial, X_train, y_train, X_val, y_val, spw, preprocessor, use_smote):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        enable_categorical=False,
        random_state=CONFIG["training"]["random_state"],
        n_jobs=-1,
        **params,
    )
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t = preprocessor.transform(X_val)
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=CONFIG["training"]["random_state"])
        X_train_t, y_train = smote.fit_resample(X_train_t, y_train)
    classifier.fit(X_train_t, y_train)
    proba = classifier.predict_proba(X_val_t)[:, 1]
    _, f1 = find_best_threshold(y_val, proba)
    return f1


def _fit_classifier(X_train, y_train, spw, use_smote, best_params):
    final_preprocessor = build_preprocessor()
    X_train_t = final_preprocessor.fit_transform(X_train, y_train)
    y_fit = y_train
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=CONFIG["training"]["random_state"])
        X_train_t, y_fit = smote.fit_resample(X_train_t, y_train)
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        enable_categorical=False,
        random_state=CONFIG["training"]["random_state"],
        n_jobs=-1,
        **best_params,
    )
    classifier.fit(X_train_t, y_fit)
    return Pipeline([
        ("preprocessor", final_preprocessor),
        ("classifier", classifier),
    ])


def main():
    df = load_raw_data()
    X, y = prepare_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    churn_rate = float(y_train.mean())
    use_smote = CONFIG["training"].get("use_smote", False)
    if use_smote:
        spw = 1.0
        print(f"Churn rate: {churn_rate:.2%} | SMOTE enabled | scale_pos_weight={spw}")
    else:
        spw = (1 - churn_rate) / max(churn_rate, 1e-6)
        print(f"Churn rate: {churn_rate:.2%} | scale_pos_weight={spw:.2f}")

    study = optuna.create_study(direction="maximize")
    preprocessor = build_preprocessor()
    study.optimize(
        lambda t: objective(
            t, X_train, y_train, X_val, y_val, spw, preprocessor, use_smote
        ),
        n_trials=CONFIG["training"]["optuna_trials"],
        show_progress_bar=True,
    )
    print(f"Best validation F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    final_pipe = _fit_classifier(
        X_train_full, y_train_full, spw, use_smote, study.best_params
    )

    val_proba = final_pipe.predict_proba(X_val)[:, 1]
    threshold, _ = find_best_threshold(y_val, val_proba)

    test_proba = final_pipe.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_proba, threshold)
    metrics["best_params"] = study.best_params
    metrics["validation_f1"] = study.best_value

    print("\n=== TEST METRICS ===")
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value}")

    target_f1 = CONFIG["training"]["target_f1"]
    if metrics["f1"] < target_f1:
        print(f"\nWARNING: Test F1 {metrics['f1']:.3f} < target {target_f1}")
    else:
        print(f"\nSUCCESS: Test F1 {metrics['f1']:.3f} >= target {target_f1}")

    artifacts_dir = Path(CONFIG["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = CONFIG["paths"]["model_file"]
    joblib.dump({"pipeline": final_pipe, "threshold": threshold}, model_path)

    feat_names = get_feature_names(final_pipe.named_steps["preprocessor"])
    joblib.dump(feat_names, artifacts_dir / "feature_names.joblib")

    with open(CONFIG["paths"]["metrics_file"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    export_schema()
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
