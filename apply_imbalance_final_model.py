"""
Apply the selected imbalance strategy as the production/dashboard model.

The selection is read from reports/imbalance_comparison.csv using the best
tuned F1, then the corresponding pipeline is retrained and exported.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "churn"
DROP_COLUMNS = ["customer_id"]
DATA_PATH = Path("data/customer_churn_business_dataset.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    y = df[TARGET].astype(int)
    numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return df, X, y, numerical, categorical


def preprocessor(numerical, categorical):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def make_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=130, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(80, 40),
            alpha=0.001,
            max_iter=140,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name}")


def make_sampler(strategy: str):
    if strategy == "Random Over-Sampling":
        return RandomOverSampler(random_state=RANDOM_STATE)
    if strategy == "SMOTE":
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    if strategy == "Random Under-Sampling":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    return None


def make_pipeline(model_name, strategy, numerical, categorical):
    model = make_model(model_name)
    sampler = make_sampler(strategy)
    prep = preprocessor(numerical, categorical)
    if sampler is None:
        return Pipeline([("preprocess", prep), ("model", model)])
    return ImbPipeline([("preprocess", prep), ("sampler", sampler), ("model", model)])


def evaluate(model, X_test, y_test, threshold):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "specificity": tn / (tn + fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def top_features(model, X_test, y_test):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=1,
    )
    rows = []
    for feature, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std):
        rows.append({"feature": feature, "importance": float(mean), "std": float(std)})
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:15]


def main():
    df, X, y, numerical, categorical = load_data()
    study = pd.read_csv(REPORTS_DIR / "imbalance_comparison.csv")
    best = study.sort_values(["tuned_f1", "tuned_pr_auc", "tuned_recall"], ascending=False).iloc[0]
    strategy = best["strategy"]
    model_name = best["model"]
    threshold = float(best["f1_threshold"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    model = make_pipeline(model_name, strategy, numerical, categorical)
    model.fit(X_train_full, y_train_full)
    metrics = evaluate(model, X_test, y_test, threshold)
    feature_importance = top_features(model, X_test, y_test)

    joblib.dump(model, MODELS_DIR / "random_forest_model.joblib")
    joblib.dump(model, MODELS_DIR / "selected_retention_pipeline.joblib")

    feature_info = {
        "all_features": X.columns.tolist(),
        "numerical_features": numerical,
        "categorical_features": categorical,
        "dropped_columns": DROP_COLUMNS,
        "n_features": len(X.columns),
    }
    metadata = {
        "model_type": model_name,
        "imbalance_strategy": strategy,
        "training_date": datetime.now().isoformat(),
        "target": TARGET,
        "dataset_rows": int(len(y)),
        "train_set_size": int(len(y_train_full)),
        "test_set_size": int(len(y_test)),
        "positive_class_rate": float(y.mean()),
        "imbalance_ratio": float(y.value_counts().max() / y.value_counts().min()),
        "n_features": len(X.columns),
        "n_classes": 2,
        "class_names": ["No Churn", "Churn"],
        "selection_metric": "f1_score_with_business_recall_constraint",
        "threshold": threshold,
        **metrics,
        "feature_importance": feature_importance,
    }

    (MODELS_DIR / "feature_names.json").write_text(json.dumps(json_safe(feature_info), indent=2))
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(json_safe(metadata), indent=2))
    print(json.dumps(json_safe({"selected": {"strategy": strategy, "model": model_name, "threshold": threshold}, "metrics": metrics}), indent=2))


if __name__ == "__main__":
    main()
