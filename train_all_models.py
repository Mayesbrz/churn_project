"""
Train and compare all churn prediction models.

This script is the reproducible training entrypoint for the project:
- stratified train/test split
- preprocessing fitted on train only through sklearn pipelines
- 4 models including one Deep Learning MLP
- threshold tuning on validation data to avoid F1=0 on imbalanced classes
- model artifacts and metrics exported for the dashboard/API/report
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RANDOM_STATE = 42
DATA_PATH = Path("data/customer_churn_business_dataset.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
TARGET = "churn"
DROP_COLUMNS = ["customer_id"]


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_json_safe(value):
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    y = df[TARGET].astype(int)

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return X, y, numerical_features, categorical_features


def make_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_models(numerical_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    def pipe(model):
        return Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numerical_features, categorical_features)),
                ("model", model),
            ]
        )

    return {
        "Logistic Regression": pipe(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": pipe(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": pipe(
            GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        ),
        "MLP": pipe(
            MLPClassifier(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=64,
                learning_rate_init=0.001,
                max_iter=250,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=RANDOM_STATE,
            )
        ),
    }


def best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def evaluate(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "specificity": tn / (tn + fp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def permutation_top_features(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    rows = []
    for feature, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std):
        rows.append({"feature": feature, "importance": float(mean), "std": float(std)})
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows[:15]


def train() -> None:
    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X, y, numerical_features, categorical_features = load_data()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    models = get_models(numerical_features, categorical_features)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        cv_scores = cross_validate(model, X_train_full, y_train_full, cv=cv, scoring=scoring, n_jobs=1)
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = best_threshold(y_val, val_proba)

        final_model = get_models(numerical_features, categorical_features)[name]
        final_model.fit(X_train_full, y_train_full)
        metrics = evaluate(name, final_model, X_test, y_test, threshold)
        metrics["validation_f1_at_threshold"] = val_f1
        metrics["cv"] = {
            metric: {
                "mean": float(np.mean(cv_scores[f"test_{metric}"])),
                "std": float(np.std(cv_scores[f"test_{metric}"])),
            }
            for metric in scoring
        }
        results.append(metrics)
        fitted[name] = final_model
        print(
            f"{name}: ROC-AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, threshold={threshold:.2f}"
        )

    best = max(results, key=lambda row: (row["roc_auc"], row["f1_score"]))
    best_name = best["model"]
    best_model = fitted[best_name]
    feature_importance = permutation_top_features(best_model, X_test, y_test)

    joblib.dump(best_model, MODELS_DIR / "random_forest_model.joblib")
    for name, model in fitted.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{slug}_pipeline.joblib")

    feature_info = {
        "all_features": X.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "dropped_columns": DROP_COLUMNS,
        "n_features": len(X.columns),
    }
    metadata = {
        "model_type": best_name,
        "training_date": datetime.now().isoformat(),
        "target": TARGET,
        "dataset_rows": int(len(y)),
        "train_set_size": int(len(y_train_full)),
        "test_set_size": int(len(y_test)),
        "positive_class_rate": float(y.mean()),
        "n_features": len(X.columns),
        "n_classes": 2,
        "class_names": ["No Churn", "Churn"],
        "selection_metric": "roc_auc",
        "threshold": best["threshold"],
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1_score": best["f1_score"],
        "roc_auc": best["roc_auc"],
        "specificity": best["specificity"],
        "confusion_matrix": {
            "tn": best["true_negatives"],
            "fp": best["false_positives"],
            "fn": best["false_negatives"],
            "tp": best["true_positives"],
        },
        "feature_importance": feature_importance,
    }
    comparison = {
        "created_at": datetime.now().isoformat(),
        "models": results,
        "best_model": best_name,
        "feature_importance": feature_importance,
    }

    (MODELS_DIR / "feature_names.json").write_text(json.dumps(make_json_safe(feature_info), indent=2))
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(make_json_safe(metadata), indent=2))
    (MODELS_DIR / "model_comparison.json").write_text(json.dumps(make_json_safe(comparison), indent=2))
    pd.DataFrame(results).drop(columns=["cv"]).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\nBest model:", best_name)
    print(pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "threshold"]])


if __name__ == "__main__":
    train()
