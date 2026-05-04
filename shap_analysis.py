"""
SHAP explainability for the final churn model.

This script explains the final pipeline exported in models/random_forest_model.joblib.
It produces global and local explanations for the positive class churn=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "churn"
DROP_COLUMNS = ["customer_id"]
DATA_PATH = Path("data/customer_churn_business_dataset.csv")
MODEL_PATH = Path("models/random_forest_model.joblib")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


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


def get_class_one_shap(raw_values):
    """Return SHAP values for class 1 regardless of SHAP version output shape."""
    if isinstance(raw_values, list):
        return raw_values[1]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def collapse_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # OneHotEncoder names use the form column_category. Keep the original
    # column when it can be recovered from known categorical variables later.
    return name


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = joblib.load(MODEL_PATH)
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    sample_size = min(800, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[X_sample.index]
    X_transformed = preprocess.transform(X_sample)
    feature_names = preprocess.get_feature_names_out()
    feature_names = [collapse_feature_name(str(name)) for name in feature_names]

    explainer = shap.TreeExplainer(model)
    shap_values = get_class_one_shap(explainer.shap_values(X_transformed))
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(np.asarray(expected_value)[1])
    else:
        expected_value = float(expected_value)

    mean_abs = np.abs(shap_values).mean(axis=0)
    global_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "mean_shap": shap_values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    global_df.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)

    top_global = global_df.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top_global["feature"], top_global["mean_abs_shap"], color="#1f4e79")
    plt.xlabel("Mean absolute SHAP value")
    plt.title("SHAP global importance - churn class")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_bar.png", dpi=180)
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    probabilities = pipeline.predict_proba(X_sample)[:, 1]
    local_rows = []
    chosen_positions = np.argsort(probabilities)[-5:][::-1]
    for rank, pos in enumerate(chosen_positions, start=1):
        row_values = shap_values[pos]
        order = np.argsort(np.abs(row_values))[-8:][::-1]
        explanation = []
        for idx in order:
            explanation.append(
                {
                    "feature": feature_names[idx],
                    "shap_value": float(row_values[idx]),
                    "effect": "augmente le risque" if row_values[idx] > 0 else "réduit le risque",
                }
            )
        local_rows.append(
            {
                "rank": rank,
                "row_index": int(X_sample.index[pos]),
                "true_churn": int(y_sample.iloc[pos]),
                "predicted_probability": float(probabilities[pos]),
                "top_factors": explanation,
            }
        )

    flat_local = []
    for row in local_rows:
        for factor in row["top_factors"]:
            flat_local.append(
                {
                    "rank": row["rank"],
                    "row_index": row["row_index"],
                    "true_churn": row["true_churn"],
                    "predicted_probability": row["predicted_probability"],
                    **factor,
                }
            )
    pd.DataFrame(flat_local).to_csv(REPORTS_DIR / "shap_local_examples.csv", index=False)

    output = {
        "model_path": str(MODEL_PATH),
        "explained_class": "churn=1",
        "sample_size": sample_size,
        "expected_value": expected_value,
        "top_global_features": global_df.head(15).to_dict(orient="records"),
        "local_examples": local_rows,
        "artifacts": {
            "global_csv": "reports/shap_global_importance.csv",
            "local_csv": "reports/shap_local_examples.csv",
            "summary_bar": "reports/shap_summary_bar.png",
            "beeswarm": "reports/shap_beeswarm.png",
        },
    }
    (MODELS_DIR / "shap_explanations.json").write_text(json.dumps(json_safe(output), indent=2))
    print(json.dumps(json_safe({"sample_size": sample_size, "top_features": output["top_global_features"][:5]}), indent=2))


if __name__ == "__main__":
    main()
