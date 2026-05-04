"""
Prediction helpers for the optional FastAPI layer.

The dashboard does not depend on this module. It is kept functional so the API
can still serve the same trained sklearn pipelines when required.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from app.config import BASE_DIR, model_loader


FEATURE_ORDER = [
    "gender",
    "age",
    "country",
    "city",
    "customer_segment",
    "tenure_months",
    "signup_channel",
    "contract_type",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
    "monthly_fee",
    "total_revenue",
    "payment_method",
    "payment_failures",
    "discount_applied",
    "price_increase_last_3m",
    "support_tickets",
    "avg_resolution_time",
    "complaint_type",
    "csat_score",
    "escalations",
    "email_open_rate",
    "marketing_click_rate",
    "nps_score",
    "survey_response",
    "referral_count",
]


def client_to_dataframe(client_data) -> pd.DataFrame:
    row = {feature: getattr(client_data, feature) for feature in FEATURE_ORDER}
    if row["complaint_type"] in {"nan", "None", ""}:
        row["complaint_type"] = None
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def get_risk_level(probability: float) -> str:
    if probability >= 0.5:
        return "HIGH"
    if probability >= 0.15:
        return "MEDIUM"
    return "LOW"


def generate_recommendations(client_data, probability: float) -> list[str]:
    recommendations = []
    if probability >= 0.7:
        recommendations.append("Contact client immediately with a targeted retention offer.")
    elif probability >= 0.4:
        recommendations.append("Schedule proactive CRM outreach.")

    if client_data.csat_score < 3:
        recommendations.append("Investigate low satisfaction and open a support quality review.")
    if client_data.monthly_logins < 5:
        recommendations.append("Launch a re-engagement campaign to increase product usage.")
    if client_data.payment_failures >= 2:
        recommendations.append("Offer billing support or adapted payment terms.")
    if client_data.contract_type == "Monthly":
        recommendations.append("Propose an incentive for a longer-term contract.")

    if not recommendations:
        recommendations.append("Customer health is good; maintain standard relationship monitoring.")
    return recommendations


def _predict_with_pipeline(model, client_data, threshold: float) -> dict:
    X = client_to_dataframe(client_data)
    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= threshold)
    confidence = max(probability, 1 - probability)
    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
        "risk_level": get_risk_level(probability),
        "confidence": confidence,
        "recommendations": generate_recommendations(client_data, probability),
    }


def predict_churn(client_data):
    metadata = model_loader.get_metadata()
    threshold = float(metadata.get("threshold", 0.5))
    return _predict_with_pipeline(model_loader.get_model(), client_data, threshold)


def predict_churn_mlp(client_data):
    mlp_path = Path(BASE_DIR) / "models" / "mlp_pipeline.joblib"
    if not mlp_path.exists():
        raise Exception("MLP pipeline is not available. Run train_all_models.py first.")
    model = joblib.load(mlp_path)
    return _predict_with_pipeline(model, client_data, threshold=0.5)
