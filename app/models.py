"""
📊 PYDANTIC MODELS - Request/Response schemas
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════
# 📥 REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ClientPredictionRequest(BaseModel):
    """Client data for churn prediction - All 31 features required"""
    
    # Categorical features
    customer_id: str = Field(..., description="Unique customer ID")
    gender: str = Field(..., description="Gender: Male, Female")
    country: str = Field(..., description="Customer country")
    city: str = Field(..., description="Customer city")
    customer_segment: str = Field(..., description="Segment: Standard, Premium, Basic, VIP")
    signup_channel: str = Field(..., description="How customer signed up: Web, Mobile, Partner")
    contract_type: str = Field(..., description="Contract type: Monthly, Annual, Two year")
    payment_method: str = Field(..., description="Payment method: Credit Card, Bank Transfer, PayPal")
    discount_applied: str = Field(..., description="Discount status: Yes, No")
    price_increase_last_3m: str = Field(..., description="Price increase: Yes, No")
    complaint_type: str = Field(..., description="Complaint type: Technical, Billing, Service, None")
    survey_response: str = Field(..., description="Survey response: Positive, Negative, Neutral, No response")
    
    # Numerical features
    age: float = Field(..., ge=18, le=100, description="Client age (18-100)")
    tenure_months: float = Field(..., ge=0, le=120, description="Account tenure in months (0-120)")
    monthly_logins: float = Field(..., ge=0, le=30, description="Monthly logins count (0-30)")
    weekly_active_days: float = Field(..., ge=0, le=7, description="Weekly active days (0-7)")
    avg_session_time: float = Field(..., ge=0, le=1000, description="Average session time in minutes")
    features_used: float = Field(..., ge=0, le=50, description="Number of features used")
    usage_growth_rate: float = Field(..., ge=-100, le=300, description="Usage growth rate (%)")
    last_login_days_ago: float = Field(..., ge=0, le=365, description="Days since last login")
    monthly_fee: float = Field(..., ge=0, le=500, description="Monthly fee in €")
    total_revenue: float = Field(..., ge=0, le=10000, description="Total revenue in €")
    payment_failures: int = Field(..., ge=0, le=20, description="Number of payment failures")
    support_tickets: int = Field(..., ge=0, le=100, description="Number of support tickets")
    avg_resolution_time: float = Field(..., ge=0, le=1000, description="Average resolution time in hours")
    csat_score: float = Field(..., ge=1, le=5, description="Customer satisfaction score (1-5)")
    escalations: int = Field(..., ge=0, le=50, description="Number of escalations")
    email_open_rate: float = Field(..., ge=0, le=1, description="Email open rate (0-1)")
    marketing_click_rate: float = Field(..., ge=0, le=1, description="Marketing click rate (0-1)")
    nps_score: float = Field(..., ge=-100, le=100, description="Net Promoter Score (-100 to 100)")
    referral_count: int = Field(..., ge=0, le=100, description="Number of referrals")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "gender": "Male",
                "country": "France",
                "city": "Paris",
                "customer_segment": "Standard",
                "signup_channel": "Web",
                "contract_type": "Monthly",
                "payment_method": "Credit Card",
                "discount_applied": "No",
                "price_increase_last_3m": "No",
                "complaint_type": "None",
                "survey_response": "Positive",
                "age": 35,
                "tenure_months": 24,
                "monthly_logins": 10,
                "weekly_active_days": 5,
                "avg_session_time": 45,
                "features_used": 15,
                "usage_growth_rate": 10,
                "last_login_days_ago": 2,
                "monthly_fee": 50,
                "total_revenue": 500,
                "payment_failures": 1,
                "support_tickets": 2,
                "avg_resolution_time": 24,
                "csat_score": 3.5,
                "escalations": 0,
                "email_open_rate": 0.6,
                "marketing_click_rate": 0.3,
                "nps_score": 45,
                "referral_count": 2
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    clients: List[ClientPredictionRequest] = Field(..., description="List of clients")
    
    class Config:
        json_schema_extra = {
            "example": {
                "clients": [
                    {
                        "age": 35,
                        "tenure_months": 24,
                        "monthly_logins": 10,
                        "csat_score": 3.5,
                        "total_revenue": 500,
                        "payment_failures": 1,
                        "customer_segment": "Standard",
                        "contract_type": "Monthly"
                    }
                ]
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# 📤 RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    """API response for single prediction"""
    client_id: str = Field(..., description="Unique client identifier")
    churn_prediction: int = Field(..., description="0=No Churn, 1=Churn")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="Risk category: Low, Medium, High")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    recommendations: List[str] = Field(..., description="Retention recommendations")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_predictions: int = Field(..., description="Total predictions made")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    timestamp: str = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """API health check response"""
    status: str = Field(..., description="Service status")
    model_type: str = Field(..., description="Type of model")
    model_accuracy: float = Field(..., description="Model accuracy")
    model_roc_auc: float = Field(..., description="Model ROC-AUC score")
    features_count: int = Field(..., description="Number of features")


class ModelInfoResponse(BaseModel):
    """Model detailed information"""
    model_type: str
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    n_estimators: int
    n_features: int
    n_classes: int
    class_names: List[str]
    training_date: str
    numerical_features: List[str]
    categorical_features: List[str]


class FeaturesResponse(BaseModel):
    """Model features list"""
    total_features: int
    numerical_features: List[str]
    categorical_features: List[str]
    all_features: List[str]


class RootResponse(BaseModel):
    """Root endpoint response"""
    name: str
    version: str
    endpoints: dict
