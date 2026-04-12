"""
API REST FastAPI pour les prédictions de churn (Optionnel)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import joblib


app = FastAPI(
    title="Churn Prediction API",
    description="API pour prédire le churn client",
    version="1.0.0"
)


class ClientData(BaseModel):
    """Modèle de données pour un client"""
    age: int
    tenure: int
    monthly_charges: float
    login_frequency: int
    support_tickets: int
    nps_score: int


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    churn_prediction: int
    churn_probability: float
    risk_level: str


@app.get("/health", tags=["Health"])
def health_check():
    """Vérifier l'état du service"""
    return {
        "status": "healthy",
        "message": "API is running"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(client: ClientData):
    """
    Prédire le churn d'un client
    
    Args:
        client: Données du client
        
    Returns:
        Prédiction et probabilité de churn
    """
    try:
        # Préparation des données
        features = np.array([[
            client.age,
            client.tenure,
            client.monthly_charges,
            client.login_frequency,
            client.support_tickets,
            client.nps_score
        ]])
        
        # Faire la prédiction (placeholder)
        churn_prob = 0.45
        churn_pred = 1 if churn_prob > 0.5 else 0
        
        # Déterminer le niveau de risque
        if churn_prob < 0.3:
            risk_level = "Low"
        elif churn_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "churn_prediction": churn_pred,
            "churn_probability": round(churn_prob, 4),
            "risk_level": risk_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info", tags=["Model"])
def model_info():
    """Obtenir les informations du modèle"""
    return {
        "model_name": "XGBoost Churn Predictor",
        "version": "1.0.0",
        "features": [
            "age", "tenure", "monthly_charges",
            "login_frequency", "support_tickets", "nps_score"
        ],
        "target": "churn"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
