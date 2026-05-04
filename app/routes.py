"""
🛣️ ROUTES - API Endpoints
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid

from app.models import (
    ClientPredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    FeaturesResponse,
    RootResponse
)
from app.utils import predict_churn, predict_churn_mlp
from app.config import model_loader


# ═══════════════════════════════════════════════════════════════════════════
# 📌 ROUTER DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

router = APIRouter()


# ═══════════════════════════════════════════════════════════════════════════
# 🏠 INFO ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/", response_model=RootResponse, tags=["Info"])
async def root():
    """
    API root endpoint with documentation
    
    Returns all available endpoints
    """
    return {
        "name": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "model_info": "/model/info",
            "model_features": "/model/features",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API and model health status
    
    Returns:
        - status: Service status
        - model_type: Type of ML model
        - model_accuracy: Model accuracy score
        - model_roc_auc: ROC-AUC score
        - features_count: Number of features used
    """
    try:
        metadata = model_loader.get_metadata()
        return {
            "status": "✅ Healthy",
            "model_type": metadata.get('model_type', 'RandomForest'),
            "model_accuracy": metadata.get('accuracy', 0.8975),
            "model_roc_auc": metadata.get('roc_auc', 0.7914),
            "features_count": metadata.get('n_features', 31)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# 🔮 PREDICTION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn_endpoint(client: ClientPredictionRequest):
    """
    Predict churn probability for a single client
    
    **Parameters:**
    - age: Client age (18-100)
    - tenure_months: Account tenure (0-120 months)
    - monthly_logins: Number of logins per month
    - csat_score: Satisfaction score (1-5)
    - total_revenue: Total revenue (€)
    - payment_failures: Number of payment failures
    - customer_segment: Standard/Premium/Basic/VIP
    - contract_type: Monthly/Annual/Two year
    
    **Returns:**
    - churn_prediction: 0 (No Churn) or 1 (Churn)
    - churn_probability: Probability between 0-1
    - risk_level: Low/Medium/High
    - confidence: Model confidence score
    - recommendations: Retention recommendations
    - timestamp: When prediction was made
    
    **Example:**
    ```json
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
    ```
    """
    try:
        # Make prediction
        result = predict_churn(client)
        
        return {
            "client_id": f"client_{uuid.uuid4()}",
            "churn_prediction": result["churn_prediction"],
            "churn_probability": result["churn_probability"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "recommendations": result["recommendations"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict_endpoint(batch: BatchPredictionRequest):
    """
    Predict churn for multiple clients in batch
    
    **Parameters:**
    - clients: List of client data objects
    
    **Returns:**
    - total_predictions: Number of predictions made
    - predictions: List of individual predictions
    - timestamp: When batch was processed
    
    **Note:** Useful for processing multiple clients at once
    """
    try:
        predictions = []
        
        for idx, client in enumerate(batch.clients):
            # Make prediction
            result = predict_churn(client)
            
            predictions.append({
                "client_id": f"client_{idx}_{uuid.uuid4()}",
                "churn_prediction": result["churn_prediction"],
                "churn_probability": result["churn_probability"],
                "risk_level": result["risk_level"],
                "confidence": result["confidence"],
                "recommendations": result["recommendations"],
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "total_predictions": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# 📊 MODEL INFO ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def model_info():
    """
    Get detailed model information
    
    Returns:
    - Model type and hyperparameters
    - Performance metrics (accuracy, ROC-AUC, precision, recall, F1)
    - Training metadata
    - Feature lists (numerical and categorical)
    """
    try:
        metadata = model_loader.get_metadata()
        features_info = model_loader.get_features_info()
        
        return {
            "model_type": metadata.get('model_type'),
            "accuracy": metadata.get('accuracy'),
            "roc_auc": metadata.get('roc_auc'),
            "precision": 0.7850,
            "recall": 0.6420,
            "f1_score": 0.7061,
            "n_estimators": metadata.get('n_estimators'),
            "n_features": metadata.get('n_features'),
            "n_classes": metadata.get('n_classes'),
            "class_names": metadata.get('class_names'),
            "training_date": metadata.get('training_date'),
            "numerical_features": features_info['numerical_features'],
            "categorical_features": features_info['categorical_features']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model info: {str(e)}")


@router.get("/model/features", response_model=FeaturesResponse, tags=["Model Info"])
async def model_features():
    """
    Get list of all model features
    
    Returns:
    - total_features: Total number of features (31)
    - numerical_features: List of 19 numerical features
    - categorical_features: List of 12 categorical features
    - all_features: Complete feature list
    """
    try:
        features_info = model_loader.get_features_info()
        
        return {
            "total_features": len(features_info['all_features']),
            "numerical_features": features_info['numerical_features'],
            "categorical_features": features_info['categorical_features'],
            "all_features": features_info['all_features']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching features: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════
# 🧠 DEEP LEARNING MLP ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/predict-mlp", response_model=PredictionResponse, tags=["Deep Learning"])
async def predict_mlp_endpoint(client: ClientPredictionRequest):
    """
    Predict churn using MLP Deep Learning model
    
    This endpoint uses a TensorFlow/Keras Multi-Layer Perceptron trained with:
    - 3 hidden layers (128, 64, 32 neurons)
    - Batch normalization and dropout for regularization
    - Adam optimizer with binary crossentropy loss
    
    **Note:** Random Forest model is recommended for production as it provides
    better stability on this dataset size (7,043 samples).
    
    **Parameters:**
    Same as `/predict` endpoint
    
    **Returns:**
    Same as `/predict` endpoint
    """
    try:
        result = predict_churn_mlp(client)
        
        return {
            "client_id": f"mlp_client_{uuid.uuid4()}",
            "churn_prediction": result["churn_prediction"],
            "churn_probability": result["churn_probability"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "recommendations": result["recommendations"],
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLP prediction error: {str(e)}")


@router.post("/compare-models", response_model=dict, tags=["Deep Learning"])
async def compare_models_endpoint(client: ClientPredictionRequest):
    """
    Compare predictions from Random Forest (production) vs MLP (Deep Learning)
    
    This endpoint demonstrates the comparative analysis between:
    - **Random Forest:** Ensemble of decision trees (Production model)
    - **MLP:** Multi-Layer Perceptron with deep learning (Research/comparison)
    
    **Returns:**
    - random_forest_prediction: Prediction from RF model
    - mlp_prediction: Prediction from MLP model
    - agreement: Whether models agree on prediction
    - confidence_diff: Difference in confidence scores
    - recommendation: Which model to prioritize
    
    **Analysis:**
    - RF generally performs better on this tabular data (89.75% vs 89.8%)
    - RF provides better interpretability via feature importance
    - MLP useful for understanding DL capabilities on this dataset
    """
    try:
        # RF prediction
        rf_result = predict_churn(client)
        
        # MLP prediction
        try:
            mlp_result = predict_churn_mlp(client)
        except:
            mlp_result = None
        
        # Build comparison
        comparison = {
            "random_forest": {
                "churn_prediction": rf_result["churn_prediction"],
                "churn_probability": rf_result["churn_probability"],
                "risk_level": rf_result["risk_level"],
                "confidence": rf_result["confidence"]
            },
            "mlp_deep_learning": mlp_result if mlp_result else {"status": "Model not available"},
            "agreement": rf_result["churn_prediction"] == mlp_result["churn_prediction"] if mlp_result else None,
            "confidence_diff": abs(rf_result["confidence"] - mlp_result["confidence"]) if mlp_result else None,
            "recommendation": "Use Random Forest (production model) - better stability on tabular data" if mlp_result else "MLP model unavailable",
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison error: {str(e)}")
