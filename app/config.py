"""
⚙️ CONFIGURATION - API Settings & Constants
"""

import os
import json
import joblib
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 📁 PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ═══════════════════════════════════════════════════════════════════════════
# 🤖 MODEL ARTIFACTS PATHS - Random Forest (Production)
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ═══════════════════════════════════════════════════════════════════════════
# 🧠 MLP DEEP LEARNING MODEL PATHS
# ═══════════════════════════════════════════════════════════════════════════

MLP_MODEL_PATH = MODELS_DIR / "mlp_model.h5"
MLP_SCALER_PATH = MODELS_DIR / "mlp_scaler.joblib"
MLP_ENCODERS_PATH = MODELS_DIR / "mlp_label_encoders.joblib"
MLP_METADATA_PATH = MODELS_DIR / "mlp_metadata.json"

# ═══════════════════════════════════════════════════════════════════════════
# 🔧 API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

API_TITLE = "🎯 Churn Prediction API"
API_DESCRIPTION = "Production API for Customer Churn Prediction - Advanced ML Model"
API_VERSION = "1.0.0"

HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "info"

# ═══════════════════════════════════════════════════════════════════════════
# 📊 MODEL LOADING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class ModelLoader:
    """Load all model artifacts (Random Forest + MLP)"""
    
    _instance = None  # Singleton
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_artifacts()
        return cls._instance
    
    def _load_artifacts(self):
        """Load Random Forest and MLP models with artifacts"""
        try:
            print("📦 Loading model artifacts...")
            
            # Random Forest (Production Model)
            self.model = joblib.load(str(MODEL_PATH))
            self.scaler = joblib.load(str(SCALER_PATH))
            self.encoders = joblib.load(str(ENCODERS_PATH))
            
            with open(FEATURES_PATH) as f:
                self.features_info = json.load(f)
            
            with open(METADATA_PATH) as f:
                self.model_metadata = json.load(f)
            
            print("✅ Random Forest model loaded!")
            
            # MLP Deep Learning Model
            try:
                import tensorflow as tf
                self.mlp_model = tf.keras.models.load_model(str(MLP_MODEL_PATH))
                self.mlp_scaler = joblib.load(str(MLP_SCALER_PATH))
                self.mlp_encoders = joblib.load(str(MLP_ENCODERS_PATH))
                
                with open(MLP_METADATA_PATH) as f:
                    self.mlp_metadata = json.load(f)
                
                print("✅ MLP Deep Learning model loaded!")
            except Exception as e:
                print(f"⚠️  Warning: Could not load MLP model: {e}")
                self.mlp_model = None
                self.mlp_scaler = None
                self.mlp_encoders = None
                self.mlp_metadata = None
            
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            raise
    
    def get_model(self):
        return self.model
    
    def get_scaler(self):
        return self.scaler
    
    def get_encoders(self):
        return self.encoders
    
    def get_features_info(self):
        return self.features_info
    
    def get_metadata(self):
        return self.model_metadata
    
    def get_mlp_model(self):
        return self.mlp_model
    
    def get_mlp_scaler(self):
        return self.mlp_scaler
    
    def get_mlp_encoders(self):
        return self.mlp_encoders
    
    def get_mlp_metadata(self):
        return self.mlp_metadata


# Initialize on import
try:
    model_loader = ModelLoader()
except Exception as e:
    print(f"⚠️ Warning: Could not load model artifacts: {e}")
    model_loader = None
