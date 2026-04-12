"""
Module de modélisation - Entraînement et comparaison des modèles
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from typing import Dict, Tuple, Any


class ModelTrainer:
    """Classe pour entraîner et gérer plusieurs modèles"""
    
    def __init__(self, random_state: int = 42):
        """Initialiser le trainer"""
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
    def define_models(self) -> Dict[str, Any]:
        """Définir les modèles de base"""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                verbose=0
            ),
        }
        
        print(f"✅ {len(self.models)} modèles définis : {list(self.models.keys())}")
        return self.models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Entraîner tous les modèles
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            
        Returns:
            Dictionnaire des modèles entraînés
        """
        if not self.models:
            self.define_models()
        
        for name, model in self.models.items():
            print(f"\n🔄 Entraînement de {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"✅ {name} entraîné")
        
        return self.trained_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Évaluer tous les modèles sur l'ensemble de test
        
        Args:
            X_test: Features de test
            y_test: Target de test
            cv_folds: Nombre de folds pour la validation croisée
            
        Returns:
            Dictionnaire des résultats
        """
        for name, model in self.trained_models.items():
            print(f"\n📊 Évaluation de {name}...")
            
            # Score sur l'ensemble de test
            train_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds)
            
            self.results[name] = {
                'test_score': train_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            print(f"   Score test : {train_score:.4f}")
            print(f"   CV mean : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.results


class DeepLearningTrainer:
    """Classe pour les modèles Deep Learning"""
    
    def __init__(self, random_state: int = 42):
        """Initialiser le trainer DL"""
        self.random_state = random_state
        self.model = None
        
    def create_mlp(self, input_dim: int, output_dim: int = 1, 
                   hidden_layers: list = None) -> Any:
        """
        Créer un MLP simple
        
        Args:
            input_dim: Nombre de features d'entrée
            output_dim: Nombre de sorties
            hidden_layers: Liste des tailles des couches cachées
            
        Returns:
            Modèle Keras compilé
        """
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            if hidden_layers is None:
                hidden_layers = [128, 64, 32]
            
            model = keras.Sequential()
            model.add(layers.Input(shape=(input_dim,)))
            
            for units in hidden_layers:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(0.2))
            
            model.add(layers.Dense(output_dim, activation='sigmoid'))
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC()]
            )
            
            print(f"✅ MLP créé avec architecture : Input({input_dim}) -> {hidden_layers} -> Output({output_dim})")
            self.model = model
            return model
            
        except ImportError:
            print("⚠️ TensorFlow non disponible. Installez-le avec : pip install tensorflow")
            return None


if __name__ == "__main__":
    print("Module de modélisation chargé")
