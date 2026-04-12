"""
Module d'explicabilité - Feature Importance et SHAP
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Dict, Any, Tuple


class ExplainabilityAnalyzer:
    """Classe pour analyser l'explicabilité des modèles"""
    
    @staticmethod
    def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
        """
        Obtenir l'importance des features (pour modèles tree-based)
        
        Args:
            model: Modèle entraîné
            feature_names: Liste des noms de features
            
        Returns:
            DataFrame avec importance des features
        """
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False)
            
            print(f"✅ Feature Importance extraite")
            return importance_df
        else:
            print("⚠️ Ce modèle n'a pas d'attribut feature_importances_")
            return None
    
    @staticmethod
    def get_permutation_importance(model: Any, X_test: np.ndarray, 
                                   y_test: np.ndarray, 
                                   feature_names: list,
                                   n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculer l'importance par permutation (fonctionnel pour tous les modèles)
        
        Args:
            model: Modèle entraîné
            X_test: Features de test
            y_test: Target de test
            feature_names: Liste des noms de features
            n_repeats: Nombre de répétitions
            
        Returns:
            DataFrame avec importance par permutation
        """
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats, random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        print(f"✅ Permutation Importance calculée")
        return importance_df
    
    @staticmethod
    def shap_analysis(model: Any, X_test: np.ndarray, feature_names: list = None):
        """
        Analyser les prédictions avec SHAP
        
        Args:
            model: Modèle entraîné
            X_test: Features de test
            feature_names: Liste des noms de features
            
        Returns:
            SHAP Explainer et values
        """
        try:
            import shap
            
            # Créer l'explainer SHAP
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            
            print(f"✅ SHAP Explainer créé")
            return explainer, shap_values
            
        except ImportError:
            print("⚠️ SHAP n'est pas installé. Installez-le avec : pip install shap")
            return None, None


if __name__ == "__main__":
    print("Module d'explicabilité chargé")
