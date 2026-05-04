"""
Module d'explicabilité
"""

import pandas as pd
from sklearn.inspection import permutation_importance


class ExplainabilityAnalyzer:
    """Classe pour analyser l'explicabilité des modèles"""
    
    @staticmethod
    def get_feature_importance(model, feature_names):
        """Obtenir l'importance des features (pour modèles tree-based)"""
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
    def get_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
        """Calculer l'importance par permutation"""
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
