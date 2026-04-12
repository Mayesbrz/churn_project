"""
Module d'évaluation - Métriques et comparaison des modèles
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from typing import Dict, Tuple, Any


class ModelEvaluator:
    """Classe pour évaluer les modèles"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculer les métriques de classification
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions
            y_pred_proba: Probabilités prédites
            
        Returns:
            Dictionnaire des métriques
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Obtenir la matrice de confusion"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Obtenir le rapport de classification détaillé"""
        return classification_report(y_true, y_pred)
    
    @staticmethod
    def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Obtenir la courbe ROC
        
        Returns:
            Tuple de (fpr, tpr, auc_score)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score


class ModelComparator:
    """Classe pour comparer plusieurs modèles"""
    
    def __init__(self):
        """Initialiser le comparateur"""
        self.results_df = None
        
    def compare_models(self, models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Comparer plusieurs modèles
        
        Args:
            models_results: Dictionnaire contenant les résultats de chaque modèle
            
        Returns:
            DataFrame comparatif
        """
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            row = {
                'Model': model_name,
                **metrics
            }
            comparison_data.append(row)
        
        self.results_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("COMPARAISON DES MODÈLES")
        print("="*80)
        print(self.results_df.to_string(index=False))
        
        return self.results_df
    
    def get_best_model(self, metric: str = 'F1') -> str:
        """Obtenir le meilleur modèle selon une métrique"""
        if self.results_df is None:
            return None
        
        best_idx = self.results_df[metric].idxmax()
        best_model = self.results_df.loc[best_idx, 'Model']
        
        print(f"\n🏆 Meilleur modèle (selon {metric}) : {best_model}")
        return best_model
    
    def rank_models(self, metric: str = 'F1') -> pd.DataFrame:
        """Classer les modèles par une métrique"""
        if self.results_df is None:
            return None
        
        ranked = self.results_df.sort_values(metric, ascending=False)
        print(f"\n📊 Classement des modèles (selon {metric}) :")
        print(ranked[['Model', metric]].to_string(index=False))
        
        return ranked


if __name__ == "__main__":
    print("Module d'évaluation chargé")
