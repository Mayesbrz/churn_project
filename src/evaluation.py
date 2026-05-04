"""
Module d'évaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class ModelEvaluator:
    """Classe pour évaluer les modèles"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculer les métriques de classification"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics


class ModelComparator:
    """Classe pour comparer plusieurs modèles"""
    
    def __init__(self):
        self.results_df = None
        
    def compare_models(self, models_results):
        """Comparer plusieurs modèles"""
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            row = {'Model': model_name, **metrics}
            comparison_data.append(row)
        
        self.results_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("COMPARAISON DES MODÈLES")
        print("="*80)
        print(self.results_df.to_string(index=False))
        
        return self.results_df
