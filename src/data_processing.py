"""
Module de préparation et nettoyage des données
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class DataProcessor:
    """Classe pour traiter et préparer les données"""
    
    def __init__(self, csv_path: str):
        self.df = None
        self.csv_path = csv_path
        self.numerical_features = []
        self.categorical_features = []
        self.target = None
        
    def load_data(self) -> pd.DataFrame:
        """Charger le dataset"""
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Dataset chargé : {self.df.shape[0]} clients, {self.df.shape[1]} features")
        return self.df
    
    def explore_data(self) -> None:
        """Analyse exploratoire initiale"""
        if self.df is None:
            self.load_data()
            
        print("\n" + "="*70)
        print("EXPLORATION INITIALE DES DONNÉES")
        print("="*70)
        
        print(f"\n📊 Dimensions : {self.df.shape}")
        print(f"\n🔍 Types de données :\n{self.df.dtypes}")
        
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️ Valeurs manquantes :\n{missing[missing > 0]}")
        else:
            print("\n✅ Aucune valeur manquante")
        
        print(f"\n📈 Statistiques descriptives :\n{self.df.describe()}")
        print(f"\n👀 Premiers enregistrements :\n{self.df.head()}")
        
    def identify_features(self, target_col: str = "churn") -> Tuple[List[str], List[str]]:
        """Identifier les features numériques et catégorielles"""
        if self.df is None:
            self.load_data()
        
        self.target = target_col
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in self.numerical_features:
            self.numerical_features.remove(target_col)
        if target_col in self.categorical_features:
            self.categorical_features.remove(target_col)
        
        print(f"\n�� Features numériques ({len(self.numerical_features)}) : {self.numerical_features[:5]}...")
        print(f"📊 Features catégorielles ({len(self.categorical_features)}) : {self.categorical_features}")
        print(f"🎯 Variable cible : {target_col}")
        
        return self.numerical_features, self.categorical_features
    
    def check_class_balance(self, target_col: str = "churn") -> dict:
        """Vérifier l'équilibre des classes"""
        if self.df is None:
            self.load_data()
        
        distribution = self.df[target_col].value_counts(normalize=True) * 100
        print(f"\n⚖️ Distribution des classes ({target_col}) :")
        for class_val, pct in distribution.items():
            print(f"   Classe {class_val}: {pct:.2f}%")
        
        return distribution.to_dict()
    
    def get_X_y(self, target_col: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
        """Obtenir X (features) et y (target)"""
        if self.df is None:
            self.load_data()
        
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        return X, y
