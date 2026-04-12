"""
Module de préparation et nettoyage des données
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict


class DataProcessor:
    """Classe pour traiter et préparer les données"""
    
    def __init__(self, csv_path: str):
        """
        Initialiser le processeur de données
        
        Args:
            csv_path: Chemin vers le fichier CSV
        """
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
        
        # Info générales
        print(f"\n📊 Dimensions : {self.df.shape}")
        print(f"\n🔍 Types de données :\n{self.df.dtypes}")
        
        # Valeurs manquantes
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️ Valeurs manquantes :\n{missing[missing > 0]}")
        else:
            print("\n✅ Aucune valeur manquante")
        
        # Statistiques descriptives
        print(f"\n📈 Statistiques descriptives :\n{self.df.describe()}")
        
        # Aperçu des données
        print(f"\n👀 Premiers enregistrements :\n{self.df.head()}")
        
    def identify_features(self, target_col: str = "churn") -> Tuple[List[str], List[str]]:
        """
        Identifier les features numériques et catégorielles
        
        Args:
            target_col: Nom de la colonne cible
            
        Returns:
            Tuple de (features_numeriques, features_categoriques)
        """
        if self.df is None:
            self.load_data()
        
        self.target = target_col
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Retirer la cible
        if target_col in self.numerical_features:
            self.numerical_features.remove(target_col)
        if target_col in self.categorical_features:
            self.categorical_features.remove(target_col)
        
        print(f"\n📊 Features numériques ({len(self.numerical_features)}) : {self.numerical_features[:5]}...")
        print(f"📊 Features catégorielles ({len(self.categorical_features)}) : {self.categorical_features}")
        print(f"🎯 Variable cible : {target_col}")
        
        return self.numerical_features, self.categorical_features
    
    def check_class_balance(self, target_col: str = "churn") -> Dict[str, float]:
        """Vérifier l'équilibre des classes"""
        if self.df is None:
            self.load_data()
        
        distribution = self.df[target_col].value_counts(normalize=True) * 100
        print(f"\n⚖️ Distribution des classes ({target_col}) :")
        for class_val, pct in distribution.items():
            print(f"   Classe {class_val}: {pct:.2f}%")
        
        return distribution.to_dict()
    
    def get_X_y(self, target_col: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Obtenir X (features) et y (target)
        
        Args:
            target_col: Nom de la colonne cible
            
        Returns:
            Tuple de (X, y)
        """
        if self.df is None:
            self.load_data()
        
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        return X, y


class FeatureEngineer:
    """Classe pour l'ingénierie de features"""
    
    @staticmethod
    def create_preprocessing_pipeline(numerical_features: List[str], 
                                     categorical_features: List[str]) -> ColumnTransformer:
        """
        Créer un pipeline de prétraitement
        
        Args:
            numerical_features: Liste des features numériques
            categorical_features: Liste des features catégorielles
            
        Returns:
            ColumnTransformer configuré
        """
        # Pipeline pour features numériques
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Pipeline pour features catégorielles
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combiner les transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        print("✅ Pipeline de prétraitement créé")
        return preprocessor


if __name__ == "__main__":
    # Test
    processor = DataProcessor("data/customer_churn_business_dataset.csv")
    processor.load_data()
    processor.explore_data()
    processor.identify_features()
    processor.check_class_balance()
