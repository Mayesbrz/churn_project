"""
Script principal du projet - Point d'entrée
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au PATH
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import create_project_dirs, set_random_seed
from src.data_processing import DataProcessor, FeatureEngineer


def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("🚀 PROJET DE PRÉDICTION DU CHURN CLIENT - EFREI 2025-26")
    print("="*70)
    
    # 1. Créer les répertoires
    create_project_dirs()
    
    # 2. Fixer les graines aléatoires
    set_random_seed(42)
    
    # 3. Charger et explorer les données
    print("\n📥 Chargement des données...")
    processor = DataProcessor("data/customer_churn_business_dataset.csv")
    df = processor.load_data()
    
    # 4. Exploration initiale
    print("\n🔍 Analyse exploratoire...")
    processor.explore_data()
    
    # 5. Identifier les features
    print("\n📊 Identification des features...")
    numerical_features, categorical_features = processor.identify_features("churn")
    
    # 6. Vérifier l'équilibre des classes
    print("\n⚖️ Vérification du déséquilibre des classes...")
    processor.check_class_balance("churn")
    
    # 7. Obtenir X et y
    print("\n📋 Préparation de X et y...")
    X, y = processor.get_X_y("churn")
    
    print("\n" + "="*70)
    print("✅ PHASE D'INITIALISATION TERMINÉE")
    print("="*70)
    print("\n📝 Prochaines étapes :")
    print("   1. Exécuter EDA complète : jupyter notebook notebooks/01_eda.ipynb")
    print("   2. Entraîner les modèles : jupyter notebook notebooks/02_modeling.ipynb")
    print("   3. Lancer le dashboard : streamlit run app/dashboard.py")
    print("   4. Lancer l'API (optionnel) : uvicorn app.api:app --reload")


if __name__ == "__main__":
    main()
