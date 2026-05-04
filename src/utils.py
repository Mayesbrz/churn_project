"""
Fonctions utilitaires du projet
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any


def create_project_dirs():
    """Créer les répertoires nécessaires s'ils n'existent pas"""
    dirs = ["models", "reports", "data", "logs"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Répertoires de projet créés")


def save_model(model, filename: str, directory: str = "models"):
    """Sauvegarder un modèle entraîné"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.joblib")
    joblib.dump(model, filepath)
    print(f"✅ Modèle sauvegardé : {filepath}")
    return filepath


def load_model(filename: str, directory: str = "models"):
    """Charger un modèle entraîné"""
    filepath = os.path.join(directory, f"{filename}.joblib")
    model = joblib.load(filepath)
    print(f"✅ Modèle chargé : {filepath}")
    return model


def set_random_seed(seed: int = 42):
    """Fixer les graines aléatoires pour la reproductibilité"""
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass
    print(f"✅ Graines aléatoires fixées : {seed}")


if __name__ == "__main__":
    create_project_dirs()
    set_random_seed()
