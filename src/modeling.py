"""
Module de modélisation
"""

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class ModelTrainer:
    """Classe pour entraîner et gérer plusieurs modèles."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}

    def define_models(self) -> Dict[str, Any]:
        """Définir les 4 modèles exigés par le cahier des charges."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(96, 48, 24),
                alpha=0.001,
                max_iter=250,
                early_stopping=True,
                random_state=self.random_state,
            ),
        }

        print(f"✅ {len(self.models)} modèles définis : {list(self.models.keys())}")
        return self.models

    def train_models(self, X_train, y_train) -> Dict[str, Any]:
        """Entraîner tous les modèles."""
        if not self.models:
            self.define_models()

        for name, model in self.models.items():
            print(f"\n🔄 Entraînement de {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            print(f"✅ {name} entraîné")

        return self.trained_models
