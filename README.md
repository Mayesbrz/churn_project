# 🎯 Système Intelligent Multi-Modèles pour la Rétention Client

Projet Data Science EFREI 2025-26 - Expert en Ingénierie de Données (RNCP36739)

## 📋 Description du Projet

Concevoir et développer une plateforme intelligente de rétention client capable de :
- **Prédire le risque de churn** (résiliation client)
- **Évaluer le revenu à risque**
- **Comparer plusieurs algorithmes** (ML classique + Deep Learning)
- **Fournir un dashboard décisionnel** interactif
- **Exposer un service via API REST** (optionnel)

## 📊 Dataset

- **Source** : Kaggle - Customer Churn Prediction Business Dataset
- **Taille** : 10,000 clients
- **Variables** : 40+ features (numériques + catégorielles)
- **Cible principal** : `churn` (0 = Fidèle, 1 = Résiliation)

## 🏗️ Structure du Projet

```
project_root/
├── data/
│   └── customer_churn_business_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Nettoyage & préparation
│   ├── modeling.py             # Entraînement des modèles
│   ├── evaluation.py           # Métriques & comparaisons
│   ├── explainability.py       # SHAP & Feature Importance
│   └── utils.py                # Fonctions utilitaires
├── models/
│   └── (modèles sérialisés .pkl/.joblib)
├── app/
│   ├── __init__.py
│   ├── dashboard.py            # Streamlit
│   └── api.py                  # FastAPI (optionnel)
├── reports/
│   └── (figures & résultats)
├── requirements.txt
├── .gitignore
├── README.md
└── main.py                     # Point d'entrée
```

## 🚀 Installation & Configuration

### 1. Cloner le projet
```bash
cd "/Users/amayasbariz/Documents/dossier sans titre/projet ds/churn"
```

### 2. Activer l'environnement virtuel
```bash
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📝 Étapes du Projet

### Étape 1 : EDA (Exploratory Data Analysis)
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Étape 2 : Modélisation
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

### Étape 3 : Dashboard Streamlit
```bash
streamlit run app/dashboard.py
```

### Étape 4 : API REST (optionnel)
```bash
uvicorn app.api:app --reload
```

## 🎓 Modèles à Implémenter

1. **Baseline Simple** : Régression Logistique
2. **Tree-Based** : Random Forest
3. **Gradient Boosting** : XGBoost ou LightGBM
4. **Deep Learning** : MLP (Keras/TensorFlow)

**Comparaison rigoureuse** via :
- Accuracy, Precision, Recall, F1, ROC-AUC (classification)
- Confusion Matrix
- Feature Importance + SHAP
- Cross-validation

## 📊 Dashboard Features

- ✅ Visualisation des distributions clients
- ✅ Analyse des facteurs de churn
- ✅ Comparaison des modèles
- ✅ Simulation de scénarios
- ✅ Prédictions en temps réel
- ✅ Importance des variables

## 📚 Ressources

- [Kaggle Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 👥 Auteurs

Projet collectif EFREI - M2 Data Engineering & AI

## 📅 Dates Importants

- **Démarrage** : Avril 2026
- **Deadline Moodle** : [À définir par l'enseignant]
- **Présentation** : [À définir par l'enseignant]

## ⚠️ Recommandations

- ✅ Commencer par une EDA approfondie
- ✅ Vérifier les valeurs manquantes et déséquilibre des classes
- ✅ Éviter le data leakage à tout prix
- ✅ Utiliser des pipelines scikit-learn
- ✅ Comparer rigoureusement (au moins 4 modèles)
- ✅ Implémenter SHAP pour l'explicabilité
- ✅ Versionner régulièrement avec Git

---

**Good Luck! 🚀**
